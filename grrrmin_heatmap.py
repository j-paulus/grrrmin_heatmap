#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# grrrmin_heatmap.py
#
# Plot a heatmap of GPS routes
# - saved into GarminDB <https://github.com/tcgoetz/GarminDB> SQLite database,
# or
# - saved into .fit, .gpx (1.0, 1.1), .tcx files.
#
#
# Usage examples:
#  # all steps activities from year 2020, figure limited to north Nuremberg:
#  python grrrmin_heatmap.py --bounding_box 11.16 49.524 11.015 49.452 --year 2020 --zoom_level 15 --sport steps
#
#  # steps activities from all time, figure centered in Feucht:
#  python grrrmin_heatmap.py --sport steps --zoom_level 15 --start_center 49.383 11.2185 --start_max_dist 100.0 --bb_percentile 0.0
#
#  # all cycling activities from of all times using satellite image background:
#  python grrrmin_heatmap.py --sport cycling --basemap_provider Esri.WorldImagery
#
#  # all activities in the given directory and all sub-directories
#  python grrrmin_heatmap.py --input_dir ../activities/ --bounding_box 11.265 49.402 11.195 49.365
#
#
# The real bounding box of the resulting map is guided by the given (or determined bounding box),
# but in the end decided by the contextily library providing the map tiles.
#
#
# (c) Jouni Paulus, 28.12.2020
#
# License: BSD-3-Clause
#
# Copyright 2020-2021 Jouni Paulus
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#

import sys
import sqlite3  # interface GarminDB
import datetime

import dateutil  # tcx time zone parsing

from contextlib import closing  # context manager with automatic closing for the db
from pathlib import Path  # home directory

import os  # file name handling
import glob  # listing files

import argparse  # command line handling

import numpy as np
import matplotlib  # colormap

import contextily as ctx  # basemap

# coordinate transforms
import pyproj
from pyproj.transformer import Transformer

from tqdm import tqdm  # progress bar

# plotting routines
from PIL import Image, ImageDraw

import fitparse  # .fit file support

import gpxpy  # .gpx file support
import gpxpy.gpx

import tcxparser  # .tcx file support

__version__ = '0.3.2'

geod_conv = pyproj.Geod(ellps='WGS84')


def get_activities_from_db(sport_name='steps', target_year=None, garmin_db=None, verbosity=1):
    """
    Load requested activities from GarminDB SQLite database.

    Parameters
    ----------
    sport_name : string in {'cycling', 'all', 'running', 'hiking', 'steps'}
        select a specific activity type
    target_year : list of int, None, optional
        specify a year from which the data should be plotted from
    garmin_db : string, None, optional
        alternative path to the SQLite DB
    verbosity : int, optional
        printout verbosity

    Returns
    -------
    list of lists of points : activities
    float : total distance in km
    """
    if garmin_db is None:
        # default path to GarminDB database file
        garmin_db = '{}/HealthData/DBs/garmin_activities.db'.format(Path.home())

    steps_template = r'''SELECT activities.activity_id, activities.name, activities.description, activities.start_time,
                         activities.stop_time, activities.elapsed_time, ROUND(activities.distance, 1)
                         FROM steps_activities
                         JOIN activities ON activities.activity_id = steps_activities.activity_id {act_filter}
                         ORDER BY activities.start_time ASC'''

    cycle_query = r'''SELECT activities.activity_id, activities.name, activities.description, activities.start_time,
                      activities.stop_time, activities.elapsed_time, ROUND(activities.distance, 1)
                      FROM activities
                      WHERE activities.sport == "cycling"
                      OR activities.sport == "Biking"
                      ORDER BY activities.start_time ASC'''

    all_activities_query = r'''SELECT activities.activity_id, activities.name, activities.description, activities.start_time,
                               activities.stop_time, activities.elapsed_time, ROUND(activities.distance, 1)
                               FROM activities
                               ORDER BY activities.start_time ASC'''

    if sport_name == 'cycling':
        act_query = cycle_query

    elif sport_name == 'all':
        act_query = all_activities_query

    elif sport_name == 'running':
        act_filter = 'WHERE Activities.sport == "running"'
        act_query = steps_template.format(act_filter=act_filter)

    elif sport_name == 'hiking':
        act_filter = 'WHERE Activities.sport == "hiking"'
        act_query = steps_template.format(act_filter=act_filter)

    elif sport_name == 'walking':
        act_filter = 'WHERE Activities.sport == "walking"'
        act_query = steps_template.format(act_filter=act_filter)

    else:  # sport_name == 'steps':
        act_filter = ''
        pic_tag = 'steps'
        act_query = steps_template.format(act_filter=act_filter)

    with closing(sqlite3.connect(garmin_db)) as db_conn:  # this closes the connection after finishing
        c = db_conn.cursor()

        # get all activities
        act_id_list = []  # list to store activity_id keys
        act_time_list = []
        act_dist_list = []
        c.execute(act_query)
        for one_row in c:
            act_id = one_row[0]
            act_date = one_row[3]
            act_dist = one_row[6]

            act_id_list.append(act_id)
            act_time_list.append(datetime.datetime.strptime(act_date, '%Y-%m-%d %H:%M:%S.%f'))
            act_dist_list.append(act_dist)

        # for each activity in the list, fetch the points
        act_ite = tqdm(zip(act_id_list, act_time_list, act_dist_list), total=len(act_id_list), disable=(verbosity == 0))
        act_ite.set_description('Activities...')
        all_paths = []
        total_dist = 0.0
        for act_id, act_time, act_dist in act_ite:
            # are we within the given time range
            if (target_year is None) or (len(target_year) == 0) or (act_time.year in target_year):
                total_dist += act_dist
                c.execute('SELECT activity_records.activity_id, activity_records.timestamp, activity_records.position_lat, activity_records.position_long FROM activity_records WHERE activity_records.activity_id = (?) ORDER BY activity_records.timestamp DESC', (act_id,))

                # collect all points of this activity into a list
                this_points = []
                for one_point in c:
                    this_lat = one_point[2]
                    this_lon = one_point[3]

                    if (this_lat is not None) and (this_lon is not None):
                        this_points.append((this_lat, this_lon))

                all_paths.append(this_points)

    return all_paths, total_dist


def get_activities_from_dir(path_str, target_year=None, verbosity=1):
    """
    Check recursively all files in the given directory and if they are
    .fit/.gpx/.tcx, load the activities from them. This may be somewhat slow.

    Parameters
    ----------
    path_str : string
        path from which all the files are checked. run recursively into all subdirectories
    target_year : list of int, None, optional
        list of target years for filtering the plotted activities
    verbosity : int, optional
        printout verbosity

    Returns
    -------
    list of lists of points : activities
    float : total distance in km
    """
    def semi2deg(x):
        """
        Convert "semicircle" units to decimal degrees.
        """
        return x * 180.0 / (2.0**31)

    # list all files
    all_files = glob.glob(os.path.join(path_str, '**/*.*'), recursive=True)

    # collected info
    all_activities = []
    total_dist = 0.0
    for one_name in tqdm(all_files,
                         total=len(all_files),
                         desc='Checking files',
                         unit=' files',
                         disable=(verbosity == 0)):
        full_name = os.path.join(path_str, one_name)

        # check for supported file extensions
        base_str, ext_str = os.path.splitext(full_name)
        if os.path.isfile(full_name) and (ext_str.lower() == '.fit'):
            # try to parse the .fit file
            try:
                fitfile = fitparse.FitFile(full_name)
                fitfile.parse()

                # retieve activity type, even though this is not used right now
                this_activity_type = None
                for sports in fitfile.get_messages('sport'):
                    this_activity_type = sports.get_value('sport')

                this_act = []
                act_dist = 0.0
                act_time = None
                # get all data messages that are of type record
                for one_rec in fitfile.get_messages('record'):
                    this_lat = one_rec.get_value('position_lat')
                    this_lon = one_rec.get_value('position_long')
                    this_dist = one_rec.get_value('distance')

                    # convert the coordinates from the semicircle to decimal degrees
                    if (this_lat is not None) and (this_lon is not None):
                        this_act.append((semi2deg(this_lat), semi2deg(this_lon)))

                    if this_dist is not None:
                        act_dist = this_dist

                    if act_time is None:
                        this_time = one_rec.get_value('timestamp')
                        if this_time is not None:
                            act_time = this_time

                # activity date -based filtering
                if (act_time is None) or (target_year is None) or (len(target_year) == 0) or (act_time.year in target_year):
                    all_activities.append(this_act)
                    total_dist += act_dist

            except fitparse.FitParseError as e:
                if verbosity > 0:
                    print('ERROR: Could not parse file "{}". Error: {}'.format(full_name, e))

        elif os.path.isfile(full_name) and (ext_str.lower() == '.gpx'):
            # try to parse the .gpx file
            with open(full_name, 'r') as gpx_file:
                gpx = gpxpy.parse(gpx_file)

                act_dist = 0.0
                act_time = None
                this_act = []
                for one_track in gpx.tracks:
                    act_dist = one_track.length_3d()  # 3D length in meters
                    act_time = one_track.get_time_bounds()[0]  # starting time

                    for tmp_data in one_track.walk():
                        one_point = tmp_data[0]
                        this_act.append((one_point.latitude, one_point.longitude))

                # activity date -based filtering
                if (act_time is None) or (target_year is None) or (len(target_year) == 0) or (act_time.year in target_year):
                    all_activities.append(this_act)
                    total_dist += act_dist

                act_dist = 0.0
                act_time = None
                this_act = []
                for one_route in gpx.routes:
                    act_dist = one_route.length_3d()
                    act_time = one_track.get_time_bounds()[0]  # starting time
                    for tmp_data in one_route.walk():
                        one_point = tmp_data[0]
                        this_act.append((one_point.latitude, one_point.longitude))

                # activity date -based filtering
                if (act_time is None) or (target_year is None) or (len(target_year) == 0) or (act_time.year in target_year):
                    all_activities.append(this_act)
                    total_dist += act_dist

        elif os.path.isfile(full_name) and (ext_str.lower() == '.tcx'):
            # try to parse the .tcx file
            tcx_data = tcxparser.TCXParser(full_name)

            this_activity_type = tcx_data.activity_type  # could be used for filtering activity type, but not done now

            act_dist = tcx_data.distance
            act_time = dateutil.parser.isoparse(tcx_data.started_at)
            this_act = tcx_data.position_values()  # list of (lat, lon) tuples

            # activity date -based filtering
            if (act_time is None) or (target_year is None) or (len(target_year) == 0) or (act_time.year in target_year):
                all_activities.append(this_act)
                total_dist += act_dist

    return all_activities, total_dist / 1000.0


def get_year_range(year_list):
    """
    Transform a year list into a textual representation shortening consecutive values.

    E.g., get_year_range([1999, 2000, 2001, 2004]) => '1999-2001_2004

    Adapted from <https://stackoverflow.com/a/48106843>

    Parameters
    ----------
    year_list : list of int

    Returns
    -------
    string

    """
    if (year_list is None) or (len(year_list) == 0):
        return 'all'
    elif len(year_list) == 1:
        return '{}'.format(year_list[0])
    else:
        nums = sorted(set(year_list))
        gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
        edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
        range_list = [(s, e+1) for s, e in zip(edges, edges)]

        out_str = ''
        for r_idx, one_range in enumerate(range_list):
            if r_idx == 0:
                out_str = '{}'.format(one_range[0])
            else:
                out_str = '_{}'.format(one_range[0])

            if one_range[0] != one_range[1]:
                out_str = '{}-{}'.format(out_str, one_range[1])

        return out_str


# basemap
def get_basemap_provider(provider_str):
    """
    Take a string representing the desired Contextily basemap provider,
    e.g., "Esri.WorldImagery", parse it, and provide the provider
    instance contextily.providers.Esri.WorldImagery, if found.
    """
    provider_parts = provider_str.split('.')

    b_provider = None
    for one_part in provider_parts:
        if b_provider is None:
            b_provider = getattr(ctx.providers, one_part, None)
        else:
            b_provider = getattr(b_provider, one_part, None)

        if b_provider is None:
            print('ERROR: Unsupported basemap provider "{}" requested.'.format(provider_str))

    return b_provider


def run_plotting(args):
    """
    Main plotting function

    Parameters
    ----------
    args : Namespace with field fields:
        bounding_box : None, 4-list/tuple of floats
            define image bounding box in decimal WGS84: n, e, s, w. None for automatic
        bb_percentile : float
            in range 0..1, when determining the bounding box from the data, use bb_percentile
            and 1-bb_percentile quantiles as the limits to filter outliers. 0 for min/max
        zoom_level: None, int
            None for automatic zoom level, otherwise the given int
        sport : string in 'cycling, 'running', 'hiking', 'walking', 'steps'
            activity type to plot
        basemap_provider : string, None
            Contextily basemap provider name string
        img_width : int
            if basemap_provider == None, width of the blank image
        track_colormap : string
            matplotlib colormap name
        max_point_dist : float, None
            if not None and consecutive track points are further than
            this, the track is split into two
        year : list of ints
            list of years to plot, e.g., [2019, 2020]
        do_gif : bool
            if True, create a frame-per-activity animation
        fps : float
            FPS of the created animation
        start_center : None, 2-tuple of float
            only tracks starting near ("start_max_dist") this point (lat, lon, in decimal WGS84) are plotted
        start_max_dist : float
            only track starting within this radius (in meters) from "start_center" are plotted
        verbosity : int
            progree printout verbosity level
    """
    do_start_filter = (args.start_center is not None) and (args.start_max_dist is not None)

    if args.bounding_box is not None:  # n, e, s, w
        max_lon, max_lat, min_lon, min_lat = args.bounding_box

    else:
        # determine from data
        all_lat = []
        all_lon = []

    if args.zoom_level is None:
        zoom_level = 'auto'
    else:
        zoom_level = args.zoom_level

    if (args.sport is None) or (args.sport.lower() not in {'cycling', 'all', 'running', 'walking'}):
        pic_tag = 'steps'
    else:
        pic_tag = args.sport

    if args.input_dir is not None:
        all_activities, total_dist = get_activities_from_dir(args.input_dir, target_year=args.year, verbosity=args.verbosity)
    else:
        all_activities, total_dist = get_activities_from_db(sport_name=args.sport, target_year=args.year, garmin_db=None, verbosity=args.verbosity)

    all_paths = []
    all_lat = []
    all_lon = []
    for act_idx, this_points in tqdm(enumerate(all_activities),
                                     desc='Filtering points',
                                     unit=' activities',
                                     disable=(args.verbosity == 0)):
        # create a path from the points
        if len(this_points) > 1:
            path_points = []

            # distance-based filtering
            if args.max_point_dist is not None:
                prev_point = (None, None)
                for point_idx, one_point in enumerate(this_points):
                    if do_start_filter and (point_idx == 0):
                        # starting point -based filtering active and first point in activity
                        start_az1, start_az2, start_dist = geod_conv.inv(args.start_center[1], args.start_center[0],
                                                                         one_point[1], one_point[0])
                        if start_dist > args.start_max_dist:
                            # too far from the target starting location => skip the entire activity
                            if args.verbosity > 1:
                                print('WARNING: Activity starting location {:.1f} m (>{:.1f} m) from the defined start location, skipping.'.format(start_dist,
                                                                                                                                                   args.start_max_dist))
                            break

                    if args.bounding_box is None:
                        all_lat.append(one_point[0])
                        all_lon.append(one_point[1])

                    if prev_point[0] is None:
                        path_points.append(one_point)

                    else:
                        # long/lat pairs to azimuths and distance in meters
                        az1, az2, dist = geod_conv.inv(prev_point[1], prev_point[0], one_point[1], one_point[0])

                        if dist < args.max_point_dist:
                            path_points.append(one_point)
                        else:
                            # too large distance between two points => discard
                            if args.verbosity > 1:
                                print('WARNING: Track segment detached due to distance {:.1f}m exceeding the threshold of {:.1f}m.'.format(dist, args.max_point_dist))

                            # start a new path
                            all_paths.append(path_points)
                            path_points = [one_point]

                    prev_point = one_point

            else:
                # no distance filtering => use as-is
                path_points = this_points.copy()
                if args.bounding_box is None:
                    for one_point in this_points:
                        all_lat.append(one_point[0])
                        all_lon.append(one_point[1])

            all_paths.append(path_points)

    if len(all_paths) == 0:
        if args.verbosity > 0:
            print('WARNING: No mathing activities found.')

        sys.exit()

    if args.bounding_box is None:
        lat_array = np.array(all_lat)
        lon_array = np.array(all_lon)
        lat_quants = np.quantile(lat_array, (args.bb_percentile, 1.0-args.bb_percentile))
        lon_quants = np.quantile(lon_array, (args.bb_percentile, 1.0-args.bb_percentile))

        min_lat = lat_quants[0]
        max_lat = lat_quants[1]
        min_lon = lon_quants[0]
        max_lon = lon_quants[1]

    if args.verbosity > 0:
        print('INFO: Total activity distance: {:.2f}km'.format(total_dist))
        print('INFO: Using lat range: {:.3f} - {:.3f}, and lon range: {:.3f} - {:.3f}.'.format(min_lat, max_lat, min_lon, max_lon))

    if zoom_level == 'auto':
        # the default zoom level
        zoom_level = ctx.tile._calculate_zoom(w=min_lon, s=min_lat, e=max_lon, n=max_lat)
        if args.verbosity > 1:
            print('INFO: Using zoom level {}.'.format(zoom_level))

    # from WGS84 to Spherical Mercator used by contextily
    crs_trans = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)

    # fetch the basemap including the specified bounding box region
    if args.verbosity > 0:
        print('INFO: Fetching basemap...')

    if args.basemap_provider is None:
        # no map, but blank background
        # transformer input: (x,y) -> (lon, lat)
        min_point = crs_trans.transform(min_lon, min_lat)
        max_point = crs_trans.transform(max_lon, max_lat)
        imshow_extent = [min_point[0], max_point[0], min_point[1], max_point[1]]  #  [minX, maxX, minY, maxY]

        range_lon = max_point[0] - min_point[0]
        range_lat = max_point[1] - min_point[1]

        img_height = args.img_width
        img_width = int(img_height / float(range_lat) * range_lon)

        basemap_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        basemap_attr = None

    else:
        basemap_src = get_basemap_provider(args.basemap_provider)
        basemap_img, imshow_extent = ctx.bounds2img(w=min_lon, s=min_lat, e=max_lon, n=max_lat,
                                                    zoom=zoom_level, ll=True, source=basemap_src)
        basemap_attr = basemap_src['attribution']

    if args.track_colormap is None:
        # two default colormaps
        if (args.basemap_provider is None) or (args.basemap_provider == 'CartoDB.DarkMatter'):
            track_cmap = 'plasma'  # ok with CartoDB.DarkMatter
        else:
            track_cmap = 'autumn'  # works ok with Esri.WorldImagery, use also for others
    else:
        # user-defined colormap
        track_cmap = args.track_colormap

    # from RGB to RGBA
    zero_alpha = 255 * np.ones((basemap_img.shape[0], basemap_img.shape[1], 1), dtype=np.uint8)
    if basemap_img.shape[2] == 3:
        # add alpha channel
        basemap_image = Image.fromarray(np.concatenate((basemap_img, zero_alpha), axis=-1))
    else:
        # replace alpha channel
        basemap_img[:, :, -1] = zero_alpha[:, :, 0]
        basemap_image = Image.fromarray(basemap_img)

    # add attribution
    basemap_draw = ImageDraw.Draw(basemap_image)
    basemap_draw.text((5, 5), 'Created with grrrmin_heatmap.py' + (basemap_attr is not None)*'\nUsing Contextily basemap:\n{}'.format(basemap_attr))

    # a function to transform geographical coordinates to PIL coordinates: (0,0) upper left corner. (x, y)
    def coord_to_pixel(lat, lon):
        new_lat = (1.0 - (lat - imshow_extent[2]) / (imshow_extent[3] - imshow_extent[2])) * basemap_image.height
        new_lon = (lon - imshow_extent[0]) / (imshow_extent[1] - imshow_extent[0]) * basemap_image.width
        return new_lat, new_lon

    # transform point from WGS84 to pixels
    for path_idx, one_path in enumerate(all_paths):
        for point_idx, one_point in enumerate(one_path):
            # transform the points into coordinate system used by the basemap
            # transformer input: (x,y) -> (lon, lat)
            one_point = crs_trans.transform(one_point[1], one_point[0])
            one_path[point_idx] = coord_to_pixel(one_point[1], one_point[0])[::-1]

        all_paths[path_idx] = one_path

    ## plotting
    # loop through paths one-by-one
    path_sum = None
    n_paths = len(all_paths)
    if args.do_gif:
        all_frames = []

    # the paths are plotted on this dummy image
    path_image = Image.new('RGBA', (basemap_image.width, basemap_image.height), color=(0, 0, 0, 0))  # (width, height)
    path_canvas = ImageDraw.Draw(path_image)
    h = path_image.height
    w = path_image.width

    # plot each path
    plot_ite = tqdm(enumerate(all_paths), total=n_paths, unit=' activities', disable=(args.verbosity == 0))
    plot_ite.set_description('Plotting...')
    year_str = get_year_range(args.year)
    out_name_base = 'grrrmin_heatmap_{}_{}'.format(pic_tag, year_str)
    for path_idx, one_path in plot_ite:
        # PIL.Image approach
        path_canvas.line(xy=one_path, fill='black', width=args.line_width, joint='curve')
        img = np.array(path_image)

        if path_sum is None:
            path_sum = np.zeros((h, w), dtype=np.float32)

        # binary mapping from alpha channel
        path_sum[img[:, :, 3] > 128] += 1.0

        # erase the path
        draw = ImageDraw.Draw(path_image)
        draw.rectangle([(0,0), path_image.size], fill=(0, 0, 0, 0))

        if args.do_gif or (path_idx == n_paths - 1):
            comp_sum = np.log2(1.0 + 1.0*path_sum)

            # from a single float matrix to RGBA
            comp_rgba = matplotlib.cm.ScalarMappable(norm=None, cmap=track_cmap).to_rgba(comp_sum, alpha=None, bytes=True, norm=True)  # to uint8

            # set alpha channel to 0 if no path occupies the pixel
            comp_rgba[..., 3] = 255
            comp_rgba[path_sum < 1.0, 3] = 0
            path_sum_image = Image.fromarray(comp_rgba)

            # overlay sum path on the basemap
            out_image = Image.alpha_composite(basemap_image, path_sum_image)

            if args.do_gif:
                all_frames.append(out_image)
            else:
                out_file_name = '{}.png'.format(out_name_base)
                out_image.save(out_file_name)

    if args.do_gif:
        all_frames[0].save('{}.gif'.format(out_name_base),
                           save_all=True,
                           append_images=all_frames[1:],
                           loop=False,
                           duration=1.0 / args.fps)


##
def list_basemap_providers():
    """
    Print all map tile providers from Contextily
    """
    print('INFO: Supported background map tile providers:')
    prov_keys = ctx.providers.keys()
    for prov in prov_keys:
        p2 = getattr(ctx.providers, prov).keys()
        if 'url' in p2:
            print(prov)
        else:
            for k in p2:
                print('{}.{}'.format(prov, k))


##
def main(argv):
    """
    Top level input argument parsing.

    Parameters
    ----------
    argv : list of string

    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--sport',
                           action='store',
                           type=str,
                           default='all',
                           choices=('steps', 'running', 'walking', 'hiking', 'cycling', 'all'),
                           help='sport type filter')

    argparser.add_argument('--year',
                           action='store',
                           type=int,
                           default=[],
                           nargs='+',
                           help='years to plot')

    argparser.add_argument('--bounding_box',
                           action='store',
                           type=float,
                           default=None,
                           nargs=4,
                           help='output image bounding box in decimal WGS84: n, e, s, w')

    argparser.add_argument('--start_center',
                           action='store',
                           type=float,
                           default=None,
                           nargs=2,
                           help='only tracks starting near ("start_max_dist") this point (lat, lon, in decimal WGS84) are plotted')

    argparser.add_argument('--start_max_dist',
                           action='store',
                           type=float,
                           default=500.0,
                           help='only tracks starting within this radius (in meters) from "start_center" are plotted')

    argparser.add_argument('--bb_percentile',
                           action='store',
                           type=float,
                           default=0.01,
                           help='when determining bounding box from data use percentiles '
                                'bb_percentile and 1-bb_percentile. range: 0..1. to use '
                                'min/max, set to 0')

    argparser.add_argument('--zoom_level',
                           action='store',
                           type=int,
                           default=None,
                           help='zoom level (larger value mean finer details). leave empty for automatic')

    argparser.add_argument('--line_width',
                           action='store',
                           type=int,
                           default=3,
                           help='plotting line width in pixels')

    argparser.add_argument('--max_point_dist',
                           action='store',
                           type=float,
                           default=200.0,
                           help='meters or None. for filtering missing data. if distance between consecutive points is larger, the line is broken into two')

    argparser.add_argument('--do_gif',
                           action='store_true',
                           help='save as frame-per-activity animation')

    argparser.add_argument('--fps',
                           action='store',
                           type=float,
                           default=12,
                           help='animation speed as FPS')

    argparser.add_argument('--basemap_provider',
                           action='store',
                           type=str,
                           default='CartoDB.DarkMatter',
                           help='Contextily basemap provider string, e.g., "CartoDB.DarkMatter, Esri.WorldImagery", "None" (for blank)')

    argparser.add_argument('--img_width',
                           action='store',
                           type=int,
                           default=1080,
                           help='when not using a background map image width in pixels (height is computed from data)')

    argparser.add_argument('--track_colormap',
                           action='store',
                           type=str,
                           default=None,
                           help='(matplotlib) colormap to use for track plotting')

    argparser.add_argument('--list_providers',
                           action='store_true',
                           help='list basemap tile providers and exit')

    argparser.add_argument('--input_dir',
                           action='store',
                           type=str,
                           default=None,
                           help='directory-based data input: load all .fit and .gpx files here and in sub-directories')

    argparser.add_argument('--verbosity',
                           action='store',
                           type=int,
                           default=1,
                           help='message output verbosity. 0: silent, 1: default, 2: more info')

    args = argparser.parse_args(argv)
    if args.basemap_provider.lower() == 'None'.lower():
        args.basemap_provider = None

    if args.list_providers:
        list_basemap_providers()
    else:
        run_plotting(args)


## entry point
if __name__ == '__main__':
    main(sys.argv[1:])
