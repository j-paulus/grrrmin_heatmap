#!/usr/bin/python3
# -*- coding: utf-8 -*-
# 
# grrrmin_heatmap.py
#
# Plot a heatmap of GPS routes saved into GarminDB <https://github.com/tcgoetz/GarminDB>
# SQLite database.
#
# Usage examples:
#  # all steps activities from year 2020, figure limited to north Nuremberg:
#  python grrrmin_heatmap.py --bounding_box 11.16 49.524 11.015 49.452 --year 2020 --zoom_level 15 --sport steps
#
#  # all cycling activities from of all times using satellite image background:
#  python grrrmin_heatmap.py --sport cycling --basemap_provider Esri.WorldImagery
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

import numpy as np
import matplotlib  # colormap

import contextily as ctx  # basemap

# coordinate transforms
import pyproj
from pyproj.transformer import Transformer

from tqdm import tqdm  # progress bar

# plotting routines
from PIL import Image, ImageDraw

from pathlib import Path  # home directory

import argparse  # command line handling

__version__ = 0.22

# default path to GarminDB database file
garmin_db = '{}/HealthData/DBs/garmin_activities.db'.format(Path.home())

geod_conv = pyproj.Geod(ellps='WGS84')

steps_template = 'SELECT activities.activity_id, activities.name, activities.description, activities.start_time, activities.stop_time, activities.elapsed_time, ROUND(activities.distance, 1) '\
                 'FROM steps_activities JOIN activities ON activities.activity_id = steps_activities.activity_id {act_filter} ORDER BY activities.start_time ASC'

cycle_query = 'SELECT activities.activity_id, activities.name, activities.description, activities.start_time, activities.stop_time, activities.elapsed_time, ROUND(activities.distance, 1) ' \
              'FROM activities WHERE activities.sport == "cycling" OR activities.sport == "Biking" ORDER BY activities.start_time ASC'

all_activities_query = 'SELECT activities.activity_id, activities.name, activities.description, activities.start_time, activities.stop_time, activities.elapsed_time, ROUND(activities.distance, 1) ' \
                       'FROM activities ORDER BY activities.start_time ASC'


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


# main plotting function
def run_plotting(args):
    """
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

    """
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

    pic_tag = args.sport
    if args.sport == 'cycling':
        act_query = cycle_query
        
    elif args.sport == 'all':
        act_query = all_activities_query

    elif args.sport == 'running':
        act_filter = 'WHERE Activities.sport == "running"'
        act_query = steps_template.format(act_filter=act_filter)

    elif args.sport == 'hiking':
        act_filter = 'WHERE Activities.sport == "hiking"'
        act_query = steps_template.format(act_filter=act_filter)

    elif args.sport == 'walking':
        act_filter = 'WHERE Activities.sport == "walking"'
        act_query = steps_template.format(act_filter=act_filter)

    else:  # args.sport == 'steps':
        act_filter = ''
        pic_tag = 'steps'
        act_query = steps_template.format(act_filter=act_filter)

    db_conn = sqlite3.connect(garmin_db)
    c = db_conn.cursor()

    # get all "steps" activities
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
    act_ite = tqdm(zip(act_id_list, act_time_list, act_dist_list), total=len(act_id_list))
    act_ite.set_description('Activities...')
    all_paths = []
    total_dist = 0.0
    for act_id, act_time, act_dist in act_ite:
        if (len(args.year) == 0) or (act_time.year in args.year):
            total_dist += act_dist
            c.execute('SELECT activity_records.activity_id, activity_records.timestamp, activity_records.position_lat, activity_records.position_long FROM activity_records WHERE activity_records.activity_id = (?) ORDER BY activity_records.timestamp DESC', (act_id,))

            # collect all points of this activity into a list
            this_points = []
            for one_point in c:
                #print(one_point)
                this_lat = one_point[2]
                this_lon = one_point[3]

                if (this_lat is not None) and (this_lon is not None):
                    this_points.append((this_lat, this_lon))
                    
                    if args.bounding_box is None:
                        # store for determining bounding box from data
                        all_lat.append(this_lat)
                        all_lon.append(this_lon)

            # create a path from the points
            if len(this_points) > 1:
                path_points = []

                # distance-based filtering
                if args.max_point_dist is not None:
                    prev_point = (None, None)
                    for one_point in this_points:
                        if prev_point[0] is None:
                            path_points.append(one_point)

                        else:
                            # long/lat pairs to azimuths and distance in meters
                            az1, az2, dist = geod_conv.inv(prev_point[1], prev_point[0], one_point[1], one_point[0])  

                            if dist < args.max_point_dist:
                                path_points.append(one_point)
                            else:
                                # too large distance between two points => discard
                                print('WARNING: Track segment detached due to distance {:.1f}m exceeding the threshold of {:.1f}m.'.format(dist, args.max_point_dist))
                                # start a new path
                                all_paths.append(path_points)
                                path_points = [one_point]

                        prev_point = one_point

                else:
                    # no distance filtering => use as-is
                    path_points = this_points.copy()

                all_paths.append(path_points)

    if args.bounding_box is None:
        lat_array = np.array(all_lat)
        lon_array = np.array(all_lon)
        lat_quants = np.quantile(lat_array, (args.bb_percentile, 1.0-args.bb_percentile))
        lon_quants = np.quantile(lon_array, (args.bb_percentile, 1.0-args.bb_percentile))

        min_lat = lat_quants[0]
        max_lat = lat_quants[1]
        min_lon = lon_quants[0]
        max_lon = lon_quants[1]

    db_conn.close()

    print('INFO: Total activity distance: {:.2f}km'.format(total_dist))
    print('INFO: Using lat range: {:.3f} - {:.3f}, and lon range: {:.3f} - {:.3f}.'.format(min_lat, max_lat, min_lon, max_lon))

    if zoom_level == 'auto':
        # the default zoom level
        zoom_level = ctx.tile._calculate_zoom(w=min_lon, s=min_lat, e=max_lon, n=max_lat)
        print('INFO: Using zoom level {}.'.format(zoom_level))

    # from WGS84 to Spherical Mercator used by contextily
    crs_trans = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)  
    
    # fetch the basemap including the specified bounding box region
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
    basemap_image = Image.fromarray(np.concatenate((basemap_img, zero_alpha), axis=-1))

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
    plot_ite = tqdm(enumerate(all_paths), total=n_paths)
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
        all_frames[0].save('{}.gif'.format(out_name_base), save_all=True, append_images=all_frames[1:], 
                        loop=False, duration=1.0 / args.fps)


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
                           help='when not using a background map image width in pixels (height is computes from data)')

    argparser.add_argument('--track_colormap',
                           action='store',
                           type=str,
                           default=None,
                           help='(matplotlib) colormap to use for track plotting')

    argparser.add_argument('--list_providers',
                           action='store_true',
                           help='list basemap tile providers and exit')

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
