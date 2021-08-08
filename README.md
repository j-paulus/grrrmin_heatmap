# grrrmin_heatmap.py

A python script for creating a static heatmap-like visualization in PNG-format from activity location records stored into a [GarminDB](https://github.com/tcgoetz/GarminDB), or plain FIT, GPX, or TCX files (experimental support).

The main difference from a point-based heatmap is that in this visualization the line segments formed between each two points are considered in the sums. The main difference from [dérive](https://github.com/erik/derive) is (in addition to the use of Python) that the more crowded points do not only get more opaque, but also the color is changing.

## Usage
Command line options:
- `--sport {steps, running, walking, hiking, cycling, all}`
  - Select a specific sport type to be plotted.
  - Not available for directory-based plotting.
  - If omitted, everything is plotted.
- `--year <int> [int] [int]`
  - List of years to be plotted.
  - If omitted, everything is plotted.
- `--bounding_box <float> <float> <float> <float>`
  - Output image bounding box given in decimal WGS84: N E S W.
  - If omitted, bounding box is determined from the data.
- `--bb_percentile <float>`
  - When determining the bounding box from the data, this value determines the percentile in each direction to be discarded (assumed outliers).
  - To include all data, use 0.
- `--zoom_level <int>`
  - Image zoom level in OpenStreetMap-like [zoom levels](https://wiki.openstreetmap.org/wiki/Zoom_levels).
  - For automatic, leave undefined.
- `--line_width <int>`
  - Plotting line width in pixels.
- `--max_point_dist <float`
  - If the distance between two consecutive points is larger than this value (in meters), no line is drawn between them.
- `--do_gif`
  - Triggers creating a gif animation.
- `--fps <float>`
  - When creating an animation, animation speed in frames (activities) per second.
- `--basemap_provider <string>`
  - [Contextily](https://github.com/geopandas/contextily) basemap provider name string, e.g., CartoDB.DarkMatter, Esri.WorldImagery.
  - Use "None" for blank background.
- `--img_width <int>`
  - When not using a background map, image width in pixels. The height is computed from data.
- `--track_colormap <string>`
  - [Matplotlib colormap](https://matplotlib.org/stable/tutorials/colors/colormaps.html) to use for track plotting.
- `--list_providers`
  - List basemap tile providers and exit.
- `--input_dir <string>`
  - Directory-based data input instead of GarminDB.
  - Load all .fit and .gpx files here and in all sub-directories.
  - Activity type filtering is not supported.
- `--verbosity <int>`
  - Progress display verbosity (0: silent, 1: default, 2: verbose)
- `--start_center <float> <float>`
  - Only tracks starting near this point (lat, lon, in decimal WGS84) are plotted.
- `--start_max_dist <float>`
  - Only tracks starting within this radius (in meters) from "start_center" are plotted.


### Examples
- Plot all "steps" activities from the year 2020, figure limited to north Nuremberg:
    ```
    python grrrmin_heatmap.py --bounding_box 11.16 49.524 11.015 49.452 --year 2020 --zoom_level 15 --sport steps
    ```
- Plot all steps activities from all time, figure centered in Feucht:
    ```
    python grrrmin_heatmap.py --sport steps --zoom_level 15 --start_center 49.383 11.2185 --start_max_dist 100.0 --bb_percentile 0.0
    ```
- Plot all "cycling" activities from of all times using satellite image background:
    ```
    python grrrmin_heatmap.py --sport cycling --basemap_provider Esri.WorldImagery
    ```
- Plot all activities from the given directory and all its sub-directories:
    ```
    python grrrmin_heatmap.py --input_dir ../activities/ --bounding_box 11.265 49.402 11.195 49.365
    ```



## License
Copyright (c) Jouni Paulus, 2020-2021, available under [BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause) license.
