#!/usr/bin/env python3
"""Geospatial shapefile processing utilities.

This module provides utilities for processing shapefiles and GeoJSON files
in geospatial workflows, including batch creation, point clipping, buffering,
and format conversion.
"""

from __future__ import annotations

import fnmatch
import glob
import json
import os
import shutil
import time
from pathlib import Path
from typing import List, Set, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool
from random import uniform
from shapely.geometry import MultiPoint, Point
from zipfile import ZipFile


def create_geojson_batches(
    json_path: str,
    dest_path: str,
    start_batch: int = 2,
    end_batch: int = 78,
    step: int = 2,
) -> None:
    """Create grid batches for all GeoJSON files in the original directory.

    Args:
        json_path: Path to source GeoJSON directory.
        dest_path: Destination path for batched directories.
        start_batch: Starting batch number.
        end_batch: Ending batch number.
        step: Step size for batch numbers.
    """
    json_list: List[str] = []

    for batch_num in range(start_batch, end_batch, step):
        dir_name = f'grid_batch_{batch_num}'
        new_dir = f'{dest_path}{dir_name}'

        if os.path.exists(new_dir):
            for json_file in os.listdir(json_path):
                json_num = json_file.split('.')[0]
                lower_bound = (batch_num - 2) * 1000

                if lower_bound < int(json_num) <= batch_num * 1000:
                    json_list.append(json_file)
                    in_file_path = f'{json_path}{json_file}'
                    out_file_path = f'{new_dir}/{json_file}'

                    if (os.path.isfile(in_file_path) and
                        not os.path.isfile(out_file_path)):
                        shutil.copy(in_file_path, out_file_path)

        print(f'Copied: {batch_num}')


def collect_geojson_metadata(
    json_path: str,
    start_batch: int = 2,
    end_batch: int = 78,
    step: int = 2,
) -> Tuple[List[int], List[int], List[str], List[str]]:
    """Collect metadata from batched GeoJSON files.

    Args:
        json_path: Path to batched GeoJSON directories.
        start_batch: Starting batch number.
        end_batch: Ending batch number.
        step: Step size for batch numbers.

    Returns:
        Tuple of lists: counter_no, batch_no, geojson_no, licence_no.
    """
    counter = 0
    counter_no: List[int] = []
    batch_no: List[int] = []
    geojson_no: List[str] = []
    licence_no: List[str] = []

    for batch_num in range(start_batch, end_batch, step):
        dir_name = f'grid_batch_{batch_num}'
        new_dir = f'{json_path}{dir_name}'

        if os.path.exists(new_dir):
            for file in os.listdir(new_dir):
                if not file == '.ipynb_checkpoints':
                    filename = file.split('.')[0]
                    geojson_file = f'{new_dir}/{file}'

                    with open(geojson_file, 'r') as f:
                        counter += 1
                        data = json.load(f)
                        features = data['features'][0]['properties']['Licence']

                        counter_no.append(counter)
                        geojson_no.append(filename)
                        licence_no.append(features)
                        batch_no.append(batch_num)

    return counter_no, batch_no, geojson_no, licence_no


def create_reference_csv(
    counter_no: List[int],
    batch_no: List[int],
    geojson_no: List[str],
    licence_no: List[str],
    output_path: str,
) -> None:
    """Create a CSV file for reference storage.

    Args:
        counter_no: List of counter numbers.
        batch_no: List of batch numbers.
        geojson_no: List of geojson numbers.
        licence_no: List of licence numbers.
        output_path: Path to output CSV file.
    """
    df = pd.DataFrame(
        list(zip(counter_no, batch_no, geojson_no, licence_no)),
        columns=['counter_no', 'batch_no', 'geojson_no', 'licence_no']
    )
    df.to_csv(output_path, index=False)


def clip_points_to_polygons(
    json_file: str,
    points_path: str,
    poly_path: str,
    dest_dir: str,
    img_shp_csv: str,
) -> None:
    """Clip points to polygon boundaries.

    Args:
        json_file: Name of the GeoJSON file.
        points_path: Path to points GeoJSON file.
        poly_path: Path to polygon GeoJSON directory.
        dest_dir: Destination directory for clipped shapefiles.
        img_shp_csv: Path to CSV with shapefile name mappings.
    """
    # Read point and set CRS
    point1 = gpd.read_file(points_path)
    point = point1.set_crs(crs='EPSG:32611', allow_override=True)

    # Read the names CSV
    df = pd.read_csv(img_shp_csv)
    all_shps = df['shapefile'].values
    shp_set = set(all_shps)

    # Get polygon GeoJSON files from poly_path directory
    if json_file.endswith('.geojson'):
        polygon_shapefile = os.path.join(poly_path, json_file)

        # Read polygon shapefile and set CRS
        poly1 = gpd.read_file(polygon_shapefile)
        poly = poly1.set_crs(crs='EPSG:32611', allow_override=True)

        with open(polygon_shapefile, 'r') as f:
            data = json.load(f)
            features = data['features'][0]['properties']['Licence']

            # Get name for clipping
            filename = features

            # Check if both shapefiles have same CRS
            if point.crs == poly.crs:
                # Clip points using GeoPandas clip
                points_clip = gpd.clip(point, poly)

                # Convert to GeoDataFrame and set CRS
                gdf = gpd.GeoDataFrame(
                    points_clip, crs='EPSG:32611', geometry=points_clip['geometry']
                )

                for use_name in shp_set:
                    correct_name = use_name.split('_')[0]
                    if correct_name == filename:
                        # Check to see if file exists
                        if not os.path.isfile(f'{dest_dir}/{use_name}.shp'):
                            gdf.to_file(f'{dest_dir}/{use_name}.shp')


def buffer_point_shapefile(
    file_no: str,
    licence_no: str,
    batch_no: int,
    in_shp_dir: str,
    out_shp_dir: str,
    distance: int = 20,
) -> None:
    """Buffer point shapefile.

    Args:
        file_no: File number.
        licence_no: Licence number.
        batch_no: Batch number.
        in_shp_dir: Input shapefile directory.
        out_shp_dir: Output shapefile directory.
        distance: Buffer distance in meters.
    """
    in_filename = f'{file_no}.shp'
    in_shp_file = f'{in_shp_dir}{in_filename}'

    if os.path.isfile(in_shp_file):
        out_filename = f'{licence_no}_{batch_no}.shp'
        out_shp_file = f'{out_shp_dir}{out_filename}'

        if not os.path.isfile(out_shp_file):
            # Reading in the input shapefile.
            input1_df = gpd.read_file(in_shp_file)
            input_df = input1_df.explode(index_parts=True)

            # Reading the input CRS.
            input_crs = input_df.crs

            # Reprojecting the data.
            input_df = input_df.to_crs('EPSG:32611')

            # Creating the variable-sized buffer
            input_df['buffer'] = input_df.buffer(distance=distance)

            # Dropping the original geometry and setting the new geometry
            buff_df = input_df.drop(columns=['geometry']).set_geometry('buffer')

            # Reprojecting the buffered data back to the original CRS
            buff_df = buff_df.to_crs(input_crs)

            # Exporting the output to a shapefile
            buff_df.to_file(out_shp_file)

            time.sleep(1)


def plot_shapefile(shp_path: str) -> None:
    """Plot a shapefile using GeoPandas.

    Args:
        shp_path: Path to the shapefile.
    """
    shp = gpd.read_file(shp_path, encoding="utf-8")
    print(shp, shp.crs, shp.geometry)
    shp.plot()


def check_dataset_counts(
    well_points_dir: str,
    well_buffer_dir: str,
) -> Tuple[int, int]:
    """Check dataset counts for well points and buffers.

    Args:
        well_points_dir: Directory containing well point shapefiles.
        well_buffer_dir: Directory containing buffer shapefiles.

    Returns:
        Tuple of (num_wellpoints, num_buffers).
    """
    wellpoint_list: List[str] = []
    for well_file in fnmatch.filter(os.listdir(well_points_dir), '*.shp'):
        only_wells = well_file.split('.')[0]
        wellpoint_list.append(only_wells)

    wellpoint_set = set(wellpoint_list)
    print(f'Number of wellpoint shapefiles {len(wellpoint_set)}')

    buffer_list: List[str] = []
    for shp_file in fnmatch.filter(os.listdir(well_buffer_dir), '*.shp'):
        buffer_shps = shp_file.split('.')[0]
        buffer_list.append(buffer_shps)

    buffer_set = set(buffer_list)
    print(f'Number of buffer shapefiles {len(buffer_set)}')

    return len(wellpoint_set), len(buffer_set)


def generate_random_geographic_points(
    num_points: int,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    output_path: str,
) -> None:
    """Generate uniform random IID geographic points and save as shapefile.

    Args:
        num_points: Number of points to generate.
        lon_min: Minimum longitude.
        lon_max: Maximum longitude.
        lat_min: Minimum latitude.
        lat_max: Maximum latitude.
        output_path: Path to output shapefile.
    """
    new_df = pd.DataFrame()

    def newpoint() -> Tuple[float, float]:
        return uniform(lon_min, lon_max), uniform(lat_min, lat_max)

    x: List[float] = []
    y: List[float] = []
    all_counts: List[str] = []
    counter = 0
    points = (newpoint() for _ in range(num_points))
    for point in points:
        counter += 1
        x.append(point[0])
        y.append(point[1])
        all_counts.append(str(counter))

    new_df['point_id'] = all_counts
    new_df['longitude'] = x
    new_df['latitude'] = y

    # Combine lat and lon column to a shapely Point() object
    new_df['geometry'] = new_df.apply(
        lambda row: MultiPoint([(float(row.longitude), float(row.latitude))]),
        axis=1
    )

    new_g = gpd.GeoDataFrame(new_df)
    new_g.set_geometry(
        gpd.points_from_xy(new_g['longitude'], new_g['latitude']),
        inplace=True,
        crs='EPSG:4326'
    )
    new_g.drop(['longitude', 'latitude'], axis=1, inplace=True)

    new_g.to_file(output_path, driver='ESRI Shapefile')


def convert_geojson_to_shapefile(input_dir: str, output_dir: str) -> None:
    """Convert polygon GeoJSON files to shapefiles.

    Args:
        input_dir: Input directory containing GeoJSON files.
        output_dir: Output directory for shapefiles.
    """
    counter = 0
    for file in os.listdir(input_dir):
        if file.endswith('.geojson'):
            name = file.split('.')[0]

            # Check if output already exists
            if not os.path.isfile(f'{output_dir}/{name}.shp'):
                geojson_file = os.path.join(input_dir, file)

                # Open file for JSON conversion
                with open(geojson_file, 'r') as f:
                    data = json.load(f)
                    new_geometry = gpd.GeoDataFrame.from_features([
                        {
                            'type': 'Feature',
                            'properties': {},
                            'geometry': {
                                'type': 'Polygon',
                                'coordinates': data['features'][0]['geometry']['coordinates'][0],
                            },
                        }
                    ])

                # Read polygon GeoJSON file
                poly_geom = gpd.read_file(geojson_file)

                # Set polygon geometry and CRS
                poly_geom['geometry'] = new_geometry['geometry']
                new_geom = poly_geom.set_crs(crs='EPSG:32611', allow_override=True)

                # Save output to shapefile
                new_geom.to_file(f'{output_dir}/{name}.shp')

                # Track progress
                counter += 1
                if counter % 100 == 0:
                    print(f'{counter} files')

    print(f'All {counter} shapefiles completed')


def check_converted_shapefiles(output_dir: str) -> int:
    """Check converted polygon shapefiles.

    Args:
        output_dir: Directory containing shapefiles.

    Returns:
        Number of polygon shapefiles.
    """
    poly_list: List[str] = []
    for well_file in fnmatch.filter(os.listdir(output_dir), '*.shp'):
        poly_list.append(well_file)

    poly_set = set(poly_list)
    print(f'Number of polygon shapefiles {len(poly_set)}')
    return len(poly_set)


def search_geojsons_by_licences(
    json_path: str,
    licences: Set[str],
    start_batch: int = 2,
    end_batch: int = 77,
    step: int = 2,
) -> None:
    """Search all GeoJSONs and find files with specified licences.

    Args:
        json_path: Path to batched GeoJSON directories.
        licences: Set of licence strings to search for.
        start_batch: Starting batch number.
        end_batch: Ending batch number.
        step: Step size for batch numbers.
    """
    counter = 0
    found_geojsons: Set[str] = set()

    for num in range(start_batch, end_batch, step):
        geojson_path = f'{json_path}grid_batch_{num}'

        for file in os.listdir(geojson_path):
            if file.endswith('.geojson'):
                counter += 1
                geojson_file = f'{geojson_path}/{file}'

                with open(geojson_file, 'r') as f:
                    data = json.load(f)
                    features = data['features'][0]['properties']['Licence']
                    features = f'{features}_{num}'

                    for lic in licences:
                        if features == lic:
                            print(f'{file}/{features}')


def main() -> None:
    """Main function for running shapefile processing utilities."""
    # Example usage - replace with actual paths and parameters
    create_geojson_batches(
        json_path='.../polygon_geojson/',
        dest_path='.../ybdir/geojsons/',
    )

    counter_no, batch_no, geojson_no, licence_no = collect_geojson_metadata(
        json_path='.../geojsons/',
    )
    create_reference_csv(
        counter_no, batch_no, geojson_no, licence_no,
        '.../file_format.csv'
    )

    polygons = [item for item in os.listdir('.../polygon_geojson/')
                if item.endswith('.geojson') and 0 < int(item.split('.')[0]) <= 2000]
    parallelism = 10
    thread_pool = ThreadPool(parallelism)
    thread_pool.map(
        lambda json_file: clip_points_to_polygons(
            json_file,
            '.../points.geojson',
            '.../polygon_geojson/',
            '.../clipped_polypoints',
            ".../img_shp.csv"
        ),
        polygons
    )

    df = pd.read_csv(".../file_format.csv")
    names_list = df['geojson_no'].tolist()
    licence_list = df['licence_no'].tolist()
    batch_list = df['batch_no'].tolist()
    data_points = list(zip(names_list, licence_list, batch_list))
    thread_pool.map(
        lambda x: buffer_point_shapefile(
            x[0], x[1], x[2],
            '.../well_points/',
            '.../well_points_buffer/'
        ),
        data_points
    )

    plot_shapefile('.../1.shp')
    check_dataset_counts('.../well_points/', '.../well_points_buffer/')

    generate_random_geographic_points(
        5000, -120.00, -109.99, 48.99, 60.00, '.../new_points.shp'
    )

    convert_geojson_to_shapefile(
        input_dir='... /non_wells_utm',
        output_dir='... /non_wells_shp'
    )

    check_converted_shapefiles('... /non_wells_shp')

    my_set = set(['licence1', 'licence2'])  # Define your licence set
    search_geojsons_by_licences(".../geojsons/", my_set)

    print("Shapefile processing functions defined. Uncomment to run.")


if __name__ == "__main__":
    main()