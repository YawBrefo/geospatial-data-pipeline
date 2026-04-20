#!/usr/bin/env python3
"""Geospatial image processing pipeline for satellite data.

This module provides utilities for processing Planet Labs satellite imagery,
including file renaming, reprojection, mask generation, and visualization.
It is designed for batch processing of PSScene data for well detection and
related geospatial analysis.
"""

from __future__ import annotations

import fnmatch
import glob
import json
import os
import shutil
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from multiprocessing.dummy import Pool as ThreadPool
from rasterio import CRS
from rasterio.features import rasterize
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import mapping, Point, Polygon
from shapely.ops import unary_union
from zipfile import ZipFile


def extract_and_rename_images(
    img_path: str,
    dest_path: str,
    csv_path: str,
) -> None:
    """Extract images from PSScene imagery and rename to a clean directory.

    Args:
        img_path: Glob pattern for input TIFF files.
        dest_path: Destination directory for renamed images.
        csv_path: Path to CSV file with licence and batch information.
    """
    df = pd.read_csv(csv_path)
    all_files: List[str] = []
    copied_files: List[str] = []
    counter = 0

    for file in glob.glob(img_path):
        licence1 = file.split('/')[7]
        licence2 = int(licence1)
        licence = str(licence2).rjust(7, '0')

        all_files.append(licence1)

        for _, row in df.iterrows():
            lic = row['licence_no']
            lic1 = str(lic).rjust(7, '0')

            if licence == lic1:
                copied_files.append(licence)
                batch = row['batch_no']
                geojson_no = row['geojson_no']

                new_file = f'{dest_path}/{licence}_{batch}.tif'

                if not os.path.isfile(new_file):
                    shutil.copy(file, new_file)
                    counter += 1
                    if counter % 100 == 0:
                        print(f'{counter} files')


def rename_images_to_shapefile_names(
    df_csv: str,
    image_dir: str,
    new_image_dir: str,
) -> None:
    """Rename image files to match corresponding shapefile names.

    Args:
        df_csv: Path to CSV with image and shapefile name mappings.
        image_dir: Source directory containing images.
        new_image_dir: Destination directory for renamed images.
    """
    counter = 0
    df = pd.read_csv(df_csv)

    for _, row in df.iterrows():
        old_name = row['image']
        new_name = row['shapefile']

        old_file_path = f'{image_dir}/{old_name}.tif'
        new_file_path = f'{new_image_dir}/{new_name}.tif'

        shutil.copy(old_file_path, new_file_path)

        counter += 1
        if counter % 1000 == 0:
            print(f'{counter} files copied')

    print(f'All {counter} files have been completed')


def check_renamed_images(
    orig_path: str,
    renamed_path: str,
) -> Set[str]:
    """Check that all images have been renamed correctly.

    Args:
        orig_path: Directory containing original files.
        renamed_path: Directory containing renamed images.

    Returns:
        Set of licence numbers that differ between directories.
    """
    all_img_files: List[str] = []

    for dirpath, subdirs, files in os.walk(orig_path):
        for x in subdirs:
            licence2 = int(x)
            licence = str(licence2).rjust(7, '0')
            all_img_files.append(licence)

    renamed_img_files: List[str] = []

    for item in os.listdir(renamed_path):
        if item.endswith('.tif'):
            rename1 = item.split('_')[0]
            renamed_img_files.append(rename1)

    diff = set(all_img_files) - set(renamed_img_files)
    return diff


def reproject_tiff(
    tif_file: str,
    in_path: str,
    out_path: str,
    dst_crs: CRS,
) -> None:
    """Reproject a single TIFF file to the target CRS.

    Args:
        tif_file: Name of the TIFF file.
        in_path: Input directory path.
        out_path: Output directory path.
        dst_crs: Target CRS for reprojection.
    """
    tiff = os.path.join(in_path, tif_file)
    out_tiff = os.path.join(out_path, tif_file)

    if not os.path.isfile(out_tiff):
        with rasterio.open(tiff) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height,
            })

            with rasterio.open(out_tiff, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest,
                    )

            print(f'{tif_file} file reprojected')
    else:
        print(f'{tif_file} exists')

    time.sleep(1)


def check_shapefile_image_correspondence(
    shapefile_dir: str,
    image_dir: str,
) -> Tuple[int, int, int, int]:
    """Check correspondence between shapefiles and images.

    Args:
        shapefile_dir: Directory containing shapefiles.
        image_dir: Directory containing images.

    Returns:
        Tuple of (num_shapefiles, num_images, diff_count, intersection_count).
    """
    file_list: List[str] = []
    for file in fnmatch.filter(os.listdir(shapefile_dir), '*.shp'):
        only_file = file.split('.')[0]
        file_list.append(only_file)

    shapefile_set = set(file_list)
    print(f'Number of shapefiles {len(shapefile_set)}')

    img_list: List[str] = []
    for img in os.listdir(image_dir):
        only_img = img.split('.')[0]
        img_list.append(only_img)

    img_set = set(img_list)
    print(f'Number of images {len(img_set)}')

    diff_set = shapefile_set - img_set
    print(f'Difference: {len(diff_set)}')

    inter_set = shapefile_set.intersection(img_set)
    print(f'Intersection: {len(inter_set)}')

    return len(shapefile_set), len(img_set), len(diff_set), len(inter_set)


def generate_mask(
    raster_path: str,
    shape_path: str,
    output_path: str,
) -> None:
    """Generate binary masks from vector files (shp or geojson).

    Args:
        raster_path: Path to directory containing TIFF rasters.
        shape_path: Path to directory containing shapefiles.
        output_path: Path to save the binary masks.
    """
    counter = 0

    for raster in os.listdir(raster_path):
        filename = raster.split('.')[0]
        outer_file = f'{output_path}{filename}.tif'

        if not os.path.isfile(outer_file):
            raster_file = os.path.join(raster_path, raster)

            if os.path.isfile(raster_file):
                with rasterio.open(raster_file, "r") as src:
                    raster_img = src.read()
                    raster_meta = src.meta

                shape_file = f'{shape_path}{filename}.shp'

                if os.path.isfile(shape_file):
                    train_df1 = gpd.read_file(shape_file)
                    train_df = train_df1.set_crs(
                        crs='EPSG:32611', allow_override=True
                    )

                    if train_df.crs != src.crs:
                        print(
                            f"Raster CRS: {src.crs}, Vector CRS: {train_df.crs}. "
                            "Convert to same CRS."
                        )
                    else:
                        def poly_from_utm(polygon, transform):
                            poly_pts = []
                            poly = unary_union(polygon)

                            for i in np.array(poly.exterior.coords):
                                poly_pts.append(~transform * tuple(i))

                            new_poly = Polygon(poly_pts)
                            return new_poly

                        poly_shp: List[Polygon] = []
                        im_size = (src.meta['height'], src.meta['width'])

                        for num, row in train_df.iterrows():
                            try:
                                if row['geometry'].geom_type == 'Polygon':
                                    poly = poly_from_utm(
                                        row['geometry'], src.meta['transform']
                                    )
                                    poly_shp.append(poly)
                                else:
                                    for p in row['geometry']:
                                        poly = poly_from_utm(
                                            p, src.meta['transform']
                                        )
                                        poly_shp.append(poly)
                            except Exception:
                                print(shape_file)

                        mask = rasterize(shapes=poly_shp, out_shape=im_size)
                        mask = mask.astype("uint16")
                        save_path = f'{output_path}{filename}.tif'
                        bin_mask_meta = src.meta.copy()
                        bin_mask_meta.update({'count': 1})

                        with rasterio.open(save_path, 'w', **bin_mask_meta) as dst:
                            dst.write(mask * 255, 1)

                        counter += 1

        if counter % 100 == 0:
            print(f'Masked {counter} file created')

    print(f'Total of {counter} masks created')
    print('##################################################################')


def check_dataset_completeness(
    shp_dir: str,
    img_dir: str,
    mask_img: str,
) -> Tuple[int, int, int, int]:
    """Check completeness of shapefile, image, and mask datasets.

    Args:
        shp_dir: Directory containing shapefiles.
        img_dir: Directory containing images.
        mask_img: Directory containing mask images.

    Returns:
        Tuple of (num_shapefiles, num_images, num_masks, diff_count).
    """
    shp_list: List[str] = []
    for well_file in fnmatch.filter(os.listdir(shp_dir), '*.shp'):
        only_wells = well_file.split('.')[0]
        shp_list.append(only_wells)

    shp_set = set(shp_list)
    print(f'Number of shapefiles {len(shp_set)}')

    imgs: List[str] = []
    for img in os.listdir(img_dir):
        if img.endswith('.tif'):
            img1 = img.split('.')[0]
            imgs.append(img1)

    img_set = set(imgs)
    print(f'Number of images {len(img_set)}')

    masks: List[str] = []
    for new_img in os.listdir(mask_img):
        if new_img.endswith('.tif'):
            img = new_img.split('.')[0]
            masks.append(img)

    mask_set = set(masks)
    print(f'Number of mask images {len(mask_set)}')

    diff = mask_set - img_set
    print(f'Difference: {len(diff)}')

    return len(shp_set), len(img_set), len(mask_set), len(diff)


def plot_image(path: str) -> None:
    """Plot a raster image using rasterio.

    Args:
        path: Path to the TIFF file to plot.
    """
    planet_img = rasterio.open(path)
    print(f'Image CRS: {planet_img.crs}')
    show(planet_img)


def scale_band(band: np.ndarray) -> np.ndarray:
    """Scale down image band resolution for plotting.

    Args:
        band: Input band array.

    Returns:
        Scaled band array.
    """
    return band / 1000


def plot_shapefile(shp_file_path: str) -> None:
    """Plot a shapefile using geopandas.

    Args:
        shp_file_path: Path to the shapefile.
    """
    shp_file = gpd.read_file(shp_file_path)
    print(f'Shapefile CRS: {shp_file.crs}')
    shp_file.plot()


def plot_raster_with_matplotlib(path: str) -> None:
    """Plot raster using matplotlib.

    Args:
        path: Path to the TIFF file.
    """
    src = rasterio.open(path)
    plt.imshow(src.read(1))
    plt.show()


def get_shapefile_names_from_geojsons(
    poly_path: str,
    start_batch: int = 2,
    end_batch: int = 77,
    step: int = 2,
) -> Set[str]:
    """Get shapefile names from polygon geojsons.

    Args:
        poly_path: Base path to geojson directories.
        start_batch: Starting batch number.
        end_batch: Ending batch number.
        step: Step size for batch numbers.

    Returns:
        Set of shapefile names.
    """
    counter = 0
    all_geojsons: Set[str] = set()

    for num in range(start_batch, end_batch, step):
        geojson_path = f'{poly_path}grid_batch_{num}'

        for file in os.listdir(geojson_path):
            if file.endswith('.geojson'):
                geojson_file = f'{geojson_path}/{file}'
                counter += 1

                with open(geojson_file, 'r') as f:
                    data = json.load(f)
                    features = data['features'][0]['properties']['Licence']
                    features = f'{features}_{num}'
                    all_geojsons.add(features)

    print(f'Processed {counter} geojson files')
    return all_geojsons


def create_name_correspondence_library(
    img_set: Set[str],
    all_geojsons: Set[str],
    output_csv: str,
) -> None:
    """Create library for shp-img name correspondence.

    Args:
        img_set: Set of image names.
        all_geojsons: Set of geojson-derived names.
        output_csv: Path to output CSV file.
    """
    counter = 0
    shp_list: List[str] = []
    img_list: List[str] = []

    for img_item in img_set:
        for shp_item in all_geojsons:
            try:
                front_img_no = int(img_item.split('_')[0])
                back_img_no = int(img_item.split('_')[1])

                front_shp_no = int(shp_item.split('_')[0])
                back_shp_no = int(shp_item.split('_')[1])

                if ((front_shp_no == front_img_no) and
                    (back_shp_no == back_img_no)):
                    shp_list.append(shp_item)
                    img_list.append(img_item)
                    counter += 1

            except ValueError:
                if shp_item == img_item:
                    shp_list.append(shp_item)
                    img_list.append(img_item)
                    counter += 1

        if counter % 1000 == 0:
            print(f'{counter} files checked')

    df = pd.DataFrame(list(zip(img_list, shp_list)),
                      columns=['image', 'shapefile'])
    df.to_csv(output_csv, index=False)


def main() -> None:
    """Main function for running the image processing pipeline."""
    Example usage - replace with actual paths
    extract_and_rename_images(
        img_path='.../*/PSScene/*_3B_AnalyticMS_SR_clip.tif',
        dest_path='.../renamed_images',
        csv_path='.../file_format.csv',
    )

    rename_images_to_shapefile_names(
        df_csv=".../img_shp.csv",
        image_dir='.../data_images',
        new_image_dir='.../well_images',
    )

    diff = check_renamed_images(
        orig_path='.../orig_PSScene/',
        renamed_path='.../renamed_images/',
    )
    print(diff)

    # For reprojection
    target_file = '.../7_2_AnalyticMS_SR_clip.tif'
    with rasterio.open(target_file) as dst_crs:
        dst_crs_obj = dst_crs.crs

    image_list = [img for img in os.listdir('.../renamed_images/')]
    parallelism = 10
    thread_pool = ThreadPool(parallelism)
    thread_pool.map(
        lambda img: reproject_tiff(img, '.../renamed_images/',
                                   '.../images_repr/', dst_crs_obj),
        image_list
    )

    check_shapefile_image_correspondence(
        shapefile_dir='.../points_buffer/',
        image_dir='.../images/',
    )

    generate_mask(
        raster_path='.../images/',
        shape_path='.../shp/',
        output_path='.../masks/',
    )

    check_dataset_completeness(
        shp_dir='.../shp/',
        img_dir='.../images/',
        mask_img='.../masks/',
    )

    plot_image('.../non_well_masks/1001.tif')
    plot_shapefile("/network/projects/rolnick_abd23/well_points/8841.shp")
    plot_raster_with_matplotlib('.../non_well_masks/1001.tif')

    all_geojsons = get_shapefile_names_from_geojsons(poly_path='...')
    img_set = set([img.split('.')[0] for img in os.listdir('.../images/')
                   if img.endswith('.tif')])
    create_name_correspondence_library(
        img_set, all_geojsons, '.../img_shp.csv'
    )

    print("Pipeline functions defined. Uncomment and configure paths to run.")


if __name__ == "__main__":
    main()