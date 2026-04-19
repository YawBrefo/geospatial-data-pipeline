#!/usr/bin/env python3
"""Planet imagery pipeline for geojson AOI search, activation, and download.

This module demonstrates a geospatial data engineering workflow using the Planet
API. It is designed for portfolio presentation and can be run as a command-line
pipeline for:

- scanning a folder of GeoJSON AOIs,
- searching for matching PSScene imagery,
- activating analytic assets,
- creating orders and downloading clipped scenes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from planet import Auth, OrdersClient, Session, reporting
from requests.auth import HTTPBasicAuth
from shapely.geometry import shape


DEFAULT_DATE_START = "2022-06-03T00:00:00.000Z"
DEFAULT_DATE_END = "2022-07-31T00:00:00.000Z"
DEFAULT_ITEM_TYPE = "PSScene"
DEFAULT_ASSET_TYPE = "ortho_analytic_4b_sr"
DEFAULT_PRODUCT_BUNDLE = "analytic_sr_udm2"


def get_api_key() -> str:
    """Return the Planet API key from the environment.

    Raises:
        EnvironmentError: If the Planet API key is not configured.
    """
    api_key = os.getenv("PLANET_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "The Planet API key must be set in the PLANET_API_KEY environment variable."
        )
    return api_key


def load_geojson_aoi(geojson_path: Path) -> Tuple[str, Dict[str, Any]]:
    """Load a GeoJSON file and return the licence and AOI geometry.

    Args:
        geojson_path: Path to a GeoJSON file containing a single AOI feature.

    Returns:
        A tuple containing the licence identifier and the GeoJSON geometry object.
    """
    with geojson_path.open("r", encoding="utf-8") as geojson_file:
        data = json.load(geojson_file)

    first_feature = data["features"][0]
    licence_id = first_feature["properties"].get("Licence", geojson_path.stem)
    geometry = first_feature["geometry"]
    return licence_id, geometry


def search_planet_item(
    aoi_geometry: Dict[str, Any],
    api_key: str,
    item_type: str = DEFAULT_ITEM_TYPE,
    date_start: str = DEFAULT_DATE_START,
    date_end: str = DEFAULT_DATE_END,
    max_cloud_cover: float = 0.01,
) -> Optional[str]:
    """Search Planet quick-search for an item that contains the AOI.

    Args:
        aoi_geometry: The GeoJSON geometry for the AOI.
        api_key: Planet API key.
        item_type: The Planet item type to search.
        date_start: ISO-formatted acquisition start date.
        date_end: ISO-formatted acquisition end date.
        max_cloud_cover: Maximum acceptable cloud cover.

    Returns:
        The first matching Planet item ID, or None if no item is found.
    """
    request_payload = {
        "item_types": [item_type],
        "filter": {
            "type": "AndFilter",
            "config": [
                {"type": "GeometryFilter", "field_name": "geometry", "config": aoi_geometry},
                {
                    "type": "DateRangeFilter",
                    "field_name": "acquired",
                    "config": {"gte": date_start, "lte": date_end},
                },
                {
                    "type": "RangeFilter",
                    "field_name": "cloud_cover",
                    "config": {"gte": 0.0, "lt": max_cloud_cover},
                },
            ],
        },
    }

    response = requests.post(
        "https://api.planet.com/data/v1/quick-search",
        auth=HTTPBasicAuth(api_key, ""),
        json=request_payload,
    )
    response.raise_for_status()
    search_result = response.json()

    aoi_shape = shape(aoi_geometry)
    candidate_ids: List[str] = []

    for feature in search_result.get("features", []):
        feature_shape = shape(feature["geometry"])
        if feature_shape.contains(aoi_shape):
            candidate_ids.append(feature["id"])

    if not candidate_ids:
        return None
    return candidate_ids[-1]


def build_data_item_index(
    geojson_dir: Path,
    api_key: str,
    date_start: str = DEFAULT_DATE_START,
    date_end: str = DEFAULT_DATE_END,
    max_cloud_cover: float = 0.01,
) -> pd.DataFrame:
    """Build a data index of Planet item IDs for a folder of GeoJSON AOIs.

    Args:
        geojson_dir: Directory containing GeoJSON files.
        api_key: Planet API key.
        date_start: Search acquisition start date.
        date_end: Search acquisition end date.
        max_cloud_cover: Maximum cloud cover threshold.

    Returns:
        A pandas DataFrame with licence, item ID, and AOI geometry.
    """
    records: List[Dict[str, Any]] = []
    geojson_files = sorted(geojson_dir.glob("*.geojson"))

    for geojson_path in geojson_files:
        licence_id, geometry = load_geojson_aoi(geojson_path)
        item_id = search_planet_item(
            geometry,
            api_key,
            date_start=date_start,
            date_end=date_end,
            max_cloud_cover=max_cloud_cover,
        )
        records.append(
            {
                "licence_id": licence_id,
                "item_id": item_id if item_id is not None else "",
                "aoi_geometry": json.dumps(geometry),
                "source_file": geojson_path.name,
            }
        )

    data_frame = pd.DataFrame(records)
    data_frame = data_frame.sort_values(by=["licence_id"], ignore_index=True)
    return data_frame


def save_dataframe(data_frame: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to a tab-delimited CSV file.

    Args:
        data_frame: DataFrame to save.
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data_frame.to_csv(path, sep="\t", index=False)


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a tab-delimited CSV into a DataFrame and restore AOI geometry.

    Args:
        path: Path to the CSV file.

    Returns:
        A pandas DataFrame with decoded AOI geometry.
    """
    data_frame = pd.read_csv(path, sep="\t")
    if "aoi_geometry" in data_frame.columns:
        data_frame["aoi_geometry"] = data_frame["aoi_geometry"].apply(json.loads)
    return data_frame


def activate_asset(
    item_id: str,
    api_key: str,
    item_type: str = DEFAULT_ITEM_TYPE,
    asset_type: str = DEFAULT_ASSET_TYPE,
    poll_interval: int = 5,
    max_attempts: int = 12,
) -> str:
    """Activate a Planet asset and wait until activation is complete.

    Args:
        item_id: Planet item identifier.
        api_key: Planet API key.
        item_type: Item type name.
        asset_type: Asset type name.
        poll_interval: Seconds between status checks.
        max_attempts: Maximum number of status checks.

    Returns:
        The final asset status string.
    """
    asset_url = (
        f"https://api.planet.com/data/v1/item-types/{item_type}/items/{item_id}/assets"
    )
    response = requests.get(asset_url, auth=HTTPBasicAuth(api_key, ""))
    response.raise_for_status()
    asset_info = response.json().get(asset_type, {})

    if not asset_info:
        raise ValueError(f"Asset type {asset_type} not found for item {item_id}.")

    links = asset_info.get("_links", {})
    activate_url = links.get("activate")
    self_url = links.get("_self")

    if not activate_url or not self_url:
        raise ValueError("Missing activation links for asset.")

    requests.get(activate_url, auth=HTTPBasicAuth(api_key, "")).raise_for_status()

    last_status = "unknown"
    for attempt in range(max_attempts):
        time.sleep(poll_interval)
        status_response = requests.get(self_url, auth=HTTPBasicAuth(api_key, ""))
        status_response.raise_for_status()
        last_status = status_response.json().get("status", last_status)
        if last_status.lower() in {"active", "activated"}:
            break
        logging.debug("Waiting for asset %s status: %s", item_id, last_status)

    return last_status


def build_order_payload(item_id: str, aoi_geometry: Dict[str, Any]) -> Dict[str, Any]:
    """Build an order payload for a single clipped PSScene item.

    Args:
        item_id: Planet item identifier.
        aoi_geometry: The GeoJSON geometry for the clipping AOI.

    Returns:
        A dictionary representing the Planet Orders API request body.
    """
    return {
        "name": f"clip-order-{item_id}",
        "products": [
            {
                "item_ids": [item_id],
                "item_type": DEFAULT_ITEM_TYPE,
                "product_bundle": DEFAULT_PRODUCT_BUNDLE,
            }
        ],
        "tools": [{"clip": {"aoi": aoi_geometry}}],
    }


async def create_poll_and_download(
    auth: Auth, request_payloads: Iterable[Dict[str, Any]]
) -> None:
    """Create Planet orders, poll until completion, and download the results.

    Args:
        auth: Planet Auth object.
        request_payloads: Iterable of order payload dictionaries.
    """
    async with Session(auth=auth) as session:
        client = OrdersClient(session)
        for payload in request_payloads:
            with reporting.StateBar(state="creating") as bar:
                order = await client.create_order(payload)
                bar.update(state="created", order_id=order["id"])
                await client.wait(order["id"], callback=bar.update_state)
            await client.download_order(order["id"])
            logging.info("Downloaded order %s", order["id"])


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Planet GeoTIFF search, activation, and download pipeline."
    )
    parser.add_argument(
        "--geojson-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory containing GeoJSON AOI files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory where CSV outputs and downloads are saved.",
    )
    parser.add_argument(
        "--skip-activation",
        action="store_true",
        help="Skip asset activation and only perform search and order creation.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip order download and only generate the CSV files.",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_DATE_START,
        help="Search start date in ISO format.",
    )
    parser.add_argument(
        "--end-date",
        default=DEFAULT_DATE_END,
        help="Search end date in ISO format.",
    )
    parser.add_argument(
        "--max-cloud-cover",
        type=float,
        default=0.01,
        help="Maximum allowed cloud cover fraction.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the Planet imagery pipeline."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
    )

    args = parse_arguments()
    api_key = get_api_key()
    output_dir = args.output_dir.resolve()
    data_item_csv = output_dir / "data_item.csv"
    status_item_csv = output_dir / "status_item.csv"

    logging.info("Starting pipeline for geojson directory: %s", args.geojson_dir)
    data_frame = build_data_item_index(
        args.geojson_dir,
        api_key,
        date_start=args.start_date,
        date_end=args.end_date,
        max_cloud_cover=args.max_cloud_cover,
    )
    save_dataframe(data_frame, data_item_csv)
    logging.info("Saved Planet item index to %s", data_item_csv)

    if not args.skip_activation:
        activation_status: List[str] = []
        for item_id in data_frame["item_id"]:
            if item_id:
                status = activate_asset(item_id, api_key)
                activation_status.append(status)
                logging.info("Item %s activation status: %s", item_id, status)
            else:
                activation_status.append("missing_item")
        data_frame["status"] = activation_status
        save_dataframe(data_frame, status_item_csv)
        logging.info("Saved activation status to %s", status_item_csv)
    else:
        logging.info("Skipping asset activation step.")

    if not args.skip_download:
        request_payloads = []
        for _, row in data_frame.iterrows():
            item_id = row["item_id"]
            if not item_id:
                logging.warning("Skipping download for missing item in row %s", row["source_file"])
                continue
            request_payloads.append(build_order_payload(item_id, json.loads(row["aoi_geometry"])))

        if request_payloads:
            auth = Auth.from_key(api_key)
            asyncio.run(create_poll_and_download(auth, request_payloads))
            logging.info("Download pipeline completed.")
        else:
            logging.warning("No valid Planet items were found for download.")
    else:
        logging.info("Skipping order download step.")


if __name__ == "__main__":
    main()
