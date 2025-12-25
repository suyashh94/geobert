"""
Fetch NYC geocoding data from NYC Open Data API and save locally.
"""

import json
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def fetch_nyc_data(output_path: Path | None = None) -> None:
    """
    Fetch geocoding data from NYC Open Data API and save to local file.
    Uses Socrata Open Data API (SODA) format.

    :param output_path: Path where the JSON data will be saved.
                        Defaults to data/nyc_geocoding_raw.json
    """
    if output_path is None:
        output_path = DATA_DIR / "nyc_geocoding_raw.json"

    base_url = "https://data.cityofnewyork.us/resource/uf93-f8nk.json"

    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (compatible; NYC-Geocoding-Project/1.0)"
    }

    params = {
        "$limit": 50000,
        "$offset": 0
    }

    all_data = []

    print(f"Fetching data from {base_url}...")

    while True:
        print(f"  Fetching offset {params['$offset']}...")
        response = requests.get(base_url, headers=headers, params=params, timeout=300)
        response.raise_for_status()

        chunk = response.json()
        if not chunk:
            break

        all_data.extend(chunk)
        print(f"  Fetched {len(chunk)} rows (total: {len(all_data)})")

        if len(chunk) < params['$limit']:
            break

        params['$offset'] += params['$limit']

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_data, f)

    print(f"Fetched {len(all_data)} total rows")
    if all_data:
        print(f"Sample fields: {list(all_data[0].keys())}")

    print(f"Data saved to {output_path}")


if __name__ == "__main__":
    fetch_nyc_data()
