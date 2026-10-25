"""Discovery-map coordinates for V2-ranked attractions.

Map preview identity is separate from itinerary evidence approval.
Never infer attraction coordinates from city-center metadata.
"""

import csv
import math
from pathlib import Path


WIKIDATA_CANDIDATES = {
    "INPLACE_05344": {
        "latitude": 26.982777777777777,
        "longitude": 75.84472222222222,
        "wikidata_id": "Q2722759",
        "place": "Jaigarh Fort",
        "city": "Jaipur",
    },
    "INPLACE_05387": {
        "latitude": 26.9375,
        "longitude": 75.81583333333333,
        "wikidata_id": "Q2723401",
        "place": "Nahargarh Fort",
        "city": "Jaipur",
    },
    "INPLACE_05297": {
        "latitude": 26.9255,
        "longitude": 75.8236,
        "wikidata_id": "Q2723395",
        "place": "City Palace of Jaipur",
        "city": "Jaipur",
    },
    "INPLACE_05327": {
        "latitude": 26.98611111111111,
        "longitude": 75.8506111111111,
        "wikidata_id": "Q110752779",
        "place": "Ganesh Pol Gateway",
        "city": "Jaipur",
    },
    "INPLACE_05272": {
        "latitude": 26.9859,
        "longitude": 75.8507,
        "wikidata_id": "Q456817",
        "place": "Amber Fort Palace",
        "city": "Jaipur",
    },
}


def _valid_coordinates(latitude, longitude):
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        return False

    return (
        math.isfinite(lat)
        and math.isfinite(lon)
        and -90 <= lat <= 90
        and -180 <= lon <= 180
    )


def load_confirmed_attraction_locations(
    path="v3_data/attraction_locations.csv",
):
    """Load existing confirmed attraction-level coordinates."""

    location_path = Path(path)

    if not location_path.is_file():
        return {}

    locations = {}

    with location_path.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for row in csv.DictReader(handle):
            place_id = str(row.get("place_id", "")).strip()

            if not place_id:
                continue

            if row.get("verification_status") != "user_confirmed":
                continue

            lat = row.get("latitude")
            lon = row.get("longitude")

            if not _valid_coordinates(lat, lon):
                continue

            locations[place_id] = {
                "latitude": float(lat),
                "longitude": float(lon),
                "source": "User-confirmed attraction pin",
                "location_status": "USER_CONFIRMED",
                "place": row.get("place_name", ""),
                "city": row.get("city_canonical", ""),
            }

    return locations


def build_recommendation_map_records(
    gallery_records,
    confirmed_locations=None,
):
    """Return one ordered marker collection for actual recommendations."""

    if not isinstance(gallery_records, (list, tuple)):
        raise TypeError("Gallery records must be a list or tuple.")

    confirmed = (
        load_confirmed_attraction_locations()
        if confirmed_locations is None
        else confirmed_locations
    )

    if not isinstance(confirmed, dict):
        raise TypeError("Confirmed locations must be a dictionary.")

    markers = []
    missing = []

    for item in gallery_records:
        if not isinstance(item, dict):
            raise TypeError("Gallery item must be a dictionary.")

        place_id = str(item.get("place_id", "")).strip()
        place = str(item.get("place", "")).strip()
        city = str(item.get("city", "")).strip()

        if not place_id or not place or not city:
            raise ValueError("Incomplete gallery attraction identity.")

        location = confirmed.get(place_id)

        if location is not None:
            recorded_city = str(
                location.get("city", "")
            ).strip().casefold()

            if recorded_city != city.casefold():
                missing.append(place_id)
                continue

            source = location.get("source", "Confirmed location")
            status = "USER_CONFIRMED"
        else:
            location = WIKIDATA_CANDIDATES.get(place_id)

            if location is None:
                missing.append(place_id)
                continue

            if (
                location["place"].casefold() != place.casefold()
                or location["city"].casefold() != city.casefold()
            ):
                missing.append(place_id)
                continue

            source = "Wikidata candidate " + location["wikidata_id"]
            status = "WIKIDATA_CANDIDATE"

        lat = location.get("latitude")
        lon = location.get("longitude")

        if not _valid_coordinates(lat, lon):
            missing.append(place_id)
            continue

        markers.append({
            "place_id": place_id,
            "place": place,
            "city": city,
            "latitude": float(lat),
            "longitude": float(lon),
            "source": source,
            "location_status": status,
            "rank": item.get("rank"),
        })

    return {
        "markers": markers,
        "missing_place_ids": missing,
    }
