"""Merge known and nationwide discovery-map coordinates.

Never changes V2 ranking or strict itinerary evidence.
"""

import math

from src.v3_recommendation_map import (
    build_recommendation_map_records,
)
from src.v3_nationwide_map_lookup import (
    discover_recommendation_coordinates,
)


def build_nationwide_recommendation_map(
    gallery_records,
    confirmed_locations=None,
    max_lookups=3,
    max_http_requests=6,
):
    """Return ordered, deduplicated discovery markers."""

    if not isinstance(gallery_records, (list, tuple)):
        raise TypeError("Gallery records must be a list or tuple.")

    known = build_recommendation_map_records(
        gallery_records,
        confirmed_locations=confirmed_locations,
    )

    known_markers = known["markers"]

    known_ids = {
        marker["place_id"]
        for marker in known_markers
    }

    missing_gallery = [
        item
        for item in gallery_records
        if item["place_id"] not in known_ids
    ]

    discovered = discover_recommendation_coordinates(
        missing_gallery,
        max_lookups=max_lookups,
        max_http_requests=max_http_requests,
    )

    marker_by_id = {
        marker["place_id"]: marker
        for marker in known_markers
    }

    expected_by_id = {
        str(item["place_id"]): item
        for item in gallery_records
    }

    for marker in discovered["markers"]:
        if not isinstance(marker, dict):
            continue

        place_id = str(marker.get("place_id", "")).strip()
        expected = expected_by_id.get(place_id)

        if expected is None or place_id in marker_by_id:
            continue

        expected_name = str(expected["place"]).strip().casefold()
        expected_city = str(expected["city"]).strip().casefold()

        actual_name = str(marker.get("place", "")).strip().casefold()
        actual_city = str(marker.get("city", "")).strip().casefold()

        if (
            actual_name != expected_name
            or actual_city != expected_city
            or marker.get("location_status") != "WIKIDATA_CANDIDATE"
        ):
            continue

        try:
            latitude = float(marker["latitude"])
            longitude = float(marker["longitude"])
        except (KeyError, TypeError, ValueError):
            continue

        if not (
            math.isfinite(latitude)
            and math.isfinite(longitude)
            and -90 <= latitude <= 90
            and -180 <= longitude <= 180
        ):
            continue

        marker_by_id[place_id] = {
            **marker,
            "place_id": place_id,
            "place": expected["place"],
            "city": expected["city"],
            "rank": expected.get("rank"),
            "latitude": latitude,
            "longitude": longitude,
        }

    ordered = []
    missing = []
    seen = set()

    for item in gallery_records:
        place_id = item["place_id"]

        if place_id in seen:
            continue

        seen.add(place_id)

        marker = marker_by_id.get(place_id)

        if marker is None:
            missing.append(place_id)
        else:
            ordered.append(dict(marker))

    return {
        "markers": ordered,
        "missing_place_ids": missing,
        "known_marker_count": len(known_markers),
        "discovered_marker_count": (
            len(ordered) - len(known_markers)
        ),
        "http_requests_used": discovered["http_requests_used"],
        "http_request_limit": discovered["http_request_limit"],
        "rate_limited": discovered["rate_limited"],
    }
