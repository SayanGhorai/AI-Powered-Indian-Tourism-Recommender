"""Conservative V3 itinerary stop eligibility.

This module assesses attraction identity, not operational
opening status. It does not modify V2 recommendations.
"""

import re
import unicodedata

import pandas as pd
import numpy as np


REQUIRED_CATALOG_COLUMNS = {
    "place_id",
    "place",
    "city",
    "city_canonical",
    "attraction_type",
    "has_dataset2_metadata",
}


def _normalize_name(value):
    """Normalize a place or city name for exact comparison."""

    if not isinstance(value, str):
        return ""

    value = unicodedata.normalize("NFKC", value)
    value = value.casefold()
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def _has_value(value):
    """Check whether a metadata value is meaningfully present."""

    if value is None:
        return False

    if isinstance(value, str):
        return bool(value.strip())

    try:
        return bool(pd.notna(value))
    except (TypeError, ValueError):
        return False


def _is_true(value):
    """Interpret the catalog's explicit metadata-presence flag."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, str):
        return value.strip().lower() == "true"

    return False


def assess_stop_eligibility(catalog_row):
    """Assess one catalog record without inventing attraction facts."""

    place_id = catalog_row["place_id"]
    place = catalog_row["place"]
    city = catalog_row["city"]
    city_canonical = catalog_row["city_canonical"]

    attraction_type = catalog_row["attraction_type"]

    has_structured_metadata = _is_true(
        catalog_row["has_dataset2_metadata"]
    )

    has_attraction_type = _has_value(attraction_type)

    normalized_place = _normalize_name(place)

    normalized_cities = {
        _normalize_name(city),
        _normalize_name(city_canonical),
    }

    normalized_cities.discard("")

    is_city_name = (
        bool(normalized_place)
        and normalized_place in normalized_cities
    )

    result = {
        "place_id": place_id,
        "place": place,
        "city": city,
        "eligibility_status": None,
        "reason": None,
        "attraction_type": (
            str(attraction_type)
            if has_attraction_type
            else None
        ),
        "has_structured_attraction_metadata": (
            has_structured_metadata
        ),
        "requires_identity_verification": False,
        "eligible_for_automatic_stop_selection": False,
    }

    if is_city_name and not (
        has_structured_metadata and has_attraction_type
    ):
        result["eligibility_status"] = "EXCLUDE_CITY_LEVEL"
        result["reason"] = (
            "Place name matches its city, with no sufficient "
            "structured attraction identity."
        )

    elif is_city_name:
        result["eligibility_status"] = "NEEDS_VERIFICATION"
        result["reason"] = (
            "Place name matches its city despite attraction "
            "metadata; identity requires manual verification."
        )
        result["requires_identity_verification"] = True

    elif has_structured_metadata and has_attraction_type:
        result["eligibility_status"] = "CANDIDATE"
        result["reason"] = (
            "Structured attraction identity is available. "
            "Opening, access and routing still require "
            "separate verification."
        )
        result["eligible_for_automatic_stop_selection"] = True

    else:
        result["eligibility_status"] = "NEEDS_VERIFICATION"
        result["reason"] = (
            "Named catalog place lacks sufficient structured "
            "attraction identity."
        )
        result["requires_identity_verification"] = True

    return result


def assess_recommended_stops(recommendations, catalog):
    """Assess recommended place IDs in their existing V2 order."""

    if not isinstance(recommendations, pd.DataFrame):
        raise TypeError("Recommendations must be a DataFrame.")

    if not isinstance(catalog, pd.DataFrame):
        raise TypeError("Catalog must be a DataFrame.")

    if "place_id" not in recommendations.columns:
        raise ValueError("Recommendations lack place_id.")

    missing = REQUIRED_CATALOG_COLUMNS - set(catalog.columns)

    if missing:
        raise ValueError(
            f"Catalog missing required columns: {sorted(missing)}"
        )

    if not catalog["place_id"].is_unique:
        raise ValueError("Catalog contains duplicate place IDs.")

    catalog_lookup = catalog.set_index(
        "place_id",
        drop=False,
    )

    results = []

    for place_id in recommendations["place_id"].tolist():
        if place_id not in catalog_lookup.index:
            raise ValueError(
                f"Recommended place missing from catalog: "
                f"{place_id}"
            )

        catalog_row = catalog_lookup.loc[place_id]

        results.append(
            assess_stop_eligibility(catalog_row)
        )

    return results
