
"""Nationwide catalog choices for the V3 Streamlit interface.

This module builds destination and attraction choices.
It does not approve geographic identities, operational
evidence, or automatic itinerary scheduling.
"""

import pandas as pd

from src.v3_stop_eligibility import (
    REQUIRED_CATALOG_COLUMNS,
    assess_stop_eligibility,
)


def _validate_catalog(catalog):
    if not isinstance(catalog, pd.DataFrame):
        raise TypeError("Catalog must be a DataFrame.")

    missing = REQUIRED_CATALOG_COLUMNS - set(catalog.columns)

    if missing:
        raise ValueError(
            f"Missing catalog columns: {sorted(missing)}"
        )

    if not catalog["place_id"].is_unique:
        raise ValueError("Duplicate catalog place IDs.")

    if catalog["place_id"].isna().any():
        raise ValueError("Null catalog place IDs.")

    if catalog["city_canonical"].isna().any():
        raise ValueError("Null canonical locations.")


def build_destination_options(catalog):
    """Return all canonical destinations and catalog counts."""

    _validate_catalog(catalog)

    counts = (
        catalog.groupby("city_canonical")
        .size()
        .sort_index()
    )

    results = []

    for city, count in counts.items():
        city = str(city)

        results.append({
            "city_canonical": city,
            "place_count": int(count),
            "label": (
                f"{city.title()} "
                f"({int(count)} catalog places)"
            ),
        })

    return results


def build_attraction_options(catalog, city_canonical):
    """Return city-scoped choices with identity warnings.

    City-level entries identified by the existing
    eligibility policy are excluded from the choices.

    NEEDS_VERIFICATION entries may be chosen for
    review, but are not automatically itinerary-ready.
    """

    _validate_catalog(catalog)

    if not isinstance(city_canonical, str):
        raise TypeError("Canonical location must be text.")

    if not city_canonical.strip():
        raise ValueError("Canonical location is required.")

    selected = catalog.loc[
        catalog["city_canonical"] == city_canonical
    ]

    if selected.empty:
        raise ValueError(
            f"No catalog entries for: {city_canonical}"
        )

    selected = selected.sort_values(
        ["place", "place_id"],
        kind="stable",
    )

    options = []

    for _, row in selected.iterrows():
        identity = assess_stop_eligibility(row)

        status = identity["eligibility_status"]

        if status == "EXCLUDE_CITY_LEVEL":
            continue

        place_id = str(row["place_id"])
        place_name = str(row["place"])

        if status == "CANDIDATE":
            label_status = "Identity metadata available"
        else:
            label_status = "Identity verification required"

        options.append({
            "place_id": place_id,
            "place": place_name,
            "city_canonical": city_canonical,
            "identity_status": status,
            "identity_reason": identity["reason"],
            "requires_identity_verification": bool(
                identity["requires_identity_verification"]
            ),
            "identity_candidate": bool(
                identity[
                    "eligible_for_automatic_stop_selection"
                ]
            ),
            "label": (
                f"{place_name} | {place_id} "
                f"| {label_status}"
            ),
        })

    return options
