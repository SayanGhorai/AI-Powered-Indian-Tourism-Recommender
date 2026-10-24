
"""Nationwide V3 planning candidates.

Retains catalog attractions for consideration without
promoting them to operationally verified itinerary stops.
"""

import pandas as pd

from src.v3_stop_eligibility import assess_stop_eligibility


CANDIDATE_POLICY = {
    "CANDIDATE": "STRUCTURED_IDENTITY_CANDIDATE",
    "NEEDS_VERIFICATION": "IDENTITY_REVIEW_REQUIRED",
    "EXCLUDE_CITY_LEVEL": "GEOGRAPHIC_ENTITY_REVIEW",
}


def build_planning_candidates(catalog, city_canonical):
    """Build city-scoped candidate records in catalog order."""

    if not isinstance(catalog, pd.DataFrame):
        raise TypeError("Catalog must be a DataFrame.")

    required = {
        "place_id",
        "place",
        "city",
        "city_canonical",
        "attraction_type",
        "has_dataset2_metadata",
        "rating_mean",
        "review_count",
    }

    missing = required - set(catalog.columns)

    if missing:
        raise ValueError(
            f"Missing catalog columns: {sorted(missing)}"
        )

    if not catalog["place_id"].is_unique:
        raise ValueError("Duplicate catalog place IDs.")

    if not isinstance(city_canonical, str):
        raise TypeError("City must be a string.")

    city_canonical = city_canonical.strip()

    if not city_canonical:
        raise ValueError("City cannot be empty.")

    city_rows = catalog.loc[
        catalog["city_canonical"] == city_canonical
    ]

    if city_rows.empty:
        raise ValueError(
            f"Unknown catalog destination: {city_canonical}"
        )

    records = []

    for _, row in city_rows.iterrows():
        identity = assess_stop_eligibility(row)
        status = identity["eligibility_status"]

        if status not in CANDIDATE_POLICY:
            raise ValueError(
                f"Unexpected identity status: {status}"
            )

        rating = row["rating_mean"]
        reviews = row["review_count"]

        if pd.isna(rating) or pd.isna(reviews):
            raise ValueError(
                "Candidate lacks dataset rating evidence."
            )

        records.append({
            "place_id": str(row["place_id"]),
            "place": str(row["place"]),
            "city_canonical": city_canonical,
            "dataset_rating_mean": float(rating),
            "dataset_review_count": int(reviews),
            "identity_status": status,
            "candidate_category": CANDIDATE_POLICY[status],
            "identity_reason": identity["reason"],
            "eligible_for_automatic_stop_selection": (
                identity[
                    "eligible_for_automatic_stop_selection"
                ]
            ),
            "operational_verification_status": "NOT_ASSESSED",
            "is_confirmed_itinerary_stop": False,
            "is_confirmed_booking": False,
            "requires_geographic_entity_review": (
                status == "EXCLUDE_CITY_LEVEL"
            ),
        })

    return {
        "city": city_canonical,
        "catalog_count": len(city_rows),
        "candidates": records,
        "counts": {
            category: sum(
                item["candidate_category"] == category
                for item in records
            )
            for category in CANDIDATE_POLICY.values()
        },
    }
