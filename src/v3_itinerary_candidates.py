"""Combine V2 rankings, V3 stop identity, and planning evidence.

This is candidate preparation, not route optimization.
"""

import pandas as pd

from src.v3_recommendation_adapter import (
    enrich_v2_pipeline_result,
)
from src.v3_stop_eligibility import (
    assess_recommended_stops,
)


def prepare_itinerary_candidates(
    v2_result,
    catalog,
    visit_date,
    visit_time,
    visitors,
):
    """Prepare conservative itinerary candidates from V2 output.

    Preserves all original recommendations in ranked order.
    Excludes city-level entries only from the selectable-stop list.
    """

    if not isinstance(v2_result, dict):
        raise TypeError("V2 result must be a dictionary.")

    if not isinstance(catalog, pd.DataFrame):
        raise TypeError("Catalog must be a DataFrame.")

    recommendations = v2_result.get("recommendations")

    if not isinstance(recommendations, pd.DataFrame):
        raise TypeError(
            "V2 recommendations must be a DataFrame."
        )

    planning = enrich_v2_pipeline_result(
        v2_result=v2_result,
        visit_date=visit_date,
        visit_time=visit_time,
        visitors=visitors,
    )

    eligibility = assess_recommended_stops(
        recommendations=recommendations,
        catalog=catalog,
    )

    planning_items = planning["v3_recommendations"]

    if len(planning_items) != len(eligibility):
        raise RuntimeError(
            "Planning and eligibility counts do not match."
        )

    combined = []

    for planning_item, eligibility_item in zip(
        planning_items,
        eligibility,
    ):
        if (
            planning_item["place_id"]
            != eligibility_item["place_id"]
        ):
            raise RuntimeError(
                "Planning and eligibility place IDs differ."
            )

        item = dict(planning_item)

        item["v3_stop_eligibility"] = (
            eligibility_item["eligibility_status"]
        )
        item["v3_stop_eligibility_reason"] = (
            eligibility_item["reason"]
        )
        item["v3_attraction_type"] = (
            eligibility_item["attraction_type"]
        )
        item["v3_identity_verification_required"] = (
            eligibility_item[
                "requires_identity_verification"
            ]
        )

        eligible_identity = (
            eligibility_item[
                "eligible_for_automatic_stop_selection"
            ]
        )

        planning_decision = item["v3_planning_decision"]

        if not eligible_identity:
            if (
                eligibility_item["eligibility_status"]
                == "EXCLUDE_CITY_LEVEL"
            ):
                selection_status = "EXCLUDED_CITY_LEVEL"
            else:
                selection_status = "HOLD_IDENTITY_VERIFICATION"

        elif planning_decision == "NOT_FEASIBLE":
            selection_status = "EXCLUDED_SCHEDULE"

        elif planning_decision == "CONDITIONALLY_FEASIBLE":
            selection_status = "CONDITIONAL_STOP"

        else:
            selection_status = "HOLD_OPERATIONAL_VERIFICATION"

        item["v3_selection_status"] = selection_status

        item["v3_selectable_now"] = (
            selection_status == "CONDITIONAL_STOP"
        )

        combined.append(item)

    selectable = [
        item for item in combined
        if item["v3_selectable_now"]
    ]

    return {
        "query": v2_result.get("query"),
        "detected_city": v2_result.get("detected_city"),
        "visit_date": visit_date,
        "visit_time": visit_time,
        "all_recommendations": combined,
        "selectable_stops": selectable,
        "summary": {
            "total_recommendations": len(combined),
            "conditional_stops": len(selectable),
            "hold_identity_verification": sum(
                item["v3_selection_status"]
                == "HOLD_IDENTITY_VERIFICATION"
                for item in combined
            ),
            "hold_operational_verification": sum(
                item["v3_selection_status"]
                == "HOLD_OPERATIONAL_VERIFICATION"
                for item in combined
            ),
            "excluded_city_level": sum(
                item["v3_selection_status"]
                == "EXCLUDED_CITY_LEVEL"
                for item in combined
            ),
            "excluded_schedule": sum(
                item["v3_selection_status"]
                == "EXCLUDED_SCHEDULE"
                for item in combined
            ),
        },
        "limitations": [
            "Conditional stops are not confirmed bookings.",
            "Holiday exceptions and special closures are not verified.",
            "No attraction-level route coordinates or travel times "
            "are inferred.",
            "Visit duration and full-day itinerary feasibility "
            "are not yet established.",
        ],
    }
