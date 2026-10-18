"""Connect existing V2 recommendations to V3 itinerary evaluation.

V2 remains responsible for retrieval, reranking, personalization
and geographic constraints. V3 adds approved planning evidence.
"""

import pandas as pd

from src.v3_itinerary_evaluator import (
    evaluate_attraction_visit,
)


REQUIRED_COLUMNS = {
    "place_id",
    "place",
    "city",
    "recommendation_rank",
}


def enrich_v2_recommendations(
    recommendations,
    visit_date,
    visit_time,
    visitors,
    expected_city_canonical=None,
):
    """Attach V3 planning evaluations without reranking V2 results.

    expected_city_canonical is an optional additional safety check.
    It must be supplied by the caller from V2's detected_city.
    """

    if not isinstance(recommendations, pd.DataFrame):
        raise TypeError(
            "V2 recommendations must be a pandas DataFrame."
        )

    missing = REQUIRED_COLUMNS - set(recommendations.columns)

    if missing:
        raise ValueError(
            f"Missing V2 recommendation columns: {sorted(missing)}"
        )

    if expected_city_canonical is not None:
        if not isinstance(expected_city_canonical, str):
            raise TypeError(
                "Expected canonical city must be text."
            )

        if not expected_city_canonical.strip():
            raise ValueError(
                "Expected canonical city cannot be empty."
            )

        if "city_canonical" not in recommendations.columns:
            raise ValueError(
                "Canonical city column is required when "
                "enforcing a city constraint."
            )

        if not recommendations.empty:
            actual_cities = recommendations[
                "city_canonical"
            ].tolist()

            if any(
                city != expected_city_canonical
                for city in actual_cities
            ):
                raise ValueError(
                    "V3 city safety check failed: "
                    "recommendations include a different city."
                )

    enriched = []

    for row in recommendations.itertuples(index=False):
        record = row._asdict()

        place_id = record["place_id"]

        if not isinstance(place_id, str) or not place_id.strip():
            raise ValueError(
                "V2 recommendation has an invalid place_id."
            )

        evaluation = evaluate_attraction_visit(
            place_id=place_id,
            visit_date=visit_date,
            visit_time=visit_time,
            visitors=visitors,
        )

        record["v3_planning_decision"] = (
            evaluation["planning_decision"]
        )
        record["v3_schedule_status"] = (
            evaluation["schedule_status"]
        )
        record["v3_budget_status"] = (
            evaluation["budget_status"]
        )
        record["v3_admission_total_inr"] = (
            evaluation["admission_total_inr"]
        )
        record["v3_evidence_ids"] = (
            evaluation["supporting_evidence_ids"]
        )
        record["v3_evaluation"] = evaluation

        enriched.append(record)

    return enriched


def enrich_v2_pipeline_result(
    v2_result,
    visit_date,
    visit_time,
    visitors,
):
    """Enrich a real recommend_places() result.

    Returns a new dictionary. The original V2 result is unchanged.
    """

    if not isinstance(v2_result, dict):
        raise TypeError(
            "V2 pipeline result must be a dictionary."
        )

    if "recommendations" not in v2_result:
        raise ValueError(
            "V2 result has no recommendations field."
        )

    enriched = enrich_v2_recommendations(
        recommendations=v2_result["recommendations"],
        visit_date=visit_date,
        visit_time=visit_time,
        visitors=visitors,
        expected_city_canonical=v2_result.get(
            "detected_city"
        ),
    )

    return {
        "query": v2_result.get("query"),
        "detected_city": v2_result.get("detected_city"),
        "v3_recommendations": enriched,
        "v3_recommendation_count": len(enriched),
        "v3_summary": {
            "conditionally_feasible": sum(
                item["v3_planning_decision"]
                == "CONDITIONALLY_FEASIBLE"
                for item in enriched
            ),
            "not_feasible": sum(
                item["v3_planning_decision"]
                == "NOT_FEASIBLE"
                for item in enriched
            ),
            "unverified": sum(
                item["v3_planning_decision"]
                == "UNVERIFIED"
                for item in enriched
            ),
        },
    }
