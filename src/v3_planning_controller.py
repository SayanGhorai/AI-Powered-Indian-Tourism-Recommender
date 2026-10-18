"""Unified, evidence-grounded V3 planning controller.

Modes:
- recommend: real V2 recommendations followed by V3 evaluation
- user_selected: explicitly selected catalog attractions

This is deterministic orchestration, not an LLM agent.
"""

import pandas as pd

from src.retrieval import PLACES_MASTER_PATH
from src.v3_itinerary_pipeline import (
    build_v3_itinerary_from_v2,
)
from src.v3_user_selected_itinerary import (
    plan_user_selected_visits,
)


def run_v3_planning(
    mode,
    visit_date,
    start_time,
    *,
    query=None,
    city_canonical=None,
    selections=None,
    visitors=None,
    section_requests=None,
    catalog=None,
    v2_result=None,
    top_k=5,
    candidate_k=10,
):
    """Route a planning request without inventing recommendations."""

    if mode not in ("recommend", "user_selected"):
        raise ValueError(
            "mode must be 'recommend' or 'user_selected'."
        )

    if catalog is None:
        catalog = pd.read_parquet(PLACES_MASTER_PATH)

    if not isinstance(catalog, pd.DataFrame):
        raise TypeError("Catalog must be a DataFrame.")

    if mode == "user_selected":
        if not isinstance(city_canonical, str):
            raise ValueError(
                "User-selected mode requires city_canonical."
            )

        draft = plan_user_selected_visits(
            catalog=catalog,
            city_canonical=city_canonical,
            selections=selections,
            visit_date=visit_date,
            start_time=start_time,
        )

        return {
            "mode": "user_selected",
            "source": "EXPLICIT_USER_SELECTION",
            "city": city_canonical,
            "draft": draft,
            "prepared_candidates": None,
            "v2_result": None,
            "agent_status": "NOT_INVOKED",
            "route_optimized": False,
            "full_day_feasibility_verified": False,
        }

    if not isinstance(query, str) or not query.strip():
        raise ValueError(
            "Recommendation mode requires a travel query."
        )

    if visitors is None:
        raise ValueError(
            "Recommendation mode requires visitor categories."
        )

    if v2_result is None:
        from src.backend import recommend_places

        v2_result = recommend_places(
            query=query,
            top_k=top_k,
            candidate_k=candidate_k,
        )

    if not isinstance(v2_result, dict):
        raise TypeError("V2 result must be a dictionary.")

    info = v2_result.get("info", {})

    if not isinstance(info, dict):
        raise TypeError("V2 info must be a dictionary.")

    top_city = v2_result.get("detected_city")
    info_city = info.get("detected_city")

    if (
        top_city is not None
        and info_city is not None
        and top_city != info_city
    ):
        raise ValueError(
            "Conflicting detected cities in V2 output."
        )

    detected_city = info_city or top_city

    if not detected_city:
        raise ValueError(
            "V2 did not detect a city. "
            "City-scoped planning cannot proceed."
        )

    recommendations = v2_result.get("recommendations")

    if not isinstance(recommendations, pd.DataFrame):
        raise TypeError(
            "V2 recommendations must be a DataFrame."
        )

    normalized = dict(v2_result)
    normalized["query"] = query.strip()
    normalized["detected_city"] = detected_city
    normalized["recommendations"] = recommendations.copy(
        deep=True
    )

    if "recommendation_rank" not in (
        normalized["recommendations"].columns
    ):
        normalized["recommendations"][
            "recommendation_rank"
        ] = list(
            range(1, len(recommendations) + 1)
        )

    result = build_v3_itinerary_from_v2(
        v2_result=normalized,
        catalog=catalog,
        visit_date=visit_date,
        visit_time=start_time,
        visitors=visitors,
        section_requests=section_requests or {},
    )

    return {
        "mode": "recommend",
        "source": "V2_RANKED_RECOMMENDATIONS",
        "city": detected_city,
        "draft": result["itinerary_draft"],
        "prepared_candidates": result[
            "prepared_candidates"
        ],
        "v2_result": v2_result,
        "agent_status": "NOT_INVOKED",
        "route_optimized": False,
        "full_day_feasibility_verified": False,
    }
