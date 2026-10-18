"""V3 itinerary preparation from an existing V2 pipeline result.

This module does not call, retrain, or modify V2 models.
"""

from src.v3_itinerary_candidates import (
    prepare_itinerary_candidates,
)

from src.v3_section_itinerary_bridge import (
    prepare_section_aware_itinerary,
)


def build_v3_itinerary_from_v2(
    v2_result,
    catalog,
    visit_date,
    visit_time,
    visitors,
    section_requests=None,
):
    """Build a conservative V3 draft from V2 recommendations.

    section_requests is an explicit mapping of recommended
    place IDs to section and visitor-category selections.

    The resulting draft is not a route-optimized or
    fully scheduled itinerary.
    """

    if section_requests is None:
        section_requests = {}

    if not isinstance(section_requests, dict):
        raise TypeError(
            "section_requests must be a dictionary."
        )

    prepared = prepare_itinerary_candidates(
        v2_result=v2_result,
        catalog=catalog,
        visit_date=visit_date,
        visit_time=visit_time,
        visitors=visitors,
    )

    output = prepare_section_aware_itinerary(
        prepared_candidates=prepared,
        section_requests=section_requests,
    )

    draft = output["itinerary_draft"]

    return {
        "query": v2_result.get("query"),
        "detected_city": v2_result.get("detected_city"),
        "prepared_candidates": output[
            "prepared_candidates"
        ],
        "itinerary_draft": draft,
        "integration_status": "PASS",
        "schedule_verified": False,
        "route_optimized": False,
    }
