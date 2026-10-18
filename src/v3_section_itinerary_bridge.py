"""Apply explicit section choices to prepared V3 itinerary candidates.

Does not modify V2 ranking, V2 models, or the original
prepared candidate object.
"""

from copy import deepcopy

from src.v3_section_evaluator import evaluate_section_visit
from src.v3_itinerary_draft import build_itinerary_draft


def prepare_section_aware_itinerary(
    prepared_candidates,
    section_requests,
):
    """Return updated preparation and a provisional itinerary draft.

    section_requests format:
    {
        "PLACE_ID": {
            "section": "museum_galleries" or "gardens",
            "visitors": {"section_fee_category": count},
        }
    }
    """

    if not isinstance(prepared_candidates, dict):
        raise TypeError(
            "Prepared candidates must be a dictionary."
        )

    if not isinstance(section_requests, dict):
        raise TypeError(
            "Section requests must be a dictionary."
        )

    required = {
        "visit_date",
        "visit_time",
        "all_recommendations",
        "selectable_stops",
    }

    if not required.issubset(prepared_candidates):
        raise ValueError(
            "Prepared candidates are missing required fields."
        )

    result = deepcopy(prepared_candidates)

    all_items = result["all_recommendations"]

    if not isinstance(all_items, list):
        raise TypeError(
            "All recommendations must be a list."
        )

    ids = [item["place_id"] for item in all_items]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Duplicate recommended place IDs."
        )

    if not set(section_requests).issubset(set(ids)):
        raise ValueError(
            "Section requests contain places not recommended by V2."
        )

    expected_city = result.get("detected_city")

    for item in all_items:

        place_id = item["place_id"]

        if (
            expected_city is not None
            and item.get("city_canonical") != expected_city
        ):
            raise ValueError(
                "City constraint mismatch in prepared recommendations."
            )

        if place_id not in section_requests:
            continue

        request = section_requests[place_id]

        if not isinstance(request, dict):
            raise TypeError(
                "Each section request must be a dictionary."
            )

        if set(request) != {"section", "visitors"}:
            raise ValueError(
                "Each section request requires exactly "
                "'section' and 'visitors'."
            )

        evaluation = evaluate_section_visit(
            place_id=place_id,
            section=request["section"],
            visit_date=result["visit_date"],
            visit_time=result["visit_time"],
            visitors=request["visitors"],
        )

        item["v3_visit_section"] = request["section"]
        item["v3_section_visitors"] = deepcopy(
            request["visitors"]
        )

        item["v3_planning_decision"] = (
            evaluation["planning_decision"]
        )
        item["v3_schedule_status"] = (
            evaluation["schedule_status"]
        )
        item["v3_budget_status"] = (
            evaluation["budget_status"]
        )
        item["v3_admission_total_inr"] = (
            evaluation["admission_total_inr"]
        )
        item["v3_evidence_ids"] = (
            evaluation["supporting_evidence_ids"]
        )
        item["v3_evaluation"] = evaluation

        eligibility = item.get("v3_stop_eligibility")

        identity_ok = (
            eligibility == "CANDIDATE"
            and item.get(
                "v3_identity_verification_required"
            ) is False
        )

        if eligibility == "EXCLUDE_CITY_LEVEL":
            selection = "EXCLUDED_CITY_LEVEL"

        elif not identity_ok:
            selection = "HOLD_IDENTITY_VERIFICATION"

        elif (
            evaluation["planning_decision"]
            == "NOT_FEASIBLE"
        ):
            selection = "EXCLUDED_SCHEDULE"

        elif (
            evaluation["planning_decision"]
            == "CONDITIONALLY_FEASIBLE"
        ):
            selection = "CONDITIONAL_STOP"

        else:
            selection = "HOLD_OPERATIONAL_VERIFICATION"

        item["v3_selection_status"] = selection
        item["v3_selectable_now"] = (
            selection == "CONDITIONAL_STOP"
        )

    result["selectable_stops"] = [
        item
        for item in all_items
        if item.get("v3_selection_status")
        == "CONDITIONAL_STOP"
        and item.get("v3_selectable_now") is True
    ]

    result["summary"] = {
        "total_recommendations": len(all_items),
        "conditional_stops": len(
            result["selectable_stops"]
        ),
        "hold_identity_verification": sum(
            item.get("v3_selection_status")
            == "HOLD_IDENTITY_VERIFICATION"
            for item in all_items
        ),
        "hold_operational_verification": sum(
            item.get("v3_selection_status")
            == "HOLD_OPERATIONAL_VERIFICATION"
            for item in all_items
        ),
        "excluded_city_level": sum(
            item.get("v3_selection_status")
            == "EXCLUDED_CITY_LEVEL"
            for item in all_items
        ),
        "excluded_schedule": sum(
            item.get("v3_selection_status")
            == "EXCLUDED_SCHEDULE"
            for item in all_items
        ),
    }

    draft = build_itinerary_draft(result)

    section_by_id = {
        item["place_id"]: item.get("v3_visit_section")
        for item in all_items
    }

    for stop in draft["stops"]:
        stop["visit_section"] = section_by_id[
            stop["place_id"]
        ]

    for held in draft["held_recommendations"]:
        held["visit_section"] = section_by_id[
            held["place_id"]
        ]

    draft["limitations"].append(
        "Section-specific admission calculations apply "
        "only to the explicitly selected visitor section."
    )

    return {
        "prepared_candidates": result,
        "itinerary_draft": draft,
    }
