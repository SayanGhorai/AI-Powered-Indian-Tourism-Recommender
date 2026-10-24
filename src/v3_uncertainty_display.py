"""Evidence-aware uncertainty display adapter.

Uses the original user selections and approved evaluators.
Does not change planner eligibility, evidence, or admissions.
"""

from src.v3_itinerary_evaluator import evaluate_attraction_visit
from src.v3_section_evaluator import evaluate_section_visit
from src.v3_planning_uncertainty import explain_visit_uncertainty


SECTION_AWARE_PLACES = {
    "INPLACE_07148": {"museum_galleries", "gardens"},
}


def build_selection_uncertainty(
    selections,
    visit_date,
    start_time,
):
    """Explain each explicit selection using its actual inputs."""

    if not isinstance(selections, list):
        raise TypeError("Selections must be a list.")

    results = []
    seen = set()

    for selection in selections:
        if not isinstance(selection, dict):
            raise TypeError("Invalid selection.")

        place_id = selection.get("place_id")
        section = selection.get("section")
        visitors = selection.get("visitors")

        if not isinstance(place_id, str) or not place_id:
            raise ValueError("Invalid place ID.")

        if not isinstance(visitors, dict) or not visitors:
            raise ValueError("Visitor categories are required.")

        if place_id in SECTION_AWARE_PLACES:
            if section not in SECTION_AWARE_PLACES[place_id]:
                raise ValueError("Valid attraction section required.")

            evaluation = evaluate_section_visit(
                place_id=place_id,
                section=section,
                visit_date=visit_date,
                visit_time=start_time,
                visitors=dict(visitors),
            )
        else:
            if section is not None:
                raise ValueError(
                    "Unexpected section for ordinary attraction."
                )

            evaluation = evaluate_attraction_visit(
                place_id=place_id,
                visit_date=visit_date,
                visit_time=start_time,
                visitors=dict(visitors),
            )

        identity = (place_id, section)

        if identity in seen:
            raise ValueError("Duplicate attraction-section selection.")

        seen.add(identity)

        explanation = explain_visit_uncertainty(evaluation)

        results.append({
            "place_id": place_id,
            "section": section,
            "visitor_categories": dict(visitors),
            "schedule_status": explanation["schedule_status"],
            "budget_status": explanation["budget_status"],
            "planning_decision": explanation["planning_decision"],
            "schedule_explanation": explanation["schedule_explanation"],
            "admission_explanation": explanation["admission_explanation"],
            "planning_explanation": explanation["planning_explanation"],
            "admission_total_inr": explanation["admission_total_inr"],
            "admission_total_verified": explanation[
                "admission_total_verified"
            ],
            "supporting_evidence_ids": list(
                explanation["supporting_evidence_ids"]
            ),
            "is_confirmed_booking": False,
            "route_optimized": False,
            "full_day_feasibility_verified": False,
        })

    return results
