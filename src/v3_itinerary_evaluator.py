"""Combined attraction-visit evaluation for the V3 itinerary planner.

Integrates approved schedule evidence and admission budgeting.
Does not infer travel time, holiday opening, or visit duration.
"""

from src.v3_itinerary_constraints import (
    check_attraction_visit,
)
from src.v3_itinerary_budget import (
    calculate_admission_budget,
)


def evaluate_attraction_visit(
    place_id,
    visit_date,
    visit_time,
    visitors,
):
    """Evaluate a proposed visit using approved V3 evidence."""

    schedule = check_attraction_visit(
        place_id=place_id,
        visit_date=visit_date,
        visit_time=visit_time,
    )

    budget = calculate_admission_budget(
        place_id=place_id,
        visitors=visitors,
    )

    schedule_status = schedule["status"]
    budget_status = budget["status"]

    if schedule_status == "CLOSED":
        decision = "NOT_FEASIBLE"
        reason = "Approved weekly closure conflicts with visit date."

    elif schedule_status == "OUTSIDE_PUBLISHED_HOURS":
        decision = "NOT_FEASIBLE"
        reason = "Visit time is outside approved opening hours."

    elif schedule_status == "UNKNOWN":
        decision = "UNVERIFIED"
        reason = "Approved schedule evidence is unavailable."

    elif schedule_status == "OPEN_WINDOW_MATCH":
        decision = "CONDITIONALLY_FEASIBLE"
        reason = (
            "Requested time matches approved published hours; "
            "holiday exceptions, duration and travel remain unverified."
        )

    else:
        raise ValueError(
            f"Unexpected schedule status: {schedule_status}"
        )

    if budget_status not in (
        "COMPLETE",
        "PARTIAL_UNKNOWN",
        "UNKNOWN",
    ):
        raise ValueError(
            f"Unexpected budget status: {budget_status}"
        )

    evidence_ids = list(dict.fromkeys(
        schedule["supporting_evidence_ids"]
        + budget["supporting_evidence_ids"]
    ))

    return {
        "place_id": place_id,
        "visit_date": schedule["visit_date"],
        "visit_time": schedule["visit_time"],
        "weekday": schedule["weekday"],
        "planning_decision": decision,
        "planning_reason": reason,
        "schedule_status": schedule_status,
        "budget_status": budget_status,
        "admission_total_inr": budget["total_inr"],
        "known_admission_subtotal_inr": (
            budget["known_subtotal_inr"]
        ),
        "missing_admission_categories": (
            budget["missing_categories"]
        ),
        "visitor_count": budget["total_visitors"],
        "schedule": schedule,
        "budget": budget,
        "supporting_evidence_ids": evidence_ids,
        "limitations": list(dict.fromkeys(
            schedule["limitations"] + budget["limitations"]
        )),
    }
