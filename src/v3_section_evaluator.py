"""Combined section-aware visit evaluation using approved evidence."""

from src.v3_attraction_profiles import (
    get_approved_attraction_profile,
)

from src.v3_section_constraints import (
    _evaluate_section_profile,
)

from src.v3_section_budget import (
    _calculate_section_profile_budget,
)


def _evaluate_section_with_profile(
    profile,
    place_id,
    section,
    visit_date,
    visit_time,
    visitors,
):
    """Internal evaluator supporting isolated synthetic tests."""

    schedule = _evaluate_section_profile(
        profile=profile,
        place_id=place_id,
        section=section,
        visit_date=visit_date,
        visit_time=visit_time,
    )

    budget = _calculate_section_profile_budget(
        profile=profile,
        place_id=place_id,
        section=section,
        visitors=visitors,
    )

    schedule_status = schedule["status"]
    budget_status = budget["status"]

    if schedule_status in (
        "CLOSED",
        "OUTSIDE_PUBLISHED_HOURS",
    ):
        decision = "NOT_FEASIBLE"
        reason = (
            "Requested visit conflicts with the "
            "section-specific published schedule."
        )

    elif schedule_status == "UNKNOWN":
        decision = "UNVERIFIED"
        reason = (
            "Approved section-specific scheduling "
            "evidence is unavailable."
        )

    elif schedule_status == "OPEN_WINDOW_MATCH":
        decision = "CONDITIONALLY_FEASIBLE"
        reason = (
            "Requested time matches approved section "
            "hours. Holiday exceptions, visit duration "
            "and travel remain unverified."
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

    limitations = list(dict.fromkeys(
        schedule["limitations"]
        + budget["limitations"]
    ))

    return {
        "place_id": place_id,
        "section": section,
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
        "limitations": limitations,
    }


def evaluate_section_visit(
    place_id,
    section,
    visit_date,
    visit_time,
    visitors,
):
    """Public evaluator: loads approved evidence only."""

    if not isinstance(place_id, str) or not place_id:
        raise ValueError("A valid place ID is required.")

    profile = get_approved_attraction_profile(
        place_id
    )

    return _evaluate_section_with_profile(
        profile=profile,
        place_id=place_id,
        section=section,
        visit_date=visit_date,
        visit_time=visit_time,
        visitors=visitors,
    )
