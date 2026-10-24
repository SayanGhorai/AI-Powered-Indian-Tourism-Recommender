"""Read-only explanations for existing V3 visit evaluations.

No evidence approvals, planning decisions, or costs are changed.
"""

SCHEDULE_MESSAGES = {
    "OPEN_WINDOW_MATCH": (
        "Requested time matches approved published opening hours. "
        "Holiday exceptions and actual arrival feasibility remain unverified."
    ),
    "UNKNOWN": (
        "Approved opening-hour evidence is unavailable. "
        "This does not establish that the attraction is closed."
    ),
    "CLOSED": (
        "Approved weekly-closure evidence conflicts with the visit date."
    ),
    "OUTSIDE_PUBLISHED_HOURS": (
        "Requested time is outside approved published opening hours."
    ),
}

BUDGET_MESSAGES = {
    "COMPLETE": (
        "Admission is calculated for the selected visitor categories "
        "using existing approved evidence. Published prices may change."
    ),
    "PARTIAL_UNKNOWN": (
        "Some admission categories are unknown. "
        "The known subtotal is not the complete admission cost."
    ),
    "UNKNOWN": (
        "Admission for the selected visitor categories is unknown. "
        "A zero known subtotal does not mean free admission."
    ),
}

DECISION_MESSAGES = {
    "CONDITIONALLY_FEASIBLE": (
        "The requested visit is provisionally supported by "
        "approved schedule evidence, not confirmed as a booking."
    ),
    "UNVERIFIED": (
        "Operational evidence is insufficient to verify the visit."
    ),
    "NOT_FEASIBLE": (
        "Existing approved schedule evidence conflicts "
        "with the requested visit."
    ),
}


def explain_visit_uncertainty(evaluation):
    """Explain verified evaluator results without changing them."""

    if not isinstance(evaluation, dict):
        raise TypeError("Evaluation must be a dictionary.")

    schedule = evaluation.get("schedule_status")
    budget = evaluation.get("budget_status")
    decision = evaluation.get("planning_decision")

    if schedule not in SCHEDULE_MESSAGES:
        raise ValueError(f"Unknown schedule status: {schedule!r}")

    if budget not in BUDGET_MESSAGES:
        raise ValueError(f"Unknown budget status: {budget!r}")

    if decision not in DECISION_MESSAGES:
        raise ValueError(f"Unknown planning decision: {decision!r}")

    expected_decision = (
        "NOT_FEASIBLE"
        if schedule in ("CLOSED", "OUTSIDE_PUBLISHED_HOURS")
        else "UNVERIFIED"
        if schedule == "UNKNOWN"
        else "CONDITIONALLY_FEASIBLE"
    )

    if decision != expected_decision:
        raise ValueError(
            "Planning decision conflicts with schedule status."
        )

    total = evaluation.get("admission_total_inr")
    subtotal = evaluation.get("known_admission_subtotal_inr")
    missing = evaluation.get("missing_admission_categories")

    if missing is None:
        missing = []

    if not isinstance(missing, list):
        raise TypeError("Missing categories must be a list.")

    if budget == "COMPLETE" and (total is None or missing):
        raise ValueError("Complete budget lacks required evidence.")

    if budget == "UNKNOWN" and total is not None:
        raise ValueError("Unknown budget cannot have a total.")

    evidence_ids = evaluation.get("supporting_evidence_ids", [])

    if not isinstance(evidence_ids, list):
        raise TypeError("Evidence IDs must be a list.")

    return {
        "place_id": evaluation.get("place_id"),
        "planning_decision": decision,
        "schedule_status": schedule,
        "budget_status": budget,
        "schedule_explanation": SCHEDULE_MESSAGES[schedule],
        "admission_explanation": BUDGET_MESSAGES[budget],
        "planning_explanation": DECISION_MESSAGES[decision],
        "admission_total_inr": total if budget == "COMPLETE" else None,
        "known_admission_subtotal_inr": subtotal,
        "admission_total_verified": budget == "COMPLETE",
        "opening_window_supported": schedule == "OPEN_WINDOW_MATCH",
        "missing_admission_categories": list(missing),
        "supporting_evidence_ids": list(evidence_ids),
        "is_confirmed_booking": False,
        "route_optimized": False,
        "full_day_feasibility_verified": False,
    }
