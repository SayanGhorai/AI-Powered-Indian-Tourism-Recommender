"""Conservative evidence-grounded itinerary draft generation.

This module does not optimize routes or invent travel times,
visit durations, attraction coordinates or missing costs.
"""

from decimal import Decimal, InvalidOperation


def build_itinerary_draft(prepared_candidates):
    """Build a provisional itinerary from prepared V3 candidates."""

    if not isinstance(prepared_candidates, dict):
        raise TypeError(
            "Prepared candidates must be a dictionary."
        )

    required = {
        "visit_date",
        "visit_time",
        "all_recommendations",
        "selectable_stops",
    }

    missing = required - set(prepared_candidates)

    if missing:
        raise ValueError(
            f"Missing prepared candidate fields: {sorted(missing)}"
        )

    all_items = prepared_candidates["all_recommendations"]
    selectable = prepared_candidates["selectable_stops"]

    if not isinstance(all_items, list):
        raise TypeError("All recommendations must be a list.")

    if not isinstance(selectable, list):
        raise TypeError("Selectable stops must be a list.")

    all_ids = [item["place_id"] for item in all_items]

    if len(all_ids) != len(set(all_ids)):
        raise ValueError(
            "Duplicate place IDs in prepared recommendations."
        )

    selected_ids = [item["place_id"] for item in selectable]

    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError(
            "Duplicate place IDs in selectable stops."
        )

    if not set(selected_ids).issubset(set(all_ids)):
        raise ValueError(
            "Selectable stops contain an unknown place ID."
        )

    for item in selectable:
        if (
            item.get("v3_selection_status")
            != "CONDITIONAL_STOP"
            or item.get("v3_selectable_now") is not True
        ):
            raise ValueError(
                "Selectable stops must have verified "
                "conditional selection status."
            )

    # Preserve original recommendation order even if the
    # selectable list arrives in a different order.
    selectable_lookup = {
        item["place_id"]: item for item in selectable
    }

    ordered_selectable = [
        selectable_lookup[item["place_id"]]
        for item in all_items
        if item["place_id"] in selectable_lookup
    ]

    stops = []
    known_total = Decimal("0")

    for number, item in enumerate(
        ordered_selectable,
        start=1,
    ):
        admission = item.get(
            "v3_admission_total_inr"
        )

        if admission is None:
            admission_status = "UNKNOWN"
        else:
            try:
                amount = Decimal(str(admission))
            except (InvalidOperation, ValueError, TypeError):
                raise ValueError(
                    "Invalid admission amount for "
                    f"{item['place_id']}"
                )

            if not amount.is_finite() or amount < 0:
                raise ValueError(
                    "Admission amount must be finite "
                    "and nonnegative."
                )

            known_total += amount
            admission_status = "KNOWN"

        stops.append({
            "stop_number": number,
            "place_id": item["place_id"],
            "place": item["place"],
            "city": item["city"],
            "original_recommendation_rank": (
                item["recommendation_rank"]
            ),
            "proposed_arrival_time": (
                prepared_candidates["visit_time"]
                if number == 1
                else None
            ),
            "arrival_time_status": (
                "USER_REQUESTED_START"
                if number == 1
                else "NOT_SCHEDULED"
            ),
            "planning_status": (
                "CONDITIONALLY_FEASIBLE"
            ),
            "schedule_status": item[
                "v3_schedule_status"
            ],
            "admission_total_inr": admission,
            "admission_status": admission_status,
            "supporting_evidence_ids": item.get(
                "v3_evidence_ids",
                [],
            ),
            "visit_duration_minutes": None,
            "travel_time_from_previous_minutes": None,
        })

    held = [
        {
            "place_id": item["place_id"],
            "place": item["place"],
            "recommendation_rank": (
                item["recommendation_rank"]
            ),
            "selection_status": (
                item["v3_selection_status"]
            ),
        }
        for item in all_items
        if item["place_id"] not in selected_ids
    ]

    all_costs_known = all(
        stop["admission_status"] == "KNOWN"
        for stop in stops
    )

    if not stops:
        draft_status = "NO_CONDITIONAL_STOPS"
    elif len(stops) == 1:
        draft_status = "SINGLE_STOP_DRAFT"
    else:
        draft_status = "UNSCHEDULED_MULTI_STOP_DRAFT"

    return {
        "query": prepared_candidates.get("query"),
        "city": prepared_candidates.get("detected_city"),
        "visit_date": prepared_candidates["visit_date"],
        "requested_start_time": (
            prepared_candidates["visit_time"]
        ),
        "draft_status": draft_status,
        "stops": stops,
        "held_recommendations": held,
        "stop_count": len(stops),
        "known_admission_subtotal_inr": (
            str(known_total)
        ),
        "all_selected_admission_costs_known": (
            all_costs_known
        ),
        "route_optimized": False,
        "full_day_feasibility_verified": False,
        "limitations": [
            "Opening-hour checks use approved regular "
            "schedules, not live admission confirmation.",
            "Special closures and holiday exceptions "
            "have not been verified.",
            "No verified attraction-level coordinates "
            "or inter-stop travel times are available.",
            "Visit durations have not been verified.",
            "Only the first stop has a proposed arrival "
            "time; later stops are not scheduled.",
            "Known admission subtotal excludes unknown "
            "admission charges and travel expenses.",
        ],
    }
