"""Conservative itinerary planning for explicitly selected places.

User-selected places are NOT represented as V2 recommendations.
No coordinates, durations or travel times are invented.
"""

from copy import deepcopy
from decimal import Decimal, InvalidOperation

import pandas as pd

from src.v3_stop_eligibility import assess_stop_eligibility
from src.v3_itinerary_evaluator import evaluate_attraction_visit
from src.v3_section_evaluator import evaluate_section_visit


SECTION_AWARE_PLACES = {
    "INPLACE_07148": {"museum_galleries", "gardens"},
}


def plan_user_selected_visits(
    catalog,
    city_canonical,
    selections,
    visit_date,
    start_time,
):
    """Build a provisional plan in explicit user-selected order.

    Each selection:
      place_id: catalog ID
      visitors: ticket category -> visitor count
      section: required for section-aware attractions

    Only the first stop receives a proposed arrival time.
    """

    if not isinstance(catalog, pd.DataFrame):
        raise TypeError("Catalog must be a DataFrame.")

    if not isinstance(city_canonical, str) or not city_canonical.strip():
        raise ValueError("A canonical city is required.")

    if not isinstance(selections, list):
        raise TypeError("Selections must be a list.")

    if not selections:
        raise ValueError("Select at least one attraction.")

    required_catalog = {
        "place_id", "place", "city", "city_canonical",
        "attraction_type", "has_dataset2_metadata",
    }

    if not required_catalog.issubset(catalog.columns):
        raise ValueError("Catalog lacks required identity fields.")

    if not catalog["place_id"].is_unique:
        raise ValueError("Catalog contains duplicate place IDs.")

    requested_ids = []

    for selection in selections:
        if not isinstance(selection, dict):
            raise TypeError("Each selection must be a dictionary.")

        if not {"place_id", "visitors"}.issubset(selection):
            raise ValueError(
                "Each selection requires place_id and visitors."
            )

        requested_ids.append(selection["place_id"])

    if len(requested_ids) != len(set(requested_ids)):
        raise ValueError("Duplicate selected attractions.")

    lookup = catalog.set_index("place_id", drop=False)

    stops = []
    held = []
    known_subtotal = Decimal("0")

    for selection in selections:
        place_id = selection["place_id"]

        if place_id not in lookup.index:
            raise ValueError(
                f"Unknown catalog place ID: {place_id}"
            )

        row = lookup.loc[place_id]

        if row["city_canonical"] != city_canonical:
            raise ValueError(
                f"City constraint failed for {place_id}"
            )

        identity = assess_stop_eligibility(row)

        section = selection.get("section")
        visitors = selection["visitors"]

        if not isinstance(visitors, dict):
            raise TypeError(
                "Visitors must be a ticket-category dictionary."
            )

        if place_id in SECTION_AWARE_PLACES:
            if section not in SECTION_AWARE_PLACES[place_id]:
                raise ValueError(
                    f"Explicit valid section required for {place_id}"
                )

            evaluation = evaluate_section_visit(
                place_id=place_id,
                section=section,
                visit_date=visit_date,
                visit_time=start_time,
                visitors=visitors,
            )

        else:
            if section is not None:
                raise ValueError(
                    f"Unsupported section for {place_id}"
                )

            evaluation = evaluate_attraction_visit(
                place_id=place_id,
                visit_date=visit_date,
                visit_time=start_time,
                visitors=visitors,
            )

        identity_ok = (
            identity["eligibility_status"] == "CANDIDATE"
            and identity[
                "eligible_for_automatic_stop_selection"
            ] is True
        )

        decision = evaluation["planning_decision"]

        if not identity_ok:
            reason = "HOLD_IDENTITY_VERIFICATION"
        elif decision == "NOT_FEASIBLE":
            reason = "EXCLUDED_SCHEDULE"
        elif decision != "CONDITIONALLY_FEASIBLE":
            reason = "HOLD_OPERATIONAL_VERIFICATION"
        else:
            reason = None

        if reason is not None:
            held.append({
                "place_id": place_id,
                "place": str(row["place"]),
                "section": section,
                "selection_status": reason,
                "planning_decision": decision,
            })
            continue

        amount_text = evaluation["admission_total_inr"]

        if amount_text is None:
            amount = None
        else:
            try:
                amount = Decimal(str(amount_text))
            except (InvalidOperation, ValueError, TypeError):
                raise ValueError("Invalid admission amount.")

            if not amount.is_finite() or amount < 0:
                raise ValueError("Invalid admission amount.")

            known_subtotal += amount

        number = len(stops) + 1

        stops.append({
            "stop_number": number,
            "place_id": place_id,
            "place": str(row["place"]),
            "city": str(row["city"]),
            "visit_section": section,
            "selection_source": "EXPLICIT_USER_SELECTION",
            "proposed_arrival_time": (
                start_time if number == 1 else None
            ),
            "arrival_time_status": (
                "USER_REQUESTED_START"
                if number == 1 else "NOT_SCHEDULED"
            ),
            "preliminary_planning_decision": decision,
            "schedule_checked_at": start_time,
            "schedule_check_scope": (
                "REQUESTED_START_TIME_ONLY"
            ),
            "schedule_status": evaluation["schedule_status"],
            "admission_total_inr": amount_text,
            "admission_status": (
                "KNOWN" if amount is not None else "UNKNOWN"
            ),
            "supporting_evidence_ids": deepcopy(
                evaluation["supporting_evidence_ids"]
            ),
            "visit_duration_minutes": None,
            "travel_time_from_previous_minutes": None,
        })

    if not stops:
        status = "NO_CONDITIONAL_STOPS"
    elif len(stops) == 1:
        status = "SINGLE_STOP_DRAFT"
    else:
        status = "UNSCHEDULED_MULTI_STOP_DRAFT"

    return {
        "planning_mode": "USER_SELECTED",
        "city": city_canonical,
        "visit_date": visit_date,
        "requested_start_time": start_time,
        "draft_status": status,
        "stops": stops,
        "held_selections": held,
        "stop_count": len(stops),
        "known_admission_subtotal_inr": str(known_subtotal),
        "all_selected_admission_costs_known": (
            len(held) == 0
            and len(stops) == len(selections)
            and all(
                stop["admission_status"] == "KNOWN"
                for stop in stops
            )
        ),
        "route_optimized": False,
        "full_day_feasibility_verified": False,
        "limitations": [
            "Places were explicitly selected by the user, "
            "not returned or ranked by V2.",
            "Opening hours were preliminarily checked at "
            "the requested start time for each attraction.",
            "Only the first stop has a proposed arrival time.",
            "Later stop opening-hour feasibility is not "
            "verified without actual arrival times.",
            "Attraction-level travel times and visit "
            "durations are not verified.",
            "Holiday exceptions, ticket-counter cutoffs "
            "and live admission are not guaranteed.",
            "Known admission subtotal excludes travel, "
            "unknown charges and other expenses.",
        ],
    }
