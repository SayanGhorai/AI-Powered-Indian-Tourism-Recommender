"""Conservative V3 multi-stop scheduling assessment.

Preserves existing itinerary order and evidence limitations.
Does not invent arrival times, durations, routes or travel times.
"""


def assess_conservative_schedule(controller_result):
    """Assess scheduling completeness without changing the draft."""

    if not isinstance(controller_result, dict):
        raise TypeError("Controller result must be a dictionary.")

    draft = controller_result.get("draft")

    if not isinstance(draft, dict):
        raise TypeError("Planning draft must be a dictionary.")

    stops = draft.get("stops")

    if not isinstance(stops, list):
        raise TypeError("Draft stops must be a list.")

    if draft.get("route_optimized") is not False:
        raise ValueError("Unexpected route optimization claim.")

    if draft.get("full_day_feasibility_verified") is not False:
        raise ValueError("Unexpected full-day feasibility claim.")

    assessments = []

    for index, stop in enumerate(stops, start=1):
        if not isinstance(stop, dict):
            raise TypeError("Invalid itinerary stop.")

        if stop.get("stop_number") != index:
            raise ValueError("Stop numbering is inconsistent.")

        place_id = stop.get("place_id")

        if not isinstance(place_id, str) or not place_id:
            raise ValueError("Missing catalog place ID.")

        arrival = stop.get("proposed_arrival_time")
        arrival_status = stop.get("arrival_time_status")

        duration = stop.get("visit_duration_minutes")
        travel = stop.get(
            "travel_time_from_previous_minutes"
        )

        if index == 1:
            if arrival_status != "USER_REQUESTED_START":
                raise ValueError(
                    "First stop must retain requested-start status."
                )

            if arrival is None:
                raise ValueError(
                    "First stop lacks requested start time."
                )

        else:
            if arrival is not None:
                raise ValueError(
                    "Later arrival time is not verified."
                )

            if arrival_status != "NOT_SCHEDULED":
                raise ValueError(
                    "Later stop must remain unscheduled."
                )

        if duration is not None:
            raise ValueError(
                "Duration evidence integration is not yet supported."
            )

        if travel is not None:
            raise ValueError(
                "Travel-time evidence integration is not yet supported."
            )

        assessments.append({
            "stop_number": index,
            "place_id": place_id,
            "place": stop.get("place"),
            "requested_start_time": (
                arrival if index == 1 else None
            ),
            "arrival_status": arrival_status,
            "visit_duration_minutes": None,
            "travel_time_from_previous_minutes": None,
            "schedule_status": stop.get("schedule_status"),
            "schedule_check_scope": stop.get(
                "schedule_check_scope"
            ),
            "requires_duration_evidence": True,
            "requires_travel_time_evidence": index > 1,
            "later_arrival_verified": False,
            "route_optimized": False,
        })

    if not stops:
        status = "NO_SCHEDULABLE_STOPS"
    elif len(stops) == 1:
        status = "SINGLE_STOP_START_ONLY"
    else:
        status = "MULTI_STOP_UNSCHEDULED"

    return {
        "city": controller_result.get("city"),
        "mode": controller_result.get("mode"),
        "schedule_assessment_status": status,
        "stop_count": len(stops),
        "stops": assessments,
        "stop_order_preserved": True,
        "all_visit_durations_verified": False,
        "all_travel_times_verified": False,
        "route_optimized": False,
        "full_day_feasibility_verified": False,
        "limitations": [
            "Only the first stop has a requested start time.",
            "Later stop arrival times remain unscheduled.",
            "Visit durations have not been verified.",
            "Travel times between attractions have not been verified.",
            "Published opening hours do not establish live admission.",
            "No route optimization or full-day feasibility is claimed.",
        ],
    }
