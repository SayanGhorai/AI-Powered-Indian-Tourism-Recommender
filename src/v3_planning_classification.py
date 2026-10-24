
"""Shared V3 planning display classification.

Classifies existing planner decisions without changing
V2 rankings, operational evidence, or stop eligibility.
"""

from copy import deepcopy


STATUS_MAP = {
    "CONDITIONAL_STOP": "PROVISIONAL_STOP",
    "HOLD_IDENTITY_VERIFICATION": "IDENTITY_REVIEW",
    "HOLD_OPERATIONAL_VERIFICATION": "OPERATIONAL_REVIEW",
    "EXCLUDED_CITY_LEVEL": "NON_ATTRACTION_ENTRY",
    "EXCLUDED_SCHEDULE": "KNOWN_SCHEDULE_CONFLICT",
}

DISPLAY_MESSAGES = {
    "PROVISIONAL_STOP": (
        "Evidence-supported provisional stop. "
        "Travel and full-day feasibility remain unverified."
    ),
    "IDENTITY_REVIEW": (
        "Attraction identity requires verification "
        "before itinerary scheduling."
    ),
    "OPERATIONAL_REVIEW": (
        "Opening or operational evidence requires "
        "verification before itinerary scheduling."
    ),
    "NON_ATTRACTION_ENTRY": (
        "Catalog entry cannot currently be treated "
        "as a verified individual attraction."
    ),
    "KNOWN_SCHEDULE_CONFLICT": (
        "Approved schedule evidence conflicts "
        "with the requested visit."
    ),
}


def classify_planning_status(status):
    """Map a verified planner status to a display category."""

    if status not in STATUS_MAP:
        raise ValueError(
            f"Unknown V3 planning status: {status!r}"
        )

    category = STATUS_MAP[status]

    return {
        "source_status": status,
        "display_category": category,
        "display_message": DISPLAY_MESSAGES[category],
        "is_provisional_stop": (
            category == "PROVISIONAL_STOP"
        ),
        "is_confirmed_booking": False,
    }


def classify_itinerary_draft(controller_result):
    """Classify an existing V3 draft without modifying it."""

    if not isinstance(controller_result, dict):
        raise TypeError("Controller result must be a dictionary.")

    mode = controller_result.get("mode")

    if mode not in ("recommend", "user_selected"):
        raise ValueError(f"Unsupported planning mode: {mode!r}")

    draft = controller_result.get("draft")

    if not isinstance(draft, dict):
        raise TypeError("Planning draft must be a dictionary.")

    stops = draft.get("stops")
    held_key = (
        "held_recommendations"
        if mode == "recommend"
        else "held_selections"
    )
    held = draft.get(held_key)

    if not isinstance(stops, list):
        raise TypeError("Draft stops must be a list.")

    if not isinstance(held, list):
        raise TypeError("Held attractions must be a list.")

    classified = []
    seen_ids = set()

    for stop in stops:
        if not isinstance(stop, dict):
            raise TypeError("Stop must be a dictionary.")

        place_id = stop.get("place_id")

        if not isinstance(place_id, str) or not place_id:
            raise ValueError("Stop lacks a valid place_id.")

        if place_id in seen_ids:
            raise ValueError("Duplicate place_id in draft.")

        seen_ids.add(place_id)

        details = classify_planning_status(
            "CONDITIONAL_STOP"
        )

        classified.append({
            "place_id": place_id,
            "place": stop.get("place"),
            "original_record": deepcopy(stop),
            **details,
        })

    for item in held:
        if not isinstance(item, dict):
            raise TypeError("Held item must be a dictionary.")

        place_id = item.get("place_id")
        status = item.get("selection_status")

        if not isinstance(place_id, str) or not place_id:
            raise ValueError("Held item lacks a valid place_id.")

        if place_id in seen_ids:
            raise ValueError("Duplicate place_id in draft.")

        seen_ids.add(place_id)

        details = classify_planning_status(status)

        if details["is_provisional_stop"]:
            raise ValueError(
                "A held attraction cannot be a provisional stop."
            )

        classified.append({
            "place_id": place_id,
            "place": item.get("place"),
            "original_record": deepcopy(item),
            **details,
        })

    counts = {
        category: sum(
            item["display_category"] == category
            for item in classified
        )
        for category in DISPLAY_MESSAGES
    }

    if counts["PROVISIONAL_STOP"] != len(stops):
        raise RuntimeError("Provisional stop count mismatch.")

    return {
        "mode": mode,
        "city": controller_result.get("city"),
        "items": classified,
        "counts": counts,
        "provisional_stop_count": len(stops),
        "review_or_exclusion_count": len(held),
        "route_optimized": controller_result.get(
            "route_optimized"
        ),
        "full_day_feasibility_verified": controller_result.get(
            "full_day_feasibility_verified"
        ),
    }
