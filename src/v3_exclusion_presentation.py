"""Presentation-only grouping for held V3 itinerary entries.

Preserves the original held records and planning decisions.
"""

from copy import deepcopy

from src.v3_planning_classification import (
    classify_planning_status,
)


def group_presented_exclusions(held):
    """Separate geographic exclusions from attraction reviews."""

    if not isinstance(held, list):
        raise TypeError("Presented held entries must be a list.")

    review_items = []
    geographic_exclusions = []
    schedule_conflicts = []
    seen = set()

    for item in held:
        if not isinstance(item, dict):
            raise TypeError("Invalid presented held entry.")

        place_id = item.get("place_id")
        reason = item.get("reason")

        if not isinstance(place_id, str) or not place_id:
            raise ValueError("Missing place ID.")

        if not isinstance(reason, str) or not reason:
            raise ValueError("Missing held reason.")

        if place_id in seen:
            raise ValueError("Duplicate held place ID.")

        seen.add(place_id)

        classification = classify_planning_status(reason)

        category = classification["display_category"]

        if classification["is_provisional_stop"]:
            raise ValueError(
                "A held entry cannot be a provisional stop."
            )

        record = deepcopy(item)

        record["display_category"] = category
        record["display_message"] = classification[
            "display_message"
        ]

        if category == "NON_ATTRACTION_ENTRY":
            geographic_exclusions.append(record)

        elif category in (
            "IDENTITY_REVIEW",
            "OPERATIONAL_REVIEW",
        ):
            review_items.append(record)

        elif category == "KNOWN_SCHEDULE_CONFLICT":
            schedule_conflicts.append(record)

        else:
            raise ValueError(
                "Unsupported held display category: "
                + str(category)
            )

    return {
        "review_items": review_items,
        "geographic_exclusions": geographic_exclusions,
        "schedule_conflicts": schedule_conflicts,
        "total_not_planned": len(held),
    }
