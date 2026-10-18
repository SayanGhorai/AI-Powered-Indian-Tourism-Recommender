"""Section-aware schedule checks using approved V3 evidence.

The public function loads approved profiles only.
The private helper supports isolated synthetic testing.
"""

from datetime import date, datetime, time

from src.v3_attraction_profiles import (
    get_approved_attraction_profile,
)


SECTION_RULES = {
    "museum_galleries": {
        "hours": "museum_galleries_tuesday_to_sunday",
        "closure": "museum_galleries_regular_closure",
        "days": {
            "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday",
        },
    },
    "gardens": {
        "hours": "gardens_all_days",
        "closure": None,
        "days": {
            "monday", "tuesday", "wednesday",
            "thursday", "friday", "saturday",
            "sunday",
        },
    },
}


def _parse_requested_date(value):
    if isinstance(value, str):
        value = date.fromisoformat(value)

    if not isinstance(value, date) or isinstance(
        value, datetime
    ):
        raise ValueError(
            "Visit date must be YYYY-MM-DD or a date."
        )

    return value


def _parse_clock(value):
    if not isinstance(value, str):
        raise ValueError("Time must use HH:MM format.")

    try:
        parsed = datetime.strptime(value, "%H:%M")
    except ValueError as exc:
        raise ValueError(
            "Time must use HH:MM format."
        ) from exc

    return parsed.time()


def _evaluate_section_profile(
    profile,
    place_id,
    section,
    visit_date,
    visit_time,
):
    """Internal evaluator; profile injection is for testing."""

    if section not in SECTION_RULES:
        raise ValueError(
            f"Unsupported visitor section: {section}"
        )

    requested_date = _parse_requested_date(
        visit_date
    )
    requested_time = _parse_clock(
        visit_time
    )

    weekday = requested_date.strftime(
        "%A"
    ).lower()

    rules = SECTION_RULES[section]

    result = {
        "place_id": place_id,
        "section": section,
        "visit_date": requested_date.isoformat(),
        "visit_time": visit_time,
        "weekday": weekday,
        "status": "UNKNOWN",
        "reason": (
            "No applicable approved section schedule."
        ),
        "supporting_evidence_ids": [],
        "limitations": [
            "Special closures and holiday exceptions "
            "are not verified.",
            "Ticket-counter closing times are not "
            "evaluated.",
            "Visit duration and travel time are "
            "not evaluated.",
        ],
    }

    if profile is None:
        result["reason"] = (
            "No approved attraction profile is available."
        )
        return result

    if profile.get("place_id") != place_id:
        raise ValueError(
            "Profile identity does not match place ID."
        )

    closures = [
        item
        for item in profile.get("weekly_off", [])
        if item.get("category") == rules["closure"]
        and rules["closure"] is not None
    ]

    matching_closures = [
        item
        for item in closures
        if str(item.get("value", "")).strip().lower()
        == weekday
    ]

    if matching_closures:
        result["status"] = "CLOSED"
        result["reason"] = (
            "Requested weekday matches the "
            "section-specific approved closure."
        )
        result["supporting_evidence_ids"] = [
            item["evidence_id"]
            for item in matching_closures
        ]
        return result

    if weekday not in rules["days"]:
        result["reason"] = (
            "No approved opening schedule applies "
            "to this section on the requested day."
        )
        return result

    schedules = [
        item
        for item in profile.get("opening_hours", [])
        if item.get("category") == rules["hours"]
    ]

    if len(schedules) != 1:
        result["reason"] = (
            "Exactly one approved schedule is required "
            "for this visitor section."
        )
        return result

    schedule = schedules[0]
    value = str(schedule.get("value", ""))

    try:
        start_text, end_text = value.split("-", 1)
        start = _parse_clock(start_text.strip())
        end = _parse_clock(end_text.strip())
    except (ValueError, AttributeError):
        result["reason"] = (
            "Approved schedule cannot be parsed safely."
        )
        return result

    if start >= end:
        result["reason"] = (
            "Overnight or invalid opening window "
            "requires separate evaluation."
        )
        return result

    result["supporting_evidence_ids"] = [
        schedule["evidence_id"]
    ]

    if start <= requested_time < end:
        result["status"] = "OPEN_WINDOW_MATCH"
        result["reason"] = (
            "Requested time falls within the approved "
            "section-specific published hours."
        )
    else:
        result["status"] = "OUTSIDE_PUBLISHED_HOURS"
        result["reason"] = (
            "Requested time falls outside the approved "
            "section-specific published hours."
        )

    return result


def check_section_visit(
    place_id,
    section,
    visit_date,
    visit_time,
):
    """Public section check: approved evidence only."""

    if not isinstance(place_id, str) or not place_id:
        raise ValueError("A valid place ID is required.")

    profile = get_approved_attraction_profile(
        place_id
    )

    return _evaluate_section_profile(
        profile=profile,
        place_id=place_id,
        section=section,
        visit_date=visit_date,
        visit_time=visit_time,
    )
