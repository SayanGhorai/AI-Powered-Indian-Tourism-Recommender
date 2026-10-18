"""Conservative itinerary schedule constraints for V3.

Uses only approved attraction evidence.
Unknown schedules are never treated as confirmed open.
"""

from datetime import date, datetime, time
import re

from src.v3_attraction_profiles import (
    get_approved_attraction_profile,
)


WEEKDAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)


def _parse_time(value):
    """Parse a strict 24-hour HH:MM time string."""

    if not isinstance(value, str):
        raise ValueError("Visit time must be HH:MM text.")

    return datetime.strptime(value, "%H:%M").time()


def _parse_hours(value):
    """Parse a fixed opening window, such as 10:00-18:00."""

    match = re.fullmatch(
        r"(\d{2}:\d{2})-(\d{2}:\d{2})",
        value,
    )

    if not match:
        return None

    start = _parse_time(match.group(1))
    end = _parse_time(match.group(2))

    if start >= end:
        return None

    return start, end


def _schedule_applies(category, weekday):
    """Recognize explicit weekday and weekday-range categories."""

    category = category.strip().lower()

    if category in WEEKDAYS:
        return category == weekday

    match = re.fullmatch(
        r"([a-z]+)_to_([a-z]+)",
        category,
    )

    if not match:
        return False

    first, last = match.groups()

    if first not in WEEKDAYS or last not in WEEKDAYS:
        return False

    start_index = WEEKDAYS.index(first)
    end_index = WEEKDAYS.index(last)
    day_index = WEEKDAYS.index(weekday)

    if start_index <= end_index:
        return start_index <= day_index <= end_index

    return (
        day_index >= start_index
        or day_index <= end_index
    )


def check_attraction_visit(
    place_id,
    visit_date,
    visit_time,
):
    """Evaluate an attraction against approved scheduling evidence.

    OPEN_WINDOW_MATCH means the requested time falls inside the
    published schedule. It is not a real-time open guarantee.
    """

    if isinstance(visit_date, str):
        visit_date = date.fromisoformat(visit_date)

    if not isinstance(visit_date, date) or isinstance(
        visit_date, datetime
    ):
        raise ValueError("Visit date must be a date or YYYY-MM-DD.")

    requested_time = _parse_time(visit_time)
    weekday = WEEKDAYS[visit_date.weekday()]

    result = {
        "place_id": place_id,
        "visit_date": visit_date.isoformat(),
        "visit_time": visit_time,
        "weekday": weekday,
        "status": "UNKNOWN",
        "reason": "No applicable approved schedule.",
        "supporting_evidence_ids": [],
        "limitations": [
            "Holiday and special-event exceptions are not verified.",
            "Visit duration is not evaluated.",
            "Travel time is not evaluated.",
        ],
    }

    profile = get_approved_attraction_profile(place_id)

    if profile is None:
        result["reason"] = (
            "No approved attraction profile is available."
        )
        return result

    for closure in profile["weekly_off"]:
        closed_day = closure["value"].strip().lower()

        if (
            closure["category"] == "regular_closure"
            and closed_day == weekday
        ):
            result["status"] = "CLOSED"
            result["reason"] = (
                "Requested day matches an approved weekly closure."
            )
            result["supporting_evidence_ids"] = [
                closure["evidence_id"]
            ]
            return result

    applicable = []

    for schedule in profile["opening_hours"]:
        if not _schedule_applies(
            schedule["category"],
            weekday,
        ):
            continue

        parsed = _parse_hours(schedule["value"])

        if parsed is not None:
            applicable.append((schedule, parsed))

    if not applicable:
        result["reason"] = (
            "No applicable fixed opening-hours evidence."
        )
        return result

    inside = [
        schedule
        for schedule, (start, end) in applicable
        if start <= requested_time < end
    ]

    if inside:
        result["status"] = "OPEN_WINDOW_MATCH"
        result["reason"] = (
            "Requested time falls within approved "
            "published opening hours."
        )
        result["supporting_evidence_ids"] = [
            schedule["evidence_id"]
            for schedule in inside
        ]
    else:
        result["status"] = "OUTSIDE_PUBLISHED_HOURS"
        result["reason"] = (
            "Requested time is outside the applicable "
            "published opening hours."
        )
        result["supporting_evidence_ids"] = [
            schedule["evidence_id"]
            for schedule, _ in applicable
        ]

    return result
