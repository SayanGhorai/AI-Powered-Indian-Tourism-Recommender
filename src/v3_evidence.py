from dataclasses import dataclass
from datetime import date
from typing import Optional
from urllib.parse import urlparse
import math


EVIDENCE_TYPES = {
    "attraction_latitude",
    "attraction_longitude",
    "entrance_fee",
    "visit_duration",
    "opening_hours",
    "weekly_off",
}

EVIDENCE_STATUSES = {
    "candidate",
    "verified",
    "rejected",
    "superseded",
}

EVIDENCE_COLUMNS = [
    "evidence_id",
    "place_id",
    "city_canonical",
    "evidence_type",
    "category",
    "value",
    "unit",
    "source_name",
    "source_url",
    "observed_on",
    "status",
    "notes",
]


@dataclass(frozen=True)
class AttractionEvidence:
    evidence_id: str
    place_id: str
    city_canonical: str
    evidence_type: str
    category: str
    value: str
    unit: str
    source_name: str
    source_url: str
    observed_on: str
    status: str = "candidate"
    notes: str = ""


def validate_evidence(record: AttractionEvidence) -> None:
    """Validate evidence structure and basic value constraints."""

    if not record.evidence_id.strip():
        raise ValueError("Evidence ID is required.")

    if not record.place_id.startswith("INPLACE_"):
        raise ValueError("Invalid place ID format.")

    if not record.city_canonical.strip():
        raise ValueError("Canonical city is required.")

    if record.evidence_type not in EVIDENCE_TYPES:
        raise ValueError(
            f"Unsupported evidence type: {record.evidence_type}"
        )

    if record.status not in EVIDENCE_STATUSES:
        raise ValueError(
            f"Unsupported status: {record.status}"
        )

    if not record.category.strip():
        raise ValueError("Evidence category is required.")

    if not record.value.strip():
        raise ValueError("Evidence value is required.")

    if not record.source_name.strip():
        raise ValueError("Source name is required.")

    parsed = urlparse(record.source_url)

    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("A valid HTTPS source URL is required.")

    if not record.observed_on:
        raise ValueError("Observation date is required.")

    observation_date = date.fromisoformat(record.observed_on)

    if observation_date > date.today():
        raise ValueError("Observation date cannot be in the future.")

    numeric_types = {
        "attraction_latitude": (-90, 90, "degrees"),
        "attraction_longitude": (-180, 180, "degrees"),
        "entrance_fee": (0, None, "INR"),
        "visit_duration": (0, None, "hours"),
    }

    if record.evidence_type in numeric_types:
        minimum, maximum, expected_unit = numeric_types[
            record.evidence_type
        ]

        try:
            numeric_value = float(record.value)
        except ValueError as exc:
            raise ValueError(
                "Numeric evidence must contain a number."
            ) from exc

        if not math.isfinite(numeric_value):
            raise ValueError("Numeric evidence must be finite.")

        if numeric_value < minimum:
            raise ValueError("Evidence value below valid range.")

        if maximum is not None and numeric_value > maximum:
            raise ValueError("Evidence value above valid range.")

        if record.unit != expected_unit:
            raise ValueError(
                f"Expected unit: {expected_unit}"
            )

    if record.evidence_type == "entrance_fee":
        if record.category == "general":
            raise ValueError(
                "Entrance fees require a specific ticket category."
            )


def evidence_to_dict(record: AttractionEvidence) -> dict:
    """Return a validated evidence record as a dictionary."""
    validate_evidence(record)

    return {
        column: getattr(record, column)
        for column in EVIDENCE_COLUMNS
    }
