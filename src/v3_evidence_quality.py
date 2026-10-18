from collections import defaultdict
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation

from src.v3_evidence import AttractionEvidence
from src.v3_evidence_validator import EvidenceValidator


NUMERIC_TYPES = {
    "attraction_latitude",
    "attraction_longitude",
    "entrance_fee",
    "visit_duration",
}

ACTIVE_STATUSES = {"candidate", "verified"}


@dataclass(frozen=True)
class EvidenceIssue:
    issue_type: str
    place_id: str
    evidence_type: str
    category: str
    evidence_ids: tuple[str, ...]
    values: tuple[str, ...]


def normalized_value(record: AttractionEvidence) -> str:
    """Normalize numeric values for comparison."""

    value = record.value.strip()

    if record.evidence_type in NUMERIC_TYPES:
        try:
            return str(Decimal(value).normalize())
        except InvalidOperation as exc:
            raise ValueError("Invalid numeric evidence.") from exc

    return " ".join(value.casefold().split())


def analyze_evidence(
    records: list[AttractionEvidence],
    validator: EvidenceValidator,
) -> dict:
    """Analyze valid records without modifying evidence."""

    validator.validate_batch(records)

    groups = defaultdict(list)

    for record in records:
        if record.status not in ACTIVE_STATUSES:
            continue

        key = (
            record.place_id,
            record.evidence_type,
            record.category.strip().casefold(),
            record.unit.strip().casefold(),
        )

        groups[key].append(record)

    issues = []

    for key, group in groups.items():
        if len(group) < 2:
            continue

        by_value = defaultdict(list)

        for record in group:
            by_value[normalized_value(record)].append(record)

        # Multiple observations of the same value.
        for value, matching in by_value.items():
            if len(matching) > 1:
                issues.append(
                    EvidenceIssue(
                        issue_type="duplicate_observation",
                        place_id=key[0],
                        evidence_type=key[1],
                        category=key[2],
                        evidence_ids=tuple(
                            r.evidence_id for r in matching
                        ),
                        values=(value,),
                    )
                )

        # Different values for the same attraction and category.
        if len(by_value) > 1:
            issues.append(
                EvidenceIssue(
                    issue_type="conflicting_values",
                    place_id=key[0],
                    evidence_type=key[1],
                    category=key[2],
                    evidence_ids=tuple(
                        r.evidence_id for r in group
                    ),
                    values=tuple(sorted(by_value)),
                )
            )

    return {
        "status": "PASS",
        "total_records": len(records),
        "active_records": sum(
            r.status in ACTIVE_STATUSES for r in records
        ),
        "duplicate_groups": sum(
            issue.issue_type == "duplicate_observation"
            for issue in issues
        ),
        "conflict_groups": sum(
            issue.issue_type == "conflicting_values"
            for issue in issues
        ),
        "issues": issues,
    }


def run_quality_tests(
    validator: EvidenceValidator,
) -> dict:
    """Run synthetic quality tests entirely in memory."""

    base = AttractionEvidence(
        evidence_id="QUALITY_TEST_001",
        place_id="INPLACE_07148",
        city_canonical="kolkata",
        evidence_type="entrance_fee",
        category="indian_national_museum_galleries",
        value="50",
        unit="INR",
        source_name="Victoria Memorial Hall",
        source_url="https://victoriamemorial-cal.org/visit-us/",
        observed_on="2026-10-01",
        status="candidate",
        notes="Synthetic test only.",
    )

    duplicate = replace(
        base,
        evidence_id="QUALITY_TEST_002",
        value="50.00",
    )

    conflict = replace(
        base,
        evidence_id="QUALITY_TEST_003",
        value="60",
    )

    different_category = replace(
        base,
        evidence_id="QUALITY_TEST_004",
        category="garden_daily_single_entry",
        value="30",
    )

    rejected = replace(
        base,
        evidence_id="QUALITY_TEST_005",
        value="75",
        status="rejected",
    )

    checks = {}

    result = analyze_evidence([base, duplicate], validator)
    assert result["duplicate_groups"] == 1
    assert result["conflict_groups"] == 0
    checks["numeric_duplicate_detection"] = "PASS"

    result = analyze_evidence([base, conflict], validator)
    assert result["duplicate_groups"] == 0
    assert result["conflict_groups"] == 1
    checks["conflicting_price_detection"] = "PASS"

    result = analyze_evidence(
        [base, different_category], validator
    )
    assert result["conflict_groups"] == 0
    assert result["duplicate_groups"] == 0
    checks["separate_ticket_categories"] = "PASS"

    result = analyze_evidence([base, rejected], validator)
    assert result["conflict_groups"] == 0
    assert result["active_records"] == 1
    checks["rejected_record_exclusion"] = "PASS"

    result = analyze_evidence(
        [base, duplicate, conflict], validator
    )
    assert result["duplicate_groups"] == 1
    assert result["conflict_groups"] == 1
    checks["combined_issue_detection"] = "PASS"

    return checks
