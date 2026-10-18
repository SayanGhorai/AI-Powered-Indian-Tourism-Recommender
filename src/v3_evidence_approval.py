from dataclasses import dataclass
from datetime import date
from pathlib import Path
import csv

from src.v3_evidence import AttractionEvidence
from src.v3_evidence_validator import EvidenceValidator
from src.v3_evidence_quality import analyze_evidence


APPROVAL_COLUMNS = [
    "decision_id",
    "evidence_id",
    "decision",
    "reviewer",
    "reviewed_on",
    "source_checked",
    "review_notes",
]

ALLOWED_DECISIONS = {"approved", "rejected"}


@dataclass(frozen=True)
class EvidenceDecision:
    decision_id: str
    evidence_id: str
    decision: str
    reviewer: str
    reviewed_on: str
    source_checked: bool
    review_notes: str


def validate_decision(
    decision: EvidenceDecision,
) -> None:
    """Validate approval metadata, not the truth of the source."""

    if not decision.decision_id.strip():
        raise ValueError("Decision ID is required.")

    if not decision.evidence_id.strip():
        raise ValueError("Evidence ID is required.")

    if decision.decision not in ALLOWED_DECISIONS:
        raise ValueError("Unsupported approval decision.")

    if not decision.reviewer.strip():
        raise ValueError("Reviewer identity is required.")

    if not decision.review_notes.strip():
        raise ValueError("Review notes are required.")

    reviewed = date.fromisoformat(decision.reviewed_on)

    if reviewed > date.today():
        raise ValueError("Review date cannot be in the future.")

    if not isinstance(decision.source_checked, bool):
        raise ValueError("source_checked must be boolean.")

    if decision.decision == "approved":
        if decision.source_checked is not True:
            raise ValueError(
                "Approval requires confirmed source review."
            )


def validate_approval_batch(
    evidence: list[AttractionEvidence],
    decisions: list[EvidenceDecision],
    validator: EvidenceValidator,
) -> dict:
    """Validate review decisions without changing evidence."""

    validator.validate_batch(evidence)

    evidence_by_id = {
        record.evidence_id: record
        for record in evidence
    }

    decision_ids = [d.decision_id for d in decisions]
    reviewed_ids = [d.evidence_id for d in decisions]

    if len(decision_ids) != len(set(decision_ids)):
        raise ValueError("Duplicate decision IDs.")

    if len(reviewed_ids) != len(set(reviewed_ids)):
        raise ValueError(
            "Multiple decisions for the same evidence record."
        )

    quality = analyze_evidence(evidence, validator)

    conflicting_ids = set()

    for issue in quality["issues"]:
        if issue.issue_type == "conflicting_values":
            conflicting_ids.update(issue.evidence_ids)

    for decision in decisions:
        validate_decision(decision)

        if decision.evidence_id not in evidence_by_id:
            raise ValueError(
                "Approval references unknown evidence."
            )

        record = evidence_by_id[decision.evidence_id]

        if record.status != "candidate":
            raise ValueError(
                "Only candidate evidence can receive a decision."
            )

        if (
            decision.decision == "approved"
            and decision.evidence_id in conflicting_ids
        ):
            raise ValueError(
                "Conflicting evidence requires resolution first."
            )

        if (
            decision.decision == "approved"
            and date.fromisoformat(decision.reviewed_on)
            < date.fromisoformat(record.observed_on)
        ):
            raise ValueError(
                "Approval cannot predate evidence observation."
            )

    return {
        "status": "PASS",
        "evidence_records": len(evidence),
        "decision_records": len(decisions),
        "approved": sum(
            d.decision == "approved" for d in decisions
        ),
        "rejected": sum(
            d.decision == "rejected" for d in decisions
        ),
        "conflicting_evidence_records": len(conflicting_ids),
    }


def read_approval_ledger(path: str | Path) -> list[EvidenceDecision]:
    """Read the existing approval ledger without modifying it."""

    with Path(path).open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames != APPROVAL_COLUMNS:
            raise ValueError("Unexpected approval ledger schema.")

        decisions = []

        for row in reader:
            if row["source_checked"] not in ("True", "False"):
                raise ValueError(
                    "Invalid source_checked value in ledger."
                )

            row["source_checked"] = (
                row["source_checked"] == "True"
            )

            decisions.append(EvidenceDecision(**row))

        return decisions


def run_approval_tests(
    validator: EvidenceValidator,
) -> dict:
    """Exercise approval rules using synthetic in-memory records."""

    from dataclasses import replace

    sample = AttractionEvidence(
        evidence_id="APPROVAL_TEST_001",
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
        notes="Synthetic test record.",
    )

    approval = EvidenceDecision(
        decision_id="DECISION_TEST_001",
        evidence_id=sample.evidence_id,
        decision="approved",
        reviewer="synthetic_test_reviewer",
        reviewed_on="2026-10-02",
        source_checked=True,
        review_notes="Synthetic source review test.",
    )

    checks = {}

    result = validate_approval_batch(
        [sample], [approval], validator
    )
    assert result["approved"] == 1
    checks["valid_approval_structure"] = "PASS"

    invalid_cases = {
        "missing_reviewer": replace(
            approval, reviewer=""
        ),
        "source_not_checked": replace(
            approval, source_checked=False
        ),
        "missing_review_notes": replace(
            approval, review_notes=""
        ),
        "future_review_date": replace(
            approval, reviewed_on="2099-01-01"
        ),
        "approval_predates_observation": replace(
            approval, reviewed_on="2026-09-30"
        ),
        "unknown_evidence_id": replace(
            approval, evidence_id="UNKNOWN_EVIDENCE"
        ),
    }

    for name, invalid in invalid_cases.items():
        try:
            validate_approval_batch(
                [sample], [invalid], validator
            )
        except ValueError:
            checks[name] = "REJECTED"
        else:
            raise AssertionError(
                f"Invalid approval accepted: {name}"
            )

    try:
        validate_approval_batch(
            [sample],
            [approval, replace(
                approval,
                decision_id="DECISION_TEST_002",
            )],
            validator,
        )
    except ValueError:
        checks["duplicate_review"] = "REJECTED"
    else:
        raise AssertionError("Duplicate review was accepted.")

    conflict = replace(
        sample,
        evidence_id="APPROVAL_TEST_002",
        value="60",
    )

    try:
        validate_approval_batch(
            [sample, conflict],
            [approval],
            validator,
        )
    except ValueError:
        checks["unresolved_conflict"] = "REJECTED"
    else:
        raise AssertionError(
            "Conflicting evidence was approved."
        )

    return checks
