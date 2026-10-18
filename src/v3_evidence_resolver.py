import csv
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

from src.v3_evidence import (
    AttractionEvidence,
    EVIDENCE_COLUMNS,
)
from src.v3_evidence_validator import EvidenceValidator
from src.v3_evidence_approval import (
    EvidenceDecision,
    read_approval_ledger,
    validate_approval_batch,
)


def read_evidence_csv(path):
    """Read evidence without altering the source CSV."""

    with Path(path).open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames != EVIDENCE_COLUMNS:
            raise ValueError("Unexpected evidence CSV schema.")

        return [
            AttractionEvidence(**row)
            for row in reader
        ]


def resolve_approved_evidence(
    evidence,
    decisions,
    validator,
):
    """Expose only evidence with a valid approval decision."""

    validation = validate_approval_batch(
        evidence,
        decisions,
        validator,
    )

    decision_lookup = {
        decision.evidence_id: decision
        for decision in decisions
    }

    approved_records = []
    rejected_count = 0
    pending_count = 0

    for record in evidence:
        decision = decision_lookup.get(record.evidence_id)

        if decision is None:
            pending_count += 1
            continue

        if decision.decision == "rejected":
            rejected_count += 1
            continue

        if decision.decision != "approved":
            raise ValueError("Unexpected approval decision.")

        approved_records.append({
            "evidence": asdict(record),
            "approval": asdict(decision),
            "effective_status": "approved",
        })

    if len(approved_records) != validation["approved"]:
        raise AssertionError("Approval count mismatch.")

    return {
        "status": "PASS",
        "total_evidence": len(evidence),
        "approved_count": len(approved_records),
        "rejected_count": rejected_count,
        "pending_count": pending_count,
        "approved_records": approved_records,
    }


def load_current_approved_evidence(
    validator=None,
    evidence_path="v3_data/attraction_evidence.csv",
    ledger_path="v3_data/evidence_approval_ledger.csv",
):
    """Load and resolve current V3 evidence read-only."""

    if validator is None:
        validator = EvidenceValidator()

    evidence = read_evidence_csv(evidence_path)
    decisions = read_approval_ledger(ledger_path)

    return resolve_approved_evidence(
        evidence,
        decisions,
        validator,
    )


def run_resolver_tests(validator):
    """Use synthetic decisions; never write test approvals."""

    evidence = read_evidence_csv(
        "v3_data/attraction_evidence.csv"
    )

    if len(evidence) != 2:
        raise ValueError(
            "Expected two existing candidate evidence records."
        )

    first, second = evidence

    approval = EvidenceDecision(
        decision_id="RESOLVER_TEST_DECISION_001",
        evidence_id=first.evidence_id,
        decision="approved",
        reviewer="synthetic_test_reviewer",
        reviewed_on=date.today().isoformat(),
        source_checked=True,
        review_notes=(
            "In-memory structural test only; "
            "no real approval has occurred."
        ),
    )

    checks = {}

    empty = resolve_approved_evidence(
        evidence, [], validator
    )
    assert empty["approved_count"] == 0
    assert empty["pending_count"] == 2
    checks["unreviewed_evidence_hidden"] = "PASS"

    single = resolve_approved_evidence(
        [first], [approval], validator
    )
    assert single["approved_count"] == 1
    assert single["pending_count"] == 0
    checks["synthetic_approval_resolution"] = "PASS"

    mixed = resolve_approved_evidence(
        evidence, [approval], validator
    )
    assert mixed["approved_count"] == 1
    assert mixed["pending_count"] == 1
    checks["other_category_remains_pending"] = "PASS"

    approved = mixed["approved_records"][0]
    assert (
        approved["evidence"]["source_url"]
        == first.source_url
    )
    assert (
        approved["approval"]["decision_id"]
        == approval.decision_id
    )
    checks["source_and_review_provenance"] = "PASS"

    rejection = replace(
        approval,
        decision="rejected",
        source_checked=False,
    )

    rejected = resolve_approved_evidence(
        [first], [rejection], validator
    )
    assert rejected["approved_count"] == 0
    assert rejected["rejected_count"] == 1
    checks["rejected_evidence_hidden"] = "PASS"

    unknown = replace(
        approval,
        evidence_id="UNKNOWN_EVIDENCE_ID",
    )

    try:
        resolve_approved_evidence(
            [first], [unknown], validator
        )
    except ValueError:
        checks["unknown_approval_rejected"] = "PASS"
    else:
        raise AssertionError(
            "Approval of unknown evidence was accepted."
        )

    conflicting_record = replace(
        first,
        evidence_id="RESOLVER_TEST_CONFLICT_001",
        value="999999",
    )

    try:
        resolve_approved_evidence(
            [first, conflicting_record],
            [approval],
            validator,
        )
    except ValueError:
        checks["unresolved_conflict_blocked"] = "PASS"
    else:
        raise AssertionError(
            "Conflicting evidence was exposed."
        )

    duplicate_decision = replace(
        approval,
        decision_id="RESOLVER_TEST_DECISION_002",
    )

    try:
        resolve_approved_evidence(
            [first],
            [approval, duplicate_decision],
            validator,
        )
    except ValueError:
        checks["duplicate_decision_blocked"] = "PASS"
    else:
        raise AssertionError(
            "Duplicate review was accepted."
        )

    return checks
