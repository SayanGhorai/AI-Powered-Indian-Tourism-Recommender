from dataclasses import replace
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd

from src.v3_evidence import (
    AttractionEvidence,
    EVIDENCE_COLUMNS,
    validate_evidence,
)


def normalize_source_url(url: str) -> str:
    """Normalize safe URL variations without changing the URL path."""
    parsed = urlsplit(url.strip())

    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ValueError("Source must use HTTPS.")

    if parsed.username or parsed.password:
        raise ValueError("Source URLs must not contain credentials.")

    if parsed.port not in (None, 443):
        raise ValueError("Unexpected source URL port.")

    hostname = parsed.hostname.lower()

    if parsed.port == 443:
        hostname += ":443"

    path = parsed.path.rstrip("/") or "/"

    return urlunsplit(
        ("https", hostname, path, parsed.query, "")
    )


class EvidenceValidator:
    """Validate attraction evidence against local catalog and sources."""

    def __init__(
        self,
        seed_path="v3_data/attraction_enrichment_seed.csv",
        registry_path="v3_data/source_registry.csv",
    ):
        self.seed = pd.read_csv(seed_path, encoding="utf-8-sig")
        self.registry = pd.read_csv(
            registry_path, encoding="utf-8-sig"
        )

        seed_required = {
            "place_id",
            "city_canonical",
            "catalog_place_name",
        }
        registry_required = {
            "place_id",
            "source_name",
            "source_url",
            "review_status",
        }

        if not seed_required.issubset(self.seed.columns):
            raise ValueError("Seed is missing required columns.")

        if not registry_required.issubset(self.registry.columns):
            raise ValueError("Registry is missing required columns.")

        if self.seed["place_id"].isna().any():
            raise ValueError("Seed contains missing place IDs.")

        if self.registry["place_id"].isna().any():
            raise ValueError("Registry contains missing place IDs.")

        if self.seed["place_id"].duplicated().any():
            raise ValueError("Duplicate place IDs in seed.")

        if self.registry["place_id"].duplicated().any():
            raise ValueError("Duplicate place IDs in registry.")

        self.catalog = self.seed.set_index("place_id")
        self.sources = self.registry.set_index("place_id")

        for place_id, source in self.sources.iterrows():
            if place_id not in self.catalog.index:
                raise ValueError(
                    f"Registry source has unknown place ID: {place_id}"
                )

            if pd.isna(source["source_name"]):
                raise ValueError("Missing source name.")

            if pd.isna(source["source_url"]):
                raise ValueError("Missing source URL.")

            normalize_source_url(str(source["source_url"]))

    def validate_record(
        self,
        record: AttractionEvidence,
    ) -> None:
        """Check schema, catalog identity, and source provenance."""

        validate_evidence(record)

        if record.place_id not in self.catalog.index:
            raise ValueError(
                f"Unknown attraction ID: {record.place_id}"
            )

        catalog_row = self.catalog.loc[record.place_id]

        if record.city_canonical != catalog_row["city_canonical"]:
            raise ValueError(
                "Evidence city does not match catalog city."
            )

        if record.place_id not in self.sources.index:
            raise ValueError(
                "No registered source for this attraction."
            )

        source = self.sources.loc[record.place_id]

        if record.source_name != source["source_name"]:
            raise ValueError(
                "Evidence source name does not match registry."
            )

        if normalize_source_url(
            record.source_url
        ) != normalize_source_url(source["source_url"]):
            raise ValueError(
                "Evidence URL does not match registered source."
            )

        # Structural validation is not independent fact verification.
        # The registry's candidate state cannot authorize 'verified'.
        if record.status == "verified":
            raise ValueError(
                "Verified status requires a separate evidence "
                "review and approval workflow."
            )

    def validate_batch(
        self,
        records: list[AttractionEvidence],
    ) -> dict:
        """Validate an evidence batch and reject duplicate IDs."""

        evidence_ids = [r.evidence_id for r in records]

        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("Duplicate evidence IDs detected.")

        for record in records:
            self.validate_record(record)

        return {
            "status": "PASS",
            "records": len(records),
            "unique_evidence_ids": len(set(evidence_ids)),
            "catalog_identity_match": True,
            "source_registry_match": True,
            "verified_records": 0,
        }

    def validate_csv(self, evidence_path: str | Path) -> dict:
        """Validate the evidence CSV without modifying it."""

        frame = pd.read_csv(
            evidence_path,
            encoding="utf-8-sig",
            dtype=str,
            keep_default_na=False,
        )

        if frame.columns.tolist() != EVIDENCE_COLUMNS:
            raise ValueError(
                "Evidence CSV columns do not match schema."
            )

        records = [
            AttractionEvidence(
                **row
            )
            for row in frame.to_dict(orient="records")
        ]

        return self.validate_batch(records)


def run_negative_tests(validator: EvidenceValidator) -> dict:
    """Check that deliberately invalid evidence is rejected."""

    sample = AttractionEvidence(
        evidence_id="TEST_001",
        place_id="INPLACE_07148",
        city_canonical="kolkata",
        evidence_type="entrance_fee",
        category="indian_adult_museum",
        value="50",
        unit="INR",
        source_name="Victoria Memorial Hall",
        source_url="https://victoriamemorial-cal.org/visit-us/",
        observed_on="2026-10-01",
        status="candidate",
        notes="In-memory test only.",
    )

    validator.validate_record(sample)

    invalid_cases = {
        "unknown_place_id": replace(
            sample, place_id="INPLACE_99999999"
        ),
        "wrong_city": replace(
            sample, city_canonical="jaipur"
        ),
        "wrong_source": replace(
            sample, source_name="Unregistered Provider"
        ),
        "wrong_source_url": replace(
            sample, source_url="https://example.org/incorrect"
        ),
        "negative_fee": replace(
            sample, value="-10"
        ),
        "invalid_status": replace(
            sample, status="approved_without_review"
        ),
        "unapproved_verified_status": replace(
            sample, status="verified"
        ),
        "future_date": replace(
            sample, observed_on="2099-01-01"
        ),
    }

    results = {}

    for name, record in invalid_cases.items():
        try:
            validator.validate_record(record)
        except ValueError:
            results[name] = "REJECTED"
        else:
            raise AssertionError(
                f"Invalid record incorrectly accepted: {name}"
            )

    try:
        validator.validate_batch([sample, sample])
    except ValueError:
        results["duplicate_evidence_id"] = "REJECTED"
    else:
        raise AssertionError(
            "Duplicate evidence ID was incorrectly accepted."
        )

    return results
