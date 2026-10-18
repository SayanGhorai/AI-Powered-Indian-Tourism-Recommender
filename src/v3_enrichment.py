from pathlib import Path
import pandas as pd

from src.retrieval import PLACES_MASTER_PATH


REQUIRED_COLUMNS = [
    "place_id",
    "city_canonical",
    "catalog_place_name",
    "legacy_attraction_type",
    "legacy_entrance_fee_inr",
    "legacy_visit_time_hours",
    "legacy_weekly_off",
    "verified_attraction_latitude",
    "verified_attraction_longitude",
    "verified_entrance_fee_inr",
    "verified_visit_time_hours",
    "verified_opening_hours",
    "verified_weekly_off",
    "source_name",
    "source_url",
    "verified_at",
    "verification_notes",
    "record_status",
]

VERIFIED_FIELDS = [
    "verified_attraction_latitude",
    "verified_attraction_longitude",
    "verified_entrance_fee_inr",
    "verified_visit_time_hours",
    "verified_opening_hours",
    "verified_weekly_off",
]


def validate_enrichment_seed(
    seed_path: str | Path,
) -> dict:
    """Validate the initial V3 seed against the V2 master catalog."""

    seed_path = Path(seed_path)

    if not seed_path.is_file():
        raise FileNotFoundError(seed_path)

    seed = pd.read_csv(seed_path, encoding="utf-8-sig")
    master = pd.read_parquet(PLACES_MASTER_PATH)

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in seed.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing seed columns: {missing_columns}"
        )

    if seed.empty:
        raise ValueError("Enrichment seed is empty.")

    if seed["place_id"].isna().any():
        raise ValueError("Seed contains missing place IDs.")

    if seed["place_id"].duplicated().any():
        raise ValueError("Seed contains duplicate place IDs.")

    if master["place_id"].duplicated().any():
        raise ValueError("Master contains duplicate place IDs.")

    comparison = seed.merge(
        master[["place_id", "city_canonical", "place"]],
        on="place_id",
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("_seed", "_master"),
    )

    if not comparison["_merge"].eq("both").all():
        raise ValueError(
            "Some seed place IDs are absent from the master."
        )

    city_matches = (
        comparison["city_canonical_seed"]
        == comparison["city_canonical_master"]
    )

    if not city_matches.all():
        raise ValueError(
            "Seed city names do not match master identities."
        )

    place_matches = (
        comparison["catalog_place_name"]
        == comparison["place"]
    )

    if not place_matches.all():
        raise ValueError(
            "Seed attraction names do not match master identities."
        )

    # The initial seed contains reference data only.
    # No new fields may be labeled verified at this stage.
    if not seed["record_status"].eq("legacy_unverified").all():
        raise ValueError(
            "Unexpected verification status in initial seed."
        )

    for column in VERIFIED_FIELDS:
        if seed[column].notna().any():
            raise ValueError(
                f"Unverified seed has populated field: {column}"
            )

    for column in ["source_name", "source_url", "verified_at"]:
        if seed[column].notna().any():
            raise ValueError(
                f"Unexpected source data in initial seed: {column}"
            )

    return {
        "status": "PASS",
        "rows": len(seed),
        "unique_place_ids": seed["place_id"].nunique(),
        "cities": seed["city_canonical"].value_counts().to_dict(),
        "verified_records": 0,
        "master_identity_match": True,
    }
