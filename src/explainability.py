from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRODUCTION_ROOT = (
    PROJECT_ROOT
    / "production_package"
    / "runtime"
    / "Indian-Tourism-AI-Recommender-Production"
)

EXPLAINABILITY_DIR = (
    PRODUCTION_ROOT
    / "explainability"
)

EXPLAINABILITY_POLICY_PATH = (
    EXPLAINABILITY_DIR
    / "explainability_policy.json"
)


METADATA_FIELDS = [
    "attraction_type",
    "significance",
    "entrance_fee_inr",
    "visit_time_hours",
    "dslr_allowed",
    "airport_within_50km",
    "best_time_to_visit",
]


# ============================================================
# 2. POLICY LOADING
# ============================================================

def _load_json(
    path: Path,
) -> dict[str, Any]:

    if not path.exists():

        raise FileNotFoundError(
            "Required explainability artifact "
            f"not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def load_explainability_policy():

    policy = _load_json(
        EXPLAINABILITY_POLICY_PATH
    )

    required_keys = {
        "architecture",
        "allowed_evidence",
        "missing_metadata_policy",
        "sentiment_policy",
        "llm_policy",
        "streamlit_policy",
    }

    missing = sorted(
        required_keys.difference(
            policy
        )
    )

    if missing:

        raise RuntimeError(
            "Explainability policy is missing "
            "required keys: "
            + ", ".join(missing)
        )

    return policy


# ============================================================
# 3. SAFE VALUE HELPERS
# ============================================================

def safe_value(
    value,
):

    if value is None:
        return None

    try:

        if pd.isna(value):
            return None

    except Exception:
        pass

    return value


def safe_float(
    value,
):

    value = safe_value(
        value
    )

    if value is None:
        return None

    try:

        return float(
            value
        )

    except Exception:

        return None


def safe_int(
    value,
):

    value = safe_value(
        value
    )

    if value is None:
        return None

    try:

        return int(
            value
        )

    except Exception:

        return None


def clean_text_value(
    value,
):

    value = safe_value(
        value
    )

    if value is None:
        return None

    value = str(
        value
    ).strip()

    return (
        value
        if value
        else None
    )


# ============================================================
# 4. ATTACH CANONICAL GEOGRAPHY
#
# Kaggle explainability verifies city safety using
# city_canonical.
#
# The local retrieval result carries the display city but not
# city_canonical, so recover it from the packaged master table
# using place_id.
#
# No city is inferred from text.
# ============================================================

def attach_explainability_geography(
    results: pd.DataFrame,
    places_master: pd.DataFrame,
):

    if results is None:

        raise ValueError(
            "results cannot be None."
        )

    if results.empty:

        return results.copy()

    if "place_id" not in results.columns:

        raise RuntimeError(
            "Explainability results are missing "
            "required column: place_id"
        )

    required_master_columns = {
        "place_id",
        "city_canonical",
    }

    missing_master_columns = (
        required_master_columns
        .difference(
            places_master.columns
        )
    )

    if missing_master_columns:

        raise RuntimeError(
            "places_master is missing required "
            "explainability columns: "
            + ", ".join(
                sorted(
                    missing_master_columns
                )
            )
        )

    result = results.copy()

    # If already supplied by an upstream production layer,
    # do not merge it again.
    if (
        "city_canonical"
        in result.columns
    ):

        return result

    city_lookup = (
        places_master[
            [
                "place_id",
                "city_canonical",
            ]
        ]
        .copy()
    )

    if (
        city_lookup[
            "place_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "places_master contains duplicate "
            "place_id values."
        )

    result = result.merge(
        city_lookup,
        on="place_id",
        how="left",
        validate="many_to_one",
    )

    missing_city = (
        result[
            "city_canonical"
        ]
        .isna()
    )

    if missing_city.any():

        missing_ids = (
            result.loc[
                missing_city,
                "place_id",
            ]
            .astype(str)
            .tolist()
        )

        raise RuntimeError(
            "Canonical city missing for "
            "recommendation place IDs: "
            + ", ".join(
                missing_ids[:10]
            )
        )

    return result


# ============================================================
# 5. BUILD STRUCTURED EXPLANATION
# ============================================================

def build_recommendation_explanation(
    row,
    query,
    detected_city=None,
):

    evidence = {

        "place_id":
            str(
                row["place_id"]
            ),

        "place":
            clean_text_value(
                row.get(
                    "place"
                )
            ),

        "city":
            clean_text_value(
                row.get(
                    "city"
                )
            ),

        "query":
            str(
                query
            ),

        "recommendation_rank":
            safe_int(
                row.get(
                    "recommendation_rank"
                )
            ),

        # ----------------------------------------------------
        # Retrieval evidence
        # ----------------------------------------------------

        "retrieval": {

            "cross_encoder_score":
                safe_float(
                    row.get(
                        "cross_encoder_score"
                    )
                ),

            "hybrid_rank":
                safe_int(
                    row.get(
                        "hybrid_rank"
                    )
                ),

            "ce_near_tie_group":
                safe_int(
                    row.get(
                        "ce_near_tie_group"
                    )
                ),

            "ranking_reason":
                clean_text_value(
                    row.get(
                        "ranking_reason"
                    )
                ),
        },

        # ----------------------------------------------------
        # Quality evidence
        # ----------------------------------------------------

        "quality": {

            "rating_mean":
                safe_float(
                    row.get(
                        "rating_mean"
                    )
                ),

            "review_count":
                safe_int(
                    row.get(
                        "review_count"
                    )
                ),

            "bayesian_rating":
                safe_float(
                    row.get(
                        "bayesian_rating"
                    )
                ),

            "quality_score":
                safe_float(
                    row.get(
                        "quality_score"
                    )
                ),

            "review_support_score":
                safe_float(
                    row.get(
                        "review_support_score"
                    )
                ),
        },

        # ----------------------------------------------------
        # Sentiment evidence
        #
        # Descriptive/supporting evidence only.
        # It is not treated as independent rating quality.
        # ----------------------------------------------------

        "sentiment": {

            "dominant":
                clean_text_value(
                    row.get(
                        "nlp_dominant_sentiment"
                    )
                ),

            "sentiment_index":
                safe_float(
                    row.get(
                        "nlp_sentiment_index"
                    )
                ),

            "support_tier":
                clean_text_value(
                    row.get(
                        "nlp_sentiment_support_tier"
                    )
                ),
        },

        # ----------------------------------------------------
        # Preference evidence
        # ----------------------------------------------------

        "preferences": {

            "matched_count":
                (
                    safe_int(
                        row.get(
                            "preference_match_count"
                        )
                    )
                    or 0
                ),

            "known_count":
                (
                    safe_int(
                        row.get(
                            "preference_known_count"
                        )
                    )
                    or 0
                ),

            "match_ratio":
                safe_float(
                    row.get(
                        "preference_match_ratio"
                    )
                ),

            "reasons":
                (
                    row.get(
                        "preference_reasons"
                    )

                    if isinstance(
                        row.get(
                            "preference_reasons"
                        ),
                        list,
                    )

                    else []
                ),
        },

        # ----------------------------------------------------
        # Structured tourism metadata
        #
        # Missing values remain None.
        # Nothing is inferred.
        # ----------------------------------------------------

        "metadata": {

            "attraction_type":
                clean_text_value(
                    row.get(
                        "attraction_type"
                    )
                ),

            "significance":
                clean_text_value(
                    row.get(
                        "significance"
                    )
                ),

            "entrance_fee_inr":
                safe_float(
                    row.get(
                        "entrance_fee_inr"
                    )
                ),

            "visit_time_hours":
                safe_float(
                    row.get(
                        "visit_time_hours"
                    )
                ),

            "dslr_allowed":
                clean_text_value(
                    row.get(
                        "dslr_allowed"
                    )
                ),

            "airport_within_50km":
                clean_text_value(
                    row.get(
                        "airport_within_50km"
                    )
                ),

            "best_time_to_visit":
                clean_text_value(
                    row.get(
                        "best_time_to_visit"
                    )
                ),
        },
    }

    # --------------------------------------------------------
    # Geographic evidence
    # --------------------------------------------------------

    if detected_city is not None:

        result_city = (
            clean_text_value(
                row.get(
                    "city_canonical"
                )
            )
        )

        evidence[
            "geographic_constraint"
        ] = {

            "requested_city":
                detected_city,

            "result_city":
                result_city,

            "constraint_satisfied":
                (
                    result_city
                    ==
                    detected_city
                ),
        }

    else:

        evidence[
            "geographic_constraint"
        ] = None

    return evidence


# ============================================================
# 6. CREATE HUMAN-READABLE REASONS
# ============================================================

def explanation_to_reasons(
    evidence,
):

    reasons = []

    # --------------------------------------------------------
    # A. Retrieval relevance
    # --------------------------------------------------------

    retrieval = (
        evidence[
            "retrieval"
        ]
    )

    if (
        retrieval.get(
            "cross_encoder_score"
        )
        is not None
    ):

        reasons.append(
            "Ranked as a strong semantic match "
            "for the travel query by the "
            "retrieval and reranking pipeline."
        )

    # --------------------------------------------------------
    # B. Geographic correctness
    # --------------------------------------------------------

    geo = (
        evidence.get(
            "geographic_constraint"
        )
    )

    if (
        geo is not None
        and
        geo.get(
            "constraint_satisfied"
        )
    ):

        reasons.append(
            "Matches the requested city constraint: "
            f"{geo['requested_city']}."
        )

    # --------------------------------------------------------
    # C. Rating + review support
    # --------------------------------------------------------

    quality = (
        evidence[
            "quality"
        ]
    )

    rating = (
        quality.get(
            "rating_mean"
        )
    )

    review_count = (
        quality.get(
            "review_count"
        )
    )

    bayesian = (
        quality.get(
            "bayesian_rating"
        )
    )

    if (
        rating is not None
        and
        review_count is not None
    ):

        reasons.append(
            f"Has an average rating of "
            f"{rating:.2f}/5 from "
            f"{review_count:,} reviews."
        )

    if bayesian is not None:

        reasons.append(
            f"Its confidence-adjusted rating is "
            f"{bayesian:.2f}/5, which accounts "
            f"for review support."
        )

    # --------------------------------------------------------
    # D. Sentiment
    #
    # Descriptive only.
    # Do not claim it is independent from rating.
    # --------------------------------------------------------

    sentiment = (
        evidence[
            "sentiment"
        ]
    )

    dominant_sentiment = (
        sentiment.get(
            "dominant"
        )
    )

    sentiment_support = (
        sentiment.get(
            "support_tier"
        )
    )

    if (
        dominant_sentiment
        is not None
    ):

        sentiment_text = (
            "Aggregated review-text sentiment "
            "is predominantly "
            f"{dominant_sentiment}"
        )

        if (
            sentiment_support
            is not None
        ):

            sentiment_text += (
                f" with "
                f"{sentiment_support} "
                f"review support"
            )

        sentiment_text += "."

        reasons.append(
            sentiment_text
        )

    # --------------------------------------------------------
    # E. Explicit preference matches
    # --------------------------------------------------------

    preference_reasons = (
        evidence[
            "preferences"
        ]
        .get(
            "reasons",
            [],
        )
    )

    for preference_reason in (
        preference_reasons
    ):

        reasons.append(
            "Matches your explicit preference: "
            f"{preference_reason}."
        )

    # --------------------------------------------------------
    # F. Structured metadata
    #
    # Only state fields that are actually known.
    # --------------------------------------------------------

    metadata = (
        evidence[
            "metadata"
        ]
    )

    attraction_type = (
        metadata.get(
            "attraction_type"
        )
    )

    if (
        attraction_type
        is not None
    ):

        reasons.append(
            "Available metadata classifies it as "
            f"{attraction_type}."
        )

    significance = (
        metadata.get(
            "significance"
        )
    )

    if (
        significance
        is not None
    ):

        reasons.append(
            "Recorded significance: "
            f"{significance}."
        )

    entrance_fee = (
        metadata.get(
            "entrance_fee_inr"
        )
    )

    if (
        entrance_fee
        is not None
    ):

        reasons.append(
            "Recorded entrance fee: "
            f"₹{entrance_fee:,.0f}."
        )

    visit_time = (
        metadata.get(
            "visit_time_hours"
        )
    )

    if (
        visit_time
        is not None
    ):

        reasons.append(
            "Recorded visit duration: "
            f"about {visit_time:g} hours."
        )

    best_time = (
        metadata.get(
            "best_time_to_visit"
        )
    )

    if (
        best_time
        is not None
    ):

        reasons.append(
            "Dataset-listed best visit time: "
            f"{best_time}."
        )

    return reasons


# ============================================================
# 7. SHORT USER-FACING EXPLANATION
# ============================================================

def build_short_explanation(
    evidence,
    max_reasons=4,
):

    reasons = (
        explanation_to_reasons(
            evidence
        )
    )

    return (
        reasons[
            :max_reasons
        ]
    )


# ============================================================
# 8. EXPLAIN COMPLETE RESULT SET
# ============================================================

def explain_recommendations(
    results: pd.DataFrame,
    query,
    detected_city=None,
    places_master: pd.DataFrame | None = None,
):

    if results is None:

        raise ValueError(
            "results cannot be None."
        )

    prepared = (
        results.copy()
    )

    # --------------------------------------------------------
    # Local production adaptation:
    #
    # Kaggle result rows carried city_canonical.
    # The current local recommendation DataFrame does not.
    #
    # Recover the exact saved canonical city from places_master
    # before performing geographic evidence validation.
    # --------------------------------------------------------

    if (
        detected_city is not None
        and
        "city_canonical"
        not in prepared.columns
    ):

        if places_master is None:

            raise RuntimeError(
                "City-constrained explainability requires "
                "places_master so city_canonical can be "
                "verified from saved project data."
            )

        prepared = (
            attach_explainability_geography(
                results=prepared,
                places_master=places_master,
            )
        )

    explained_rows = []

    for _, row in (
        prepared.iterrows()
    ):

        evidence = (
            build_recommendation_explanation(
                row=row,
                query=query,
                detected_city=detected_city,
            )
        )

        reasons = (
            build_short_explanation(
                evidence
            )
        )

        explained_rows.append(
            {

                "place_id":
                    evidence[
                        "place_id"
                    ],

                "recommendation_rank":
                    evidence[
                        "recommendation_rank"
                    ],

                "place":
                    evidence[
                        "place"
                    ],

                "city":
                    evidence[
                        "city"
                    ],

                "explanation":
                    " ".join(
                        reasons
                    ),

                "evidence":
                    evidence,
            }
        )

    return explained_rows


# ============================================================
# 9. MISSING-METADATA SAFETY AUDIT
#
# Every metadata claim surfaced by explainability must have
# existed in the corresponding recommendation result row.
# ============================================================

def audit_explanation_metadata(
    results: pd.DataFrame,
    explanations,
):

    if (
        "place_id"
        not in results.columns
    ):

        raise RuntimeError(
            "results are missing place_id "
            "for metadata audit."
        )

    source = (
        results.copy()
    )

    source[
        "place_id"
    ] = (
        source[
            "place_id"
        ]
        .astype(str)
    )

    if (
        source[
            "place_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "results contain duplicate "
            "place_id values."
        )

    source = (
        source
        .set_index(
            "place_id",
            drop=False,
        )
    )

    unsupported_claims = []

    for explanation in (
        explanations
    ):

        evidence = (
            explanation[
                "evidence"
            ]
        )

        place_id = str(
            evidence[
                "place_id"
            ]
        )

        if (
            place_id
            not in source.index
        ):

            raise RuntimeError(
                "Explanation place_id "
                "not found in results: "
                f"{place_id}"
            )

        source_row = (
            source.loc[
                place_id
            ]
        )

        metadata = (
            evidence[
                "metadata"
            ]
        )

        for field in (
            METADATA_FIELDS
        ):

            surfaced_value = (
                metadata.get(
                    field
                )
            )

            if (
                surfaced_value
                is None
            ):

                continue

            source_value = (
                safe_value(
                    source_row.get(
                        field
                    )
                )
            )

            if (
                source_value
                is None
            ):

                unsupported_claims.append(
                    {
                        "place_id":
                            place_id,

                        "field":
                            field,
                    }
                )

    return {

        "unsupported_metadata_claims":
            len(
                unsupported_claims
            ),

        "details":
            unsupported_claims,

        "passed":
            (
                len(
                    unsupported_claims
                )
                == 0
            ),
    }