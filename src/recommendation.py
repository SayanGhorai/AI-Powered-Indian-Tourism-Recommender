import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRODUCTION_ROOT = (
    PROJECT_ROOT
    / "production_package"
    / "runtime"
    / "Indian-Tourism-AI-Recommender-Production"
)

RECOMMENDATION_DIR = (
    PRODUCTION_ROOT
    / "recommendation"
)


# ============================================================
# RECOMMENDATION ARTIFACT PATHS
# ============================================================

RECOMMENDATION_FEATURES_PATH = (
    RECOMMENDATION_DIR
    / "place_recommendation_features.parquet"
)

RECOMMENDATION_FEATURE_CONFIG_PATH = (
    RECOMMENDATION_DIR
    / "recommendation_feature_config.json"
)

FINAL_RECOMMENDATION_POLICY_PATH = (
    RECOMMENDATION_DIR
    / "final_recommendation_policy.json"
)

PERSONALIZATION_POLICY_PATH = (
    RECOMMENDATION_DIR
    / "personalization_policy.json"
)


# ============================================================
# POLICY CONSTANTS
# ============================================================

NEAR_TIE_RANGE_FRACTION = 0.05


QUALITY_FEATURE_COLUMNS = [
    "place_id",
    "review_count",
    "rating_mean",
    "bayesian_rating",
    "quality_score",
    "review_support_score",
    "nlp_dominant_sentiment",
    "nlp_sentiment_index",
    "nlp_sentiment_support_tier",
]


SPARSE_METADATA_COLUMNS = [
    "place_id",
    "attraction_type",
    "significance",
    "entrance_fee_inr",
    "visit_time_hours",
    "dslr_allowed",
    "airport_within_50km",
    "best_time_to_visit",
]


# ============================================================
# JSON LOADER
# ============================================================

def _load_json(path):
    """
    Load one JSON artifact.
    """

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# LOAD RECOMMENDATION ARTIFACTS
# ============================================================

def load_recommendation_artifacts():
    """
    Load and validate the saved production recommendation
    artifacts created during the Kaggle phase.
    """

    required_paths = [
        RECOMMENDATION_FEATURES_PATH,
        RECOMMENDATION_FEATURE_CONFIG_PATH,
        FINAL_RECOMMENDATION_POLICY_PATH,
        PERSONALIZATION_POLICY_PATH,
    ]

    missing_paths = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:

        raise FileNotFoundError(
            "Missing recommendation artifacts:\n"
            + "\n".join(missing_paths)
        )


    # --------------------------------------------------------
    # Load saved artifacts
    # --------------------------------------------------------

    recommendation_features = (
        pd.read_parquet(
            RECOMMENDATION_FEATURES_PATH
        )
    )

    feature_config = (
        _load_json(
            RECOMMENDATION_FEATURE_CONFIG_PATH
        )
    )

    final_policy = (
        _load_json(
            FINAL_RECOMMENDATION_POLICY_PATH
        )
    )

    personalization_policy = (
        _load_json(
            PERSONALIZATION_POLICY_PATH
        )
    )


    # --------------------------------------------------------
    # Validate recommendation feature schema
    # --------------------------------------------------------

    required_feature_columns = {
        "place_id",
        "review_count",
        "rating_mean",
        "nlp_sentiment_review_count",
        "nlp_sentiment_index",
        "nlp_dominant_sentiment",
        "nlp_sentiment_support_tier",
        "bayesian_rating",
        "review_support_score",
        "quality_score",
        "sentiment_explanation_score",
    }


    missing_columns = (
        required_feature_columns
        -
        set(
            recommendation_features.columns
        )
    )


    if missing_columns:

        raise RuntimeError(
            "Recommendation feature artifact is "
            "missing columns: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )


    # --------------------------------------------------------
    # Validate place identity
    # --------------------------------------------------------

    if (
        recommendation_features[
            "place_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Recommendation feature artifact "
            "contains duplicate place_id values."
        )


    # --------------------------------------------------------
    # Validate saved catalog size
    # --------------------------------------------------------

    expected_place_count = int(
        feature_config.get(
            "place_count",
            len(
                recommendation_features
            ),
        )
    )


    if (
        len(
            recommendation_features
        )
        !=
        expected_place_count
    ):

        raise RuntimeError(
            "Recommendation feature row count "
            "does not match saved config: "
            f"{len(recommendation_features)} "
            f"!= {expected_place_count}."
        )


    # --------------------------------------------------------
    # The saved quality table was validated as complete.
    # Missing values here would indicate artifact corruption.
    # --------------------------------------------------------

    if (
        recommendation_features[
            list(
                required_feature_columns
            )
        ]
        .isna()
        .any()
        .any()
    ):

        raise RuntimeError(
            "Recommendation feature artifact "
            "contains unexpected missing values."
        )


    # --------------------------------------------------------
    # Validate near-tie policy
    # --------------------------------------------------------

    saved_fraction = float(
        final_policy.get(
            "near_tie_range_fraction",
            NEAR_TIE_RANGE_FRACTION,
        )
    )


    if not np.isclose(
        saved_fraction,
        NEAR_TIE_RANGE_FRACTION,
    ):

        raise RuntimeError(
            "Saved near-tie policy does not "
            "match runtime constant."
        )


    return {
        "recommendation_features":
            recommendation_features,

        "feature_config":
            feature_config,

        "final_policy":
            final_policy,

        "personalization_policy":
            personalization_policy,
    }


# ============================================================
# USER PREFERENCE OBJECT
# ============================================================

def build_user_preferences(
    min_rating=None,
    min_review_count=None,
    preferred_attraction_type=None,
    preferred_significance=None,
    max_entrance_fee_inr=None,
    max_visit_time_hours=None,
    require_dslr_allowed=False,
    require_airport_within_50km=False,
    sparse_constraints_are_hard=False,
):
    """
    Build the preference object used by the production
    recommendation policy.

    Universal constraints:
        - minimum rating
        - minimum review count

    Optional sparse preferences:
        - attraction type
        - significance
        - maximum entrance fee
        - maximum visit duration
        - DSLR allowed
        - airport within 50 km

    Sparse metadata is NOT hard by default because its
    catalog coverage is very limited.
    """

    return {

        # Universal / high-coverage constraints

        "min_rating":
            min_rating,

        "min_review_count":
            min_review_count,


        # Sparse metadata preferences

        "preferred_attraction_type":
            preferred_attraction_type,

        "preferred_significance":
            preferred_significance,

        "max_entrance_fee_inr":
            max_entrance_fee_inr,

        "max_visit_time_hours":
            max_visit_time_hours,

        "require_dslr_allowed":
            bool(
                require_dslr_allowed
            ),

        "require_airport_within_50km":
            bool(
                require_airport_within_50km
            ),


        # False by default because metadata
        # coverage is extremely sparse.

        "sparse_constraints_are_hard":
            bool(
                sparse_constraints_are_hard
            ),
    }


# ============================================================
# TEXT NORMALIZATION FOR PREFERENCE MATCHING
# ============================================================

def normalize_preference_value(
    value
):
    """
    Normalize a structured text metadata value for
    preference matching.
    """

    if pd.isna(value):
        return None


    value = (
        str(value)
        .strip()
        .lower()
    )


    if not value:
        return None


    return value


# ============================================================
# ATTACH QUALITY + STRUCTURED METADATA
# ============================================================

def attach_recommendation_features(
    search_results,
    recommendation_features,
    places_master,
):
    """
    Attach saved catalog-wide quality features and sparse
    tourism metadata to Cross-Encoder-reranked candidates.
    """

    if search_results is None:

        raise ValueError(
            "search_results must not be None."
        )


    if search_results.empty:

        return (
            search_results.copy()
        )


    if (
        "place_id"
        not in search_results.columns
    ):

        raise RuntimeError(
            "Search results are missing place_id."
        )


    # --------------------------------------------------------
    # Validate quality feature columns
    # --------------------------------------------------------

    missing_quality_columns = [

        column

        for column
        in QUALITY_FEATURE_COLUMNS

        if column
        not in recommendation_features.columns
    ]


    if missing_quality_columns:

        raise RuntimeError(
            "Recommendation features are "
            "missing columns: "
            + ", ".join(
                missing_quality_columns
            )
        )


    # --------------------------------------------------------
    # Validate sparse metadata columns
    # --------------------------------------------------------

    missing_metadata_columns = [

        column

        for column
        in SPARSE_METADATA_COLUMNS

        if column
        not in places_master.columns
    ]


    if missing_metadata_columns:

        raise RuntimeError(
            "places_master is missing "
            "recommendation metadata columns: "
            + ", ".join(
                missing_metadata_columns
            )
        )


    result = (
        search_results.copy()
    )


    # --------------------------------------------------------
    # Attach quality features
    # --------------------------------------------------------

    result = result.merge(

        recommendation_features[
            QUALITY_FEATURE_COLUMNS
        ],

        on="place_id",

        how="left",

        validate="many_to_one",
    )


    # All catalog places should have recommendation
    # quality features.

    if (
        result[
            "rating_mean"
        ]
        .isna()
        .any()
    ):

        raise RuntimeError(
            "Some candidates could not be joined "
            "to recommendation features."
        )


    # --------------------------------------------------------
    # Attach sparse structured tourism metadata
    # --------------------------------------------------------

    result = result.merge(

        places_master[
            SPARSE_METADATA_COLUMNS
        ],

        on="place_id",

        how="left",

        validate="many_to_one",
    )


    return result


# ============================================================
# APPLY USER PREFERENCES + HARD CONSTRAINTS
# ============================================================

def apply_user_preferences(
    candidates,
    preferences,
):
    """
    Apply universal hard constraints and optional sparse
    preference evidence.

    IMPORTANT:

    Unknown sparse metadata receives:
        - no match
        - no penalty

    If sparse_constraints_are_hard=True, only KNOWN
    violations are rejected. Unknown values remain eligible.
    """

    if candidates.empty:

        return (
            candidates.copy()
        )


    if preferences is None:

        preferences = (
            build_user_preferences()
        )


    result = (
        candidates.copy()
    )


    # --------------------------------------------------------
    # Initialize preference evidence
    # --------------------------------------------------------

    result[
        "preference_match_count"
    ] = 0


    result[
        "preference_known_count"
    ] = 0


    result[
        "hard_constraint_pass"
    ] = True


    result[
        "preference_reasons"
    ] = [

        []

        for _ in range(
            len(result)
        )
    ]


    # ========================================================
    # UNIVERSAL CONSTRAINT:
    # Minimum rating
    # ========================================================

    min_rating = (
        preferences.get(
            "min_rating"
        )
    )


    if min_rating is not None:

        result[
            "hard_constraint_pass"
        ] &= (

            result[
                "rating_mean"
            ]

            >=

            float(
                min_rating
            )
        )


    # ========================================================
    # UNIVERSAL CONSTRAINT:
    # Minimum review count
    # ========================================================

    min_review_count = (
        preferences.get(
            "min_review_count"
        )
    )


    if min_review_count is not None:

        result[
            "hard_constraint_pass"
        ] &= (

            result[
                "review_count"
            ]

            >=

            int(
                min_review_count
            )
        )


    # ========================================================
    # SPARSE TEXT PREFERENCES
    #
    # Missing metadata:
    #     no match
    #     no penalty
    # ========================================================

    text_preferences = [

        (
            "attraction_type",
            "preferred_attraction_type",
            "attraction type",
        ),

        (
            "significance",
            "preferred_significance",
            "significance",
        ),
    ]


    for (
        field,
        preference_key,
        label,
    ) in text_preferences:


        wanted = (
            preferences.get(
                preference_key
            )
        )


        if wanted is None:
            continue


        wanted = (
            normalize_preference_value(
                wanted
            )
        )


        if wanted is None:
            continue


        known_mask = (
            result[
                field
            ]
            .notna()
        )


        normalized_values = (
            result[
                field
            ]
            .apply(
                normalize_preference_value
            )
        )


        match_mask = (

            known_mask

            &

            normalized_values
            .fillna("")
            .str.contains(
                wanted,
                regex=False,
            )
        )


        result.loc[
            known_mask,
            "preference_known_count",
        ] += 1


        result.loc[
            match_mask,
            "preference_match_count",
        ] += 1


        for idx in (
            result.index[
                match_mask
            ]
        ):

            result.at[
                idx,
                "preference_reasons",
            ] = (

                result.at[
                    idx,
                    "preference_reasons",
                ]

                +

                [
                    f"matches preferred "
                    f"{label}: {wanted}"
                ]
            )


        # ----------------------------------------------------
        # Optional sparse hard constraint
        #
        # Only KNOWN violations fail.
        # Unknown metadata remains eligible.
        # ----------------------------------------------------

        if preferences.get(
            "sparse_constraints_are_hard",
            False,
        ):

            known_violation = (

                known_mask

                &

                ~match_mask
            )


            result.loc[
                known_violation,
                "hard_constraint_pass",
            ] = False


    # ========================================================
    # MAXIMUM ENTRANCE FEE
    # ========================================================

    max_fee = (
        preferences.get(
            "max_entrance_fee_inr"
        )
    )


    if max_fee is not None:

        known_mask = (
            result[
                "entrance_fee_inr"
            ]
            .notna()
        )


        match_mask = (

            known_mask

            &

            (
                result[
                    "entrance_fee_inr"
                ]

                <=

                float(
                    max_fee
                )
            )
        )


        result.loc[
            known_mask,
            "preference_known_count",
        ] += 1


        result.loc[
            match_mask,
            "preference_match_count",
        ] += 1


        for idx in (
            result.index[
                match_mask
            ]
        ):

            result.at[
                idx,
                "preference_reasons",
            ] = (

                result.at[
                    idx,
                    "preference_reasons",
                ]

                +

                [
                    "known entrance fee "
                    f"within ₹{float(max_fee):.0f}"
                ]
            )


        if preferences.get(
            "sparse_constraints_are_hard",
            False,
        ):

            known_violation = (

                known_mask

                &

                ~match_mask
            )


            result.loc[
                known_violation,
                "hard_constraint_pass",
            ] = False


    # ========================================================
    # MAXIMUM VISIT TIME
    # ========================================================

    max_hours = (
        preferences.get(
            "max_visit_time_hours"
        )
    )


    if max_hours is not None:

        known_mask = (
            result[
                "visit_time_hours"
            ]
            .notna()
        )


        match_mask = (

            known_mask

            &

            (
                result[
                    "visit_time_hours"
                ]

                <=

                float(
                    max_hours
                )
            )
        )


        result.loc[
            known_mask,
            "preference_known_count",
        ] += 1


        result.loc[
            match_mask,
            "preference_match_count",
        ] += 1


        for idx in (
            result.index[
                match_mask
            ]
        ):

            result.at[
                idx,
                "preference_reasons",
            ] = (

                result.at[
                    idx,
                    "preference_reasons",
                ]

                +

                [
                    "known visit time "
                    f"within "
                    f"{float(max_hours):g} hours"
                ]
            )


        if preferences.get(
            "sparse_constraints_are_hard",
            False,
        ):

            known_violation = (

                known_mask

                &

                ~match_mask
            )


            result.loc[
                known_violation,
                "hard_constraint_pass",
            ] = False


    # ========================================================
    # BOOLEAN SPARSE PREFERENCES
    # ========================================================

    boolean_preferences = [

        (
            "dslr_allowed",
            "require_dslr_allowed",
            "DSLR allowed",
        ),

        (
            "airport_within_50km",
            "require_airport_within_50km",
            "airport within 50 km",
        ),
    ]


    for (
        field,
        preference_key,
        label,
    ) in boolean_preferences:


        if not preferences.get(
            preference_key,
            False,
        ):

            continue


        known_mask = (
            result[
                field
            ]
            .notna()
        )


        normalized_values = (

            result[
                field
            ]

            .astype(str)

            .str.strip()

            .str.lower()
        )


        match_mask = (

            known_mask

            &

            normalized_values.isin(
                [
                    "yes",
                    "true",
                    "1",
                ]
            )
        )


        result.loc[
            known_mask,
            "preference_known_count",
        ] += 1


        result.loc[
            match_mask,
            "preference_match_count",
        ] += 1


        for idx in (
            result.index[
                match_mask
            ]
        ):

            result.at[
                idx,
                "preference_reasons",
            ] = (

                result.at[
                    idx,
                    "preference_reasons",
                ]

                +

                [
                    label
                ]
            )


        if preferences.get(
            "sparse_constraints_are_hard",
            False,
        ):

            known_violation = (

                known_mask

                &

                ~match_mask
            )


            result.loc[
                known_violation,
                "hard_constraint_pass",
            ] = False


    # ========================================================
    # PREFERENCE MATCH RATIO
    #
    # IMPORTANT:
    # Denominator = KNOWN preference evidence only.
    #
    # Unknown sparse metadata does not lower the ratio.
    # ========================================================

    result[
        "preference_match_ratio"
    ] = np.where(

        result[
            "preference_known_count"
        ] > 0,

        (
            result[
                "preference_match_count"
            ]

            /

            result[
                "preference_known_count"
            ]
        ),

        np.nan,
    )


    return result


# ============================================================
# CROSS-ENCODER NEAR-TIE GROUPS
# ============================================================

def assign_ce_near_tie_groups(
    candidates,
    range_fraction=NEAR_TIE_RANGE_FRACTION,
):
    """
    Create relevance-preserving Cross-Encoder groups.

    The candidates are first ordered by Cross-Encoder score.

    Threshold:

        5% of the candidate-set Cross-Encoder score range.

    Adjacent CE score gaps larger than the threshold begin
    a new relevance tier.

    Candidates may only be preference/quality reordered
    within these relevance-preserving groups.
    """

    if (
        "cross_encoder_score"
        not in candidates.columns
    ):

        raise RuntimeError(
            "Candidates are missing "
            "cross_encoder_score."
        )


    result = (

        candidates

        .sort_values(
            "cross_encoder_score",
            ascending=False,
            kind="stable",
        )

        .reset_index(
            drop=True
        )

        .copy()
    )


    if result.empty:

        result[
            "ce_near_tie_group"
        ] = pd.Series(
            dtype=int
        )

        return (
            result,
            0.0,
        )


    scores = (

        result[
            "cross_encoder_score"
        ]

        .astype(float)

        .to_numpy()
    )


    if not np.isfinite(
        scores
    ).all():

        raise RuntimeError(
            "Cross-Encoder scores contain "
            "non-finite values."
        )


    score_range = float(

        scores.max()

        -

        scores.min()
    )


    threshold = (

        score_range

        *

        float(
            range_fraction
        )
    )


    # --------------------------------------------------------
    # If all scores are identical, all candidates
    # belong to one relevance group.
    # --------------------------------------------------------

    if score_range == 0:

        result[
            "ce_near_tie_group"
        ] = 0

        return (
            result,
            0.0,
        )


    groups = [
        0
    ]


    current_group = 0


    # --------------------------------------------------------
    # Group according to adjacent CE score gaps.
    #
    # A large gap starts a new relevance tier.
    # --------------------------------------------------------

    for i in range(
        1,
        len(scores),
    ):

        gap = (

            scores[
                i - 1
            ]

            -

            scores[
                i
            ]
        )


        if gap > threshold:

            current_group += 1


        groups.append(
            current_group
        )


    result[
        "ce_near_tie_group"
    ] = groups


    return (
        result,
        threshold,
    )


# ============================================================
# FINAL CONSERVATIVE RECOMMENDATION RERANKING
# ============================================================

def conservative_recommendation_rerank(
    candidates,
    preferences=None,
):
    """
    Apply the final conservative recommendation policy.

    Ranking hierarchy:

        1. Cross-Encoder relevance tier
        2. Explicit known preference matches
        3. Cross-Encoder score
        4. Bayesian rating

    Therefore:

        - preference cannot cross a clear relevance boundary
        - quality cannot cross a clear relevance boundary
        - preference evidence can matter within a CE tier
        - Cross-Encoder remains ahead of Bayesian quality
    """

    if preferences is None:

        preferences = (
            build_user_preferences()
        )


    if candidates.empty:

        return (

            candidates.copy(),

            {
                "ce_near_tie_threshold":
                    0.0,

                "near_tie_group_count":
                    0,
            },
        )


    # --------------------------------------------------------
    # Required ranking inputs
    # --------------------------------------------------------

    required_columns = {
        "cross_encoder_score",
        "bayesian_rating",
        "rating_mean",
        "review_count",
    }


    missing_columns = (

        required_columns

        -

        set(
            candidates.columns
        )
    )


    if missing_columns:

        raise RuntimeError(
            "Recommendation candidates are "
            "missing columns: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )


    # --------------------------------------------------------
    # Apply explicit preference and constraint logic first.
    # --------------------------------------------------------

    result = (
        apply_user_preferences(
            candidates,
            preferences,
        )
    )


    # --------------------------------------------------------
    # Hard constraints remove confirmed violations.
    # --------------------------------------------------------

    result = (

        result[
            result[
                "hard_constraint_pass"
            ]
        ]

        .copy()
    )


    if result.empty:

        return (

            result,

            {
                "ce_near_tie_threshold":
                    0.0,

                "near_tie_group_count":
                    0,
            },
        )


    # --------------------------------------------------------
    # Create relevance-preserving CE tiers.
    # --------------------------------------------------------

    (
        result,
        threshold,
    ) = assign_ce_near_tie_groups(
        result
    )


    # --------------------------------------------------------
    # FINAL SORT POLICY
    #
    # 1. CE relevance tier
    # 2. Known explicit preference matches
    # 3. Cross-Encoder score
    # 4. Bayesian quality
    #
    # No global weighted relevance formula.
    # --------------------------------------------------------

    result = (

        result

        .sort_values(

            [
                "ce_near_tie_group",
                "preference_match_count",
                "cross_encoder_score",
                "bayesian_rating",
            ],

            ascending=[
                True,
                False,
                False,
                False,
            ],

            kind="stable",
        )

        .reset_index(
            drop=True
        )
    )


    # --------------------------------------------------------
    # Final recommendation rank
    # --------------------------------------------------------

    result[
        "recommendation_rank"
    ] = np.arange(
        1,
        len(result) + 1,
    )


    # --------------------------------------------------------
    # Basic structured ranking reason
    # --------------------------------------------------------

    result[
        "ranking_reason"
    ] = result.apply(

        lambda row:

        (
            "preference-supported relevant candidate"

            if row[
                "preference_match_count"
            ] > 0

            else

            "cross-encoder relevance"
        ),

        axis=1,
    )


    info = {

        "ce_near_tie_threshold":
            float(
                threshold
            ),

        "near_tie_group_count":
            int(
                result[
                    "ce_near_tie_group"
                ]
                .nunique()
            ),
    }


    return (
        result,
        info,
    )


# ============================================================
# COMPLETE RECOMMENDATION STAGE
# ============================================================

def recommend_candidates(
    cross_encoder_candidates,
    recommendation_features,
    places_master,
    preferences=None,
    top_k=5,
):
    """
    Complete production recommendation stage AFTER the
    Cross-Encoder.

    Input:
        Cross-Encoder-reranked candidate DataFrame.

    Processing:
        Cross-Encoder candidates
            ->
        quality + sentiment evidence
            ->
        sparse structured metadata
            ->
        explicit hard constraints
            ->
        CE near-tie grouping
            ->
        optional preference evidence
            ->
        Bayesian quality tie-break
            ->
        final recommendation Top-K

    This function deliberately does NOT perform retrieval or
    Cross-Encoder inference. Those remain in retrieval.py.
    """

    if preferences is None:

        preferences = (
            build_user_preferences()
        )


    if top_k < 1:

        raise ValueError(
            "top_k must be at least 1."
        )


    # --------------------------------------------------------
    # Attach catalog-wide quality + sparse metadata
    # --------------------------------------------------------

    candidates = (
        attach_recommendation_features(

            cross_encoder_candidates,

            recommendation_features,

            places_master,
        )
    )


    # --------------------------------------------------------
    # Conservative recommendation reranking
    # --------------------------------------------------------

    (
        ranked,
        ranking_info,
    ) = (
        conservative_recommendation_rerank(

            candidates,

            preferences,
        )
    )


    # --------------------------------------------------------
    # Final Top-K
    # --------------------------------------------------------

    final_results = (

        ranked

        .head(
            top_k
        )

        .copy()
    )


    # --------------------------------------------------------
    # Recommendation metadata
    # --------------------------------------------------------

    recommendation_info = {

        **ranking_info,

        "candidate_k":
            len(
                cross_encoder_candidates
            ),

        "eligible_candidates":
            len(
                ranked
            ),

        "returned_results":
            len(
                final_results
            ),

        "preferences":
            preferences,

        "ranking_policy":
            (
                "Cross-Encoder relevance tiers -> "
                "preference evidence -> "
                "Cross-Encoder -> "
                "Bayesian quality"
            ),
    }


    return (
        final_results,
        recommendation_info,
    )