from pathlib import Path
import re
import unicodedata

import faiss
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PRODUCTION_ROOT = (
    PROJECT_ROOT
    / "production_package"
    / "runtime"
    / "Indian-Tourism-AI-Recommender-Production"
)

RETRIEVAL_DIR = PRODUCTION_ROOT / "retrieval"
DATA_DIR = PRODUCTION_ROOT / "data"

PLACES_MASTER_PATH = (
    DATA_DIR / "places_master_stage5.parquet"
)

# ============================================================
# RETRIEVAL ARTIFACT PATHS
# ============================================================
RETRIEVAL_DOCUMENTS_PATH = (
    RETRIEVAL_DIR
    / "place_retrieval_documents.parquet"
)

TFIDF_VECTORIZER_PATH = (
    RETRIEVAL_DIR / "tfidf_retrieval_vectorizer.joblib"
)

TFIDF_MATRIX_PATH = (
    RETRIEVAL_DIR / "tfidf_retrieval_matrix.npz"
)

TFIDF_LOOKUP_PATH = (
    RETRIEVAL_DIR / "tfidf_place_lookup.parquet"
)

SEMANTIC_INDEX_PATH = (
    RETRIEVAL_DIR / "semantic_chunks.faiss"
)

SEMANTIC_LOOKUP_PATH = (
    RETRIEVAL_DIR / "semantic_chunk_lookup.parquet"
)


# ============================================================
# SEMANTIC MODEL
# ============================================================

SEMANTIC_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)
CROSS_ENCODER_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
# ============================================================
# ARTIFACT LOADER
# ============================================================

def load_retrieval_artifacts():
    """Load the saved production retrieval artifacts."""

    tfidf_vectorizer = joblib.load(
        TFIDF_VECTORIZER_PATH
    )

    tfidf_matrix = load_npz(
        TFIDF_MATRIX_PATH
    )

    tfidf_lookup = pd.read_parquet(
        TFIDF_LOOKUP_PATH
    )

    semantic_index = faiss.read_index(
        str(SEMANTIC_INDEX_PATH)
    )

    semantic_lookup = pd.read_parquet(
        SEMANTIC_LOOKUP_PATH
    )
    places_master = pd.read_parquet(
        PLACES_MASTER_PATH
    )

    return {
        "tfidf_vectorizer": tfidf_vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "tfidf_lookup": tfidf_lookup,
        "semantic_index": semantic_index,
        "semantic_lookup": semantic_lookup,
        "places_master": places_master,
    }

def load_retrieval_documents():
    """
    Load the saved place-level retrieval documents used
    by the Cross-Encoder reranker.
    """

    documents = pd.read_parquet(
        RETRIEVAL_DOCUMENTS_PATH
    )

    required_columns = {
        "place_id",
        "retrieval_document",
    }

    missing_columns = (
        required_columns
        - set(documents.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Retrieval documents are missing "
            f"required columns: "
            f"{sorted(missing_columns)}"
        )

    if documents["place_id"].duplicated().any():
        raise RuntimeError(
            "Retrieval documents contain "
            "duplicate place_id values."
        )

    if documents[
        "retrieval_document"
    ].isna().any():
        raise RuntimeError(
            "Retrieval documents contain "
            "missing document text."
        )

    empty_documents = (
        documents["retrieval_document"]
        .astype(str)
        .str.strip()
        .eq("")
    )

    if empty_documents.any():
        raise RuntimeError(
            "Retrieval documents contain "
            "empty document text."
        )

    return documents
# ============================================================
# LEXICAL PLACE SEARCH
# ============================================================

def lexical_place_search(
    query,
    tfidf_vectorizer,
    tfidf_matrix,
    tfidf_lookup,
    top_k=10,
):
    """Retrieve places using the saved TF-IDF representation."""

    query = str(query).strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be >= 1.")

    query_vector = tfidf_vectorizer.transform(
        [query]
    )

    if query_vector.nnz == 0:
        return pd.DataFrame(
            columns=[
                "rank",
                "place_id",
                "place",
                "city",
                "lexical_score",
            ]
        )

    scores = (
        tfidf_matrix
        @ query_vector.T
    ).toarray().ravel()

    top_k = min(
        top_k,
        len(tfidf_lookup),
    )

    order = np.argsort(
        scores
    )[::-1][:top_k]

    result = (
        tfidf_lookup
        .iloc[order]
        .copy()
        .reset_index(drop=True)
    )

    result["lexical_score"] = (
        scores[order]
        .astype(np.float32)
    )

    result.insert(
        0,
        "rank",
        np.arange(
            1,
            len(result) + 1,
        ),
    )

    return result
# ============================================================
# SEMANTIC MODEL LOADER
# ============================================================

def load_semantic_model():
    """Load the production SBERT semantic encoder."""

    model = SentenceTransformer(
        SEMANTIC_MODEL_NAME
    )

    return model
# ============================================================
# SEMANTIC PLACE SEARCH
# ============================================================

def semantic_place_search(
    query,
    semantic_model,
    semantic_index,
    semantic_lookup,
    top_k=10,
    chunk_candidate_k=200,
):
    """Retrieve places using SBERT + saved FAISS chunk index."""

    query = str(query).strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be >= 1.")

    if chunk_candidate_k < top_k:
        chunk_candidate_k = top_k

    chunk_candidate_k = min(
        chunk_candidate_k,
        semantic_index.ntotal,
    )

    query_embedding = semantic_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    scores, chunk_ids = semantic_index.search(
        query_embedding,
        chunk_candidate_k,
    )

    valid_mask = chunk_ids[0] >= 0

    chunk_ids = chunk_ids[0][valid_mask]
    scores = scores[0][valid_mask]

    chunk_result = (
        semantic_lookup
        .iloc[chunk_ids]
        .copy()
        .reset_index(drop=True)
    )

    chunk_result["semantic_score"] = (
        scores.astype(np.float32)
    )

    # Collapse multiple semantic chunks belonging to the
    # same place. Keep the highest-scoring chunk per place.
    best_indices = (
        chunk_result
        .groupby(
            "place_id",
            sort=False,
        )["semantic_score"]
        .idxmax()
    )

    place_result = (
        chunk_result
        .loc[best_indices]
        .copy()
        .sort_values(
            [
                "semantic_score",
                "place_id",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .head(top_k)
        .reset_index(drop=True)
    )

    place_result.insert(
        0,
        "rank",
        np.arange(
            1,
            len(place_result) + 1,
        ),
    )

    return place_result

# ============================================================
# CROSS-ENCODER MODEL LOADER
# ============================================================

def load_cross_encoder_model():
    """
    Load the pretrained production Cross-Encoder reranker.
    """

    return CrossEncoder(
        CROSS_ENCODER_MODEL_NAME,
        max_length=512,
    )
# ============================================================
# HYBRID RRF SEARCH
# ============================================================

def hybrid_place_search(
    query,
    semantic_model,
    semantic_index,
    semantic_lookup,
    tfidf_vectorizer,
    tfidf_matrix,
    tfidf_lookup,
    top_k=10,
    branch_candidate_k=50,
    rrf_k=60,
):
    """
    Combine semantic and lexical place rankings using
    unweighted Reciprocal Rank Fusion (RRF).

    Raw semantic and lexical scores are retained for
    diagnostics but are not directly combined.
    """

    query = str(query).strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be >= 1.")

    if branch_candidate_k < top_k:
        branch_candidate_k = top_k

    # --------------------------------------------------------
    # SEMANTIC BRANCH
    # --------------------------------------------------------

    semantic_df = semantic_place_search(
        query=query,
        semantic_model=semantic_model,
        semantic_index=semantic_index,
        semantic_lookup=semantic_lookup,
        top_k=branch_candidate_k,
        chunk_candidate_k=max(
            200,
            branch_candidate_k * 10,
        ),
    ).copy()

    # --------------------------------------------------------
    # LEXICAL BRANCH
    # --------------------------------------------------------

    lexical_df = lexical_place_search(
        query=query,
        tfidf_vectorizer=tfidf_vectorizer,
        tfidf_matrix=tfidf_matrix,
        tfidf_lookup=tfidf_lookup,
        top_k=branch_candidate_k,
    ).copy()

    # --------------------------------------------------------
    # SEMANTIC RANK TABLE
    # --------------------------------------------------------

    semantic_rank_table = (
        semantic_df[
            [
                "place_id",
                "place",
                "city",
                "rank",
                "semantic_score",
            ]
        ]
        .rename(
            columns={
                "rank": "semantic_rank",
            }
        )
    )

    # --------------------------------------------------------
    # LEXICAL RANK TABLE
    # --------------------------------------------------------

    lexical_rank_table = (
        lexical_df[
            [
                "place_id",
                "place",
                "city",
                "rank",
                "lexical_score",
            ]
        ]
        .rename(
            columns={
                "rank": "lexical_rank",
            }
        )
    )

    # --------------------------------------------------------
    # UNION
    # --------------------------------------------------------

    hybrid = semantic_rank_table.merge(
        lexical_rank_table,
        on="place_id",
        how="outer",
        suffixes=(
            "_semantic",
            "_lexical",
        ),
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # RECONCILE DISPLAY IDENTITY
    # --------------------------------------------------------

    hybrid["place"] = (
        hybrid["place_semantic"]
        .combine_first(
            hybrid["place_lexical"]
        )
    )

    hybrid["city"] = (
        hybrid["city_semantic"]
        .combine_first(
            hybrid["city_lexical"]
        )
    )

    hybrid = hybrid.drop(
        columns=[
            "place_semantic",
            "place_lexical",
            "city_semantic",
            "city_lexical",
        ]
    )

    # --------------------------------------------------------
    # SOURCE FLAGS
    # --------------------------------------------------------

    hybrid["in_semantic"] = (
        hybrid["semantic_rank"].notna()
    )

    hybrid["in_lexical"] = (
        hybrid["lexical_rank"].notna()
    )

    # --------------------------------------------------------
    # RRF CONTRIBUTIONS
    # --------------------------------------------------------

    hybrid["semantic_rrf"] = np.where(
        hybrid["semantic_rank"].notna(),
        1.0
        / (
            rrf_k
            + hybrid["semantic_rank"].fillna(0)
        ),
        0.0,
    )

    hybrid["lexical_rrf"] = np.where(
        hybrid["lexical_rank"].notna(),
        1.0
        / (
            rrf_k
            + hybrid["lexical_rank"].fillna(0)
        ),
        0.0,
    )

    hybrid["hybrid_rrf_score"] = (
        hybrid["semantic_rrf"]
        + hybrid["lexical_rrf"]
    )

    # --------------------------------------------------------
    # FINAL HYBRID RANKING
    # --------------------------------------------------------

    hybrid = (
        hybrid
        .sort_values(
            [
                "hybrid_rrf_score",
                "place_id",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .head(top_k)
        .reset_index(drop=True)
    )

    hybrid.insert(
        0,
        "rank",
        np.arange(
            1,
            len(hybrid) + 1,
        ),
    )

    return hybrid
# ============================================================
# CITY TEXT NORMALIZATION
# ============================================================

def normalize_city_text(value):
    """Normalize city text using the Kaggle production policy."""

    value = (
        ""
        if pd.isna(value)
        else str(value)
    )

    value = unicodedata.normalize(
        "NFKD",
        value,
    )

    value = (
        value
        .encode(
            "ascii",
            "ignore",
        )
        .decode(
            "ascii"
        )
    )

    value = value.lower()

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value

# ============================================================
# CONTROLLED CITY ALIASES
# ============================================================

CITY_ALIAS_MAP = {
    # Karnataka
    "bangalore": "bengaluru",
    "mysore": "mysuru mysore",
    "shivamogga": "shimoga",
    "halebidu": "halebid",
    "murudeshwar": "murdeshwar",

    # Delhi / NCR
    "delhi": "new delhi",
    "gurugram": "gurugram gurgaon",

    # Kerala
    "kochi": "kochi cochin",
    "trichur": "thrissur",
    "thiruvananthapuram": "trivandrum",

    # Puducherry
    "puducherry": "pondicherry",

    # Andhra Pradesh
    "visakhapatnam": "visakhapatnam vizag",

    # Goa
    "panaji": "panjim",

    # Odisha
    "bhubaneshwar": "bhubaneswar",
    "kendujhar": "keonjhar",
    "raurkela": "rourkela",

    # Gujarat
    "ahmadabad": "ahmedabad",
    "ghandinagar": "gandhinagar",

    # Maharashtra
    "ahmadnagar": "ahmednagar",

    # Uttarakhand
    "dehra dun": "dehradun",

    # West Bengal
    "shiliguri": "siliguri",
    "sundarbans": "sundarban",

    # Uttar Pradesh
    "shahjanpur": "shahjahanpur",

    # Tamil Nadu
    "rajapalaiyam": "rajapalayam",

    # Common destination aliases
    "cherrapunji": "sohra",
    "jammu": "jammu city",
    "varkala": "varkala town",
    "kaziranga": "kaziranga national park",
    "rann of kutch": "kutch",
}
# ============================================================
# MASTER CITY VOCABULARY
# ============================================================

def build_city_reference(places_master):
    """Build normalized city surfaces from the place master."""

    city_reference = (
        places_master[
            [
                "city",
                "city_canonical",
            ]
        ]
        .dropna(
            subset=[
                "city",
                "city_canonical",
            ]
        )
        .drop_duplicates()
        .copy()
    )

    city_reference["city_normalized"] = (
        city_reference["city"]
        .map(normalize_city_text)
    )

    city_reference["canonical_normalized"] = (
        city_reference["city_canonical"]
        .map(normalize_city_text)
    )

    city_surface_to_canonical = {}

    for row in city_reference.itertuples(
        index=False
    ):
        if row.city_normalized:
            city_surface_to_canonical[
                row.city_normalized
            ] = row.city_canonical

        if row.canonical_normalized:
            city_surface_to_canonical[
                row.canonical_normalized
            ] = row.city_canonical

        # --------------------------------------------------------
    # Add approved aliases
    # --------------------------------------------------------

    approved_aliases_added = 0

    for alias, canonical_surface in (
        CITY_ALIAS_MAP.items()
    ):
        alias_norm = normalize_city_text(
            alias
        )

        canonical_norm = normalize_city_text(
            canonical_surface
        )

        # First try resolving directly through the
        # master city surface vocabulary.
        target = (
            city_surface_to_canonical.get(
                canonical_norm
            )
        )

        # Fallback: match against normalized
        # canonical-city values directly.
        if target is None:
            possible = city_reference[
                city_reference[
                    "canonical_normalized"
                ]
                == canonical_norm
            ]

            if len(possible):
                target = possible.iloc[0][
                    "city_canonical"
                ]

        # Only add an alias when its target
        # exists in the actual master vocabulary.
        if target is not None:
            city_surface_to_canonical[
                alias_norm
            ] = target

            approved_aliases_added += 1

    return (
        city_reference,
        city_surface_to_canonical,
        approved_aliases_added,
    )
# ============================================================
# CITY DETECTOR
# ============================================================

def detect_query_city(
    query,
    city_surface_to_canonical,
):
    """
    Detect an explicit known city using boundary-safe,
    longest-match-first matching.

    No fuzzy city matching is used.
    """

    original_query = str(query).strip()

    normalized_query = normalize_city_text(
        original_query
    )

    city_surfaces = sorted(
        city_surface_to_canonical.keys(),
        key=lambda x: (
            len(x.split()),
            len(x),
        ),
        reverse=True,
    )

    padded_query = (
        " "
        + normalized_query
        + " "
    )

    for surface in city_surfaces:
        pattern = (
            " "
            + surface
            + " "
        )

        if pattern in padded_query:
            canonical_city = (
                city_surface_to_canonical[
                    surface
                ]
            )

            cleaned = re.sub(
                r"\b"
                + re.escape(surface)
                + r"\b",
                " ",
                normalized_query,
            )

            # Remove a location connector left at
            # the end after removing the city.
            cleaned = re.sub(
                r"\b(in|near|at|around)\b\s*$",
                " ",
                cleaned,
            )

            cleaned = re.sub(
                r"\s+",
                " ",
                cleaned,
            ).strip()

            # Never send an empty query to retrieval.
            if not cleaned:
                cleaned = normalized_query

            return {
                "detected_city": canonical_city,
                "matched_surface": surface,
                "retrieval_query": cleaned,
            }

    return {
        "detected_city": None,
        "matched_surface": None,
        "retrieval_query": normalized_query,
    }
# ============================================================
# CITY TF-IDF ROW INDEX
# ============================================================

def build_city_tfidf_row_index(
    tfidf_lookup,
    places_master,
):
    """
    Build canonical-city -> TF-IDF row IDs mapping.

    place_id is used as the stable join key.
    """

    city_lookup = (
        tfidf_lookup[
            [
                "place_id",
                "tfidf_row_id",
            ]
        ]
        .merge(
            places_master[
                [
                    "place_id",
                    "city_canonical",
                ]
            ],
            on="place_id",
            how="left",
            validate="one_to_one",
        )
    )

    if city_lookup[
        "city_canonical"
    ].isna().any():
        raise RuntimeError(
            "Some TF-IDF places have no canonical city."
        )

    city_to_tfidf_rows = {
        city: group[
            "tfidf_row_id"
        ].to_numpy(
            dtype=np.int32
        )
        for city, group in city_lookup.groupby(
            "city_canonical",
            sort=False,
        )
    }

    return (
        city_lookup,
        city_to_tfidf_rows,
    )
# ============================================================
# CITY-FIRST LEXICAL PLACE SEARCH
# ============================================================

def city_first_lexical_place_search(
    query,
    canonical_city,
    tfidf_vectorizer,
    tfidf_matrix,
    tfidf_lookup,
    city_to_tfidf_rows,
    top_k=10,
):
    """
    Run TF-IDF retrieval only inside the requested
    canonical city's candidate universe.
    """

    query = str(query).strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be >= 1.")

    city_rows = city_to_tfidf_rows.get(
        canonical_city
    )

    if city_rows is None or len(city_rows) == 0:
        return pd.DataFrame(
            columns=[
                "rank",
                "place_id",
                "place",
                "city",
                "lexical_score",
            ]
        )

    query_vector = tfidf_vectorizer.transform(
        [query]
    )

    if query_vector.nnz == 0:
        return pd.DataFrame(
            columns=[
                "rank",
                "place_id",
                "place",
                "city",
                "lexical_score",
            ]
        )

    city_matrix = tfidf_matrix[
        city_rows
    ]

    scores = (
        city_matrix
        @ query_vector.T
    ).toarray().ravel()

    top_k = min(
        top_k,
        len(city_rows),
    )

    # Deterministic ranking:
    # score descending, place_id ascending.
    candidate_lookup = (
        tfidf_lookup
        .iloc[city_rows]
        .copy()
        .reset_index(drop=True)
    )

    candidate_lookup[
        "lexical_score"
    ] = scores.astype(
        np.float32
    )

    result = (
        candidate_lookup
        .sort_values(
            [
                "lexical_score",
                "place_id",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .head(top_k)
        .reset_index(drop=True)
    )

    result.insert(
        0,
        "rank",
        np.arange(
            1,
            len(result) + 1,
        ),
    )

    return result
# ============================================================
# RECONSTRUCT SEMANTIC EMBEDDINGS FROM FAISS
# ============================================================

def reconstruct_semantic_embeddings(
    semantic_index,
):
    """
    Reconstruct the normalized semantic chunk embeddings
    stored inside the FAISS IndexFlatIP index.

    This reuses the saved Kaggle vectors and does not
    re-encode the semantic chunks with SBERT.
    """

    if semantic_index.ntotal <= 0:
        raise RuntimeError(
            "FAISS semantic index is empty."
        )

    embeddings = semantic_index.reconstruct_n(
        0,
        semantic_index.ntotal,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    expected_shape = (
        semantic_index.ntotal,
        semantic_index.d,
    )

    if embeddings.shape != expected_shape:
        raise RuntimeError(
            "Reconstructed semantic embedding "
            f"shape mismatch: {embeddings.shape} "
            f"!= {expected_shape}"
        )

    if not np.isfinite(embeddings).all():
        raise RuntimeError(
            "Reconstructed semantic embeddings "
            "contain non-finite values."
        )

    return embeddings
# ============================================================
# CITY SEMANTIC ROW INDEX
# ============================================================

def build_city_semantic_row_index(
    semantic_lookup,
    places_master,
):
    """
    Build canonical-city -> semantic chunk row IDs mapping.

    Each semantic chunk inherits its canonical city
    from its place_id.
    """

    place_city_lookup = (
        places_master[
            [
                "place_id",
                "city_canonical",
            ]
        ]
        .drop_duplicates(
            subset=["place_id"]
        )
    )

    city_lookup = (
        semantic_lookup
        .merge(
            place_city_lookup,
            on="place_id",
            how="left",
            validate="many_to_one",
        )
    )

    if city_lookup[
        "city_canonical"
    ].isna().any():
        raise RuntimeError(
            "Some semantic chunks have no "
            "canonical city."
        )

    city_to_semantic_rows = {
        city: group.index.to_numpy(
            dtype=np.int32
        )
        for city, group in city_lookup.groupby(
            "city_canonical",
            sort=False,
        )
    }

    return (
        city_lookup,
        city_to_semantic_rows,
    )
# ============================================================
# CITY-FIRST SEMANTIC PLACE SEARCH
# ============================================================

def city_first_semantic_place_search(
    query,
    canonical_city,
    semantic_model,
    semantic_embeddings,
    semantic_city_lookup,
    city_to_semantic_rows,
    top_k=10,
):
    """
    Run semantic retrieval only inside the requested
    canonical city's semantic chunk universe.

    Multiple chunks belonging to the same place are
    collapsed using that place's maximum semantic score.
    """

    query = str(query).strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be >= 1.")

    chunk_rows = city_to_semantic_rows.get(
        canonical_city
    )

    if chunk_rows is None or len(chunk_rows) == 0:
        return pd.DataFrame(
            columns=[
                "rank",
                "place_id",
                "place",
                "city",
                "semantic_score",
            ]
        )

    # Encode only the user query.
    # Saved chunk embeddings are reused from FAISS.
    query_embedding = semantic_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype(
        np.float32,
        copy=False,
    )

    city_embeddings = semantic_embeddings[
        chunk_rows
    ]

    scores = (
        city_embeddings
        @ query_embedding
    )

    chunk_result = (
        semantic_city_lookup
        .iloc[chunk_rows]
        .copy()
    )

    chunk_result[
        "semantic_score"
    ] = scores.astype(
        np.float32
    )

    # One result per place:
    # retain the highest-scoring semantic chunk.
    best_indices = (
        chunk_result
        .groupby(
            "place_id",
            sort=False,
        )["semantic_score"]
        .idxmax()
    )

    place_result = (
        chunk_result
        .loc[best_indices]
        .copy()
        .sort_values(
            [
                "semantic_score",
                "place_id",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .head(top_k)
        .reset_index(drop=True)
    )

    place_result.insert(
        0,
        "rank",
        np.arange(
            1,
            len(place_result) + 1,
        ),
    )

    return place_result
# ============================================================
# CITY-FIRST HYBRID PLACE SEARCH
# ============================================================

def city_first_hybrid_place_search(
    query,
    canonical_city,
    tfidf_vectorizer,
    tfidf_matrix,
    tfidf_lookup,
    city_to_tfidf_rows,
    semantic_model,
    semantic_embeddings,
    semantic_city_lookup,
    city_to_semantic_rows,
    top_k=10,
    branch_k=50,
    rrf_k=60,
):
    """
    Run strict city-first hybrid retrieval.

    Both lexical and semantic branches are restricted
    to the requested canonical city before ranking.

    Branch rankings are fused with Reciprocal Rank Fusion.
    """

    semantic_result = (
        city_first_semantic_place_search(
            query=query,
            canonical_city=canonical_city,
            semantic_model=semantic_model,
            semantic_embeddings=semantic_embeddings,
            semantic_city_lookup=semantic_city_lookup,
            city_to_semantic_rows=city_to_semantic_rows,
            top_k=branch_k,
        )
    )

    lexical_result = (
        city_first_lexical_place_search(
            query=query,
            canonical_city=canonical_city,
            tfidf_vectorizer=tfidf_vectorizer,
            tfidf_matrix=tfidf_matrix,
            tfidf_lookup=tfidf_lookup,
            city_to_tfidf_rows=city_to_tfidf_rows,
            top_k=branch_k,
        )
    )

    semantic_branch = (
        semantic_result[
            [
                "place_id",
                "place",
                "city",
                "rank",
                "semantic_score",
            ]
        ]
        .rename(
            columns={
                "rank": "semantic_rank",
            }
        )
    )

    lexical_branch = (
        lexical_result[
            [
                "place_id",
                "place",
                "city",
                "rank",
                "lexical_score",
            ]
        ]
        .rename(
            columns={
                "rank": "lexical_rank",
            }
        )
    )

    merged = semantic_branch.merge(
        lexical_branch,
        on="place_id",
        how="outer",
        suffixes=(
            "_semantic",
            "_lexical",
        ),
        validate="one_to_one",
    )

    merged["place"] = (
        merged["place_semantic"]
        .combine_first(
            merged["place_lexical"]
        )
    )

    merged["city"] = (
        merged["city_semantic"]
        .combine_first(
            merged["city_lexical"]
        )
    )

    merged["semantic_rrf"] = np.where(
        merged["semantic_rank"].notna(),
        1.0 / (
            rrf_k
            + merged["semantic_rank"]
        ),
        0.0,
    )

    merged["lexical_rrf"] = np.where(
        merged["lexical_rank"].notna(),
        1.0 / (
            rrf_k
            + merged["lexical_rank"]
        ),
        0.0,
    )

    merged["hybrid_rrf_score"] = (
        merged["semantic_rrf"]
        + merged["lexical_rrf"]
    )

    result = (
        merged
        .sort_values(
            [
                "hybrid_rrf_score",
                "place_id",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .head(top_k)
        .reset_index(drop=True)
    )

    result.insert(
        0,
        "rank",
        np.arange(
            1,
            len(result) + 1,
        ),
    )

    return result
# ============================================================
# PRODUCTION RETRIEVAL WRAPPER
# ============================================================

def production_retrieval_search(
    query,
    method,
    city_surface_to_canonical,
    tfidf_vectorizer,
    tfidf_matrix,
    tfidf_lookup,
    city_to_tfidf_rows,
    semantic_model,
    semantic_index,
    semantic_lookup,
    semantic_embeddings,
    semantic_city_lookup,
    city_to_semantic_rows,
    top_k=10,
):
    """
    Production retrieval entry point.

    No detected city:
        use global retrieval.

    Detected city:
        enforce strict city-first retrieval.

    The city surface is removed from the retrieval query
    before lexical/semantic/hybrid retrieval.
    """

    parsed = detect_query_city(
        query,
        city_surface_to_canonical,
    )

    city = parsed["detected_city"]
    retrieval_query = parsed[
        "retrieval_query"
    ]

    # --------------------------------------------------------
    # GLOBAL RETRIEVAL — no city detected
    # --------------------------------------------------------

    if city is None:

        if method == "semantic":
            return semantic_place_search(
                retrieval_query,
                semantic_model,
                semantic_index,
                semantic_lookup,
                top_k=top_k,
                chunk_candidate_k=max(
                    200,
                    top_k * 20,
                ),
            )

        if method == "lexical":
            return lexical_place_search(
                retrieval_query,
                tfidf_vectorizer,
                tfidf_matrix,
                tfidf_lookup,
                top_k=top_k,
            )

        if method == "hybrid":
            return hybrid_place_search(
                retrieval_query,
                semantic_model,
                semantic_index,
                semantic_lookup,
                tfidf_vectorizer,
                tfidf_matrix,
                tfidf_lookup,
                top_k=top_k,
                branch_candidate_k=50,
                rrf_k=60,
            )

        raise ValueError(
            f"Unknown retrieval method: {method}"
        )

    # --------------------------------------------------------
    # STRICT CITY-FIRST RETRIEVAL
    # --------------------------------------------------------

    if method == "semantic":
        return city_first_semantic_place_search(
            query=retrieval_query,
            canonical_city=city,
            semantic_model=semantic_model,
            semantic_embeddings=semantic_embeddings,
            semantic_city_lookup=semantic_city_lookup,
            city_to_semantic_rows=city_to_semantic_rows,
            top_k=top_k,
        )

    if method == "lexical":
        return city_first_lexical_place_search(
            query=retrieval_query,
            canonical_city=city,
            tfidf_vectorizer=tfidf_vectorizer,
            tfidf_matrix=tfidf_matrix,
            tfidf_lookup=tfidf_lookup,
            city_to_tfidf_rows=city_to_tfidf_rows,
            top_k=top_k,
        )

    if method == "hybrid":
        return city_first_hybrid_place_search(
            query=retrieval_query,
            canonical_city=city,
            tfidf_vectorizer=tfidf_vectorizer,
            tfidf_matrix=tfidf_matrix,
            tfidf_lookup=tfidf_lookup,
            city_to_tfidf_rows=city_to_tfidf_rows,
            semantic_model=semantic_model,
            semantic_embeddings=semantic_embeddings,
            semantic_city_lookup=semantic_city_lookup,
            city_to_semantic_rows=city_to_semantic_rows,
            top_k=top_k,
            branch_k=50,
            rrf_k=60,
        )

    raise ValueError(
        f"Unknown retrieval method: {method}"
    )
# ============================================================
# CROSS-ENCODER RERANKING
# ============================================================

def cross_encoder_rerank(
    query,
    candidates,
    cross_encoder,
    retrieval_documents,
    rerank_depth=10,
):
    """
    Rerank Hybrid retrieval candidates with the production
    Cross-Encoder using the original natural-language query
    and the saved place-level retrieval documents.
    """

    query = str(query).strip()

    if not query:
        raise ValueError(
            "Cross-Encoder query must not be empty."
        )

    if candidates is None or len(candidates) == 0:
        raise ValueError(
            "Cross-Encoder candidates must not be empty."
        )

    if "place_id" not in candidates.columns:
        raise RuntimeError(
            "Candidates are missing place_id."
        )

    if "rank" not in candidates.columns:
        raise RuntimeError(
            "Candidates are missing Hybrid rank."
        )

    rerank = (
        candidates
        .head(rerank_depth)
        .copy()
        .reset_index(drop=True)
    )

    # Preserve the original Hybrid ranking explicitly.
    rerank["hybrid_rank"] = (
        pd.to_numeric(
            rerank["rank"],
            errors="raise",
        )
        .astype(int)
    )

    document_table = retrieval_documents[
        [
            "place_id",
            "retrieval_document",
        ]
    ].copy()

    rerank = rerank.merge(
        document_table,
        on="place_id",
        how="left",
        validate="many_to_one",
    )

    missing_documents = (
        rerank["retrieval_document"]
        .isna()
    )

    empty_documents = (
        rerank["retrieval_document"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )

    if missing_documents.any():
        raise RuntimeError(
            "Some Cross-Encoder candidates could not "
            "be joined to retrieval documents."
        )

    if empty_documents.any():
        raise RuntimeError(
            "Some Cross-Encoder candidates have "
            "empty retrieval documents."
        )

    pairs = [
        (
            query,
            str(document),
        )
        for document in rerank[
            "retrieval_document"
        ].tolist()
    ]

    scores = cross_encoder.predict(
        pairs,
        batch_size=32,
        show_progress_bar=False,
    )

    rerank["cross_encoder_score"] = (
        np.asarray(
            scores,
            dtype=np.float32,
        )
    )

    rerank["cross_encoder_rank"] = (
        rerank["cross_encoder_score"]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    rerank = (
        rerank
        .sort_values(
            "cross_encoder_rank"
        )
        .reset_index(drop=True)
    )

    return rerank