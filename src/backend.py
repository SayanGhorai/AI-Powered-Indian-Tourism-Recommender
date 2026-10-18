
"""Production orchestration for the Indian Tourism Recommender V2.

Uses saved Kaggle inference artifacts only. No fitting, fine-tuning, or
artifact regeneration occurs here. The four stage modules remain independent:
retrieval -> Cross-Encoder -> recommendation -> explainability -> grounded RAG.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

from src.retrieval import (
    build_city_reference,
    build_city_semantic_row_index,
    build_city_tfidf_row_index,
    cross_encoder_rerank,
    detect_query_city,
    load_cross_encoder_model,
    load_retrieval_artifacts,
    load_retrieval_documents,
    load_semantic_model,
    production_retrieval_search,
    reconstruct_semantic_embeddings,
)
from src.recommendation import (
    build_user_preferences,
    load_recommendation_artifacts,
    recommend_candidates,
)
from src.explainability import (
    attach_explainability_geography,
    audit_explanation_metadata,
    explain_recommendations,
)
from src.rag import (
    build_rag_context,
    deterministic_rag_response,
    load_rag_grounding_policy,
)


# The saved Cross-Encoder configuration was validated at Top-10 depth.
MAX_VALIDATED_CANDIDATES = 10


@lru_cache(maxsize=1)
def load_backend() -> dict[str, Any]:
    """Load and cache the verified production artifacts and pretrained models.

    Call once per Python process (or use the automatic cache via recommend_places).
    This may take some time on the first call; subsequent calls reuse everything.
    """
    retrieval = load_retrieval_artifacts()
    recommendation = load_recommendation_artifacts()
    documents = load_retrieval_documents()
    policy = load_rag_grounding_policy()

    places_master = retrieval["places_master"]
    recommendation_features = recommendation["recommendation_features"]

    # A one-to-one place identity is essential for trustworthy joins and RAG.
    for label, frame in (
        ("places_master", places_master),
        ("recommendation_features", recommendation_features),
        ("retrieval_documents", documents),
    ):
        if "place_id" not in frame.columns:
            raise RuntimeError(f"{label} has no place_id column.")
        if frame["place_id"].isna().any() or frame["place_id"].duplicated().any():
            raise RuntimeError(f"{label} has missing or duplicate place IDs.")

    master_ids = set(places_master["place_id"].astype(str))
    for label, frame in (
        ("recommendation_features", recommendation_features),
        ("retrieval_documents", documents),
    ):
        if set(frame["place_id"].astype(str)) != master_ids:
            raise RuntimeError(f"{label} place IDs do not match places_master.")

    (
        city_reference,
        city_surface_to_canonical,
        approved_aliases_added,
    ) = build_city_reference(places_master)

    _, city_to_tfidf_rows = build_city_tfidf_row_index(
        retrieval["tfidf_lookup"], places_master
    )
    semantic_city_lookup, city_to_semantic_rows = build_city_semantic_row_index(
        retrieval["semantic_lookup"], places_master
    )

    # Saved FAISS vectors are reconstructed, not recomputed with SBERT.
    semantic_embeddings = reconstruct_semantic_embeddings(
        retrieval["semantic_index"]
    )

    return {
        "places_master": places_master,
        "recommendation_features": recommendation_features,
        "retrieval_documents": documents,
        "rag_policy": policy,
        "tfidf_vectorizer": retrieval["tfidf_vectorizer"],
        "tfidf_matrix": retrieval["tfidf_matrix"],
        "tfidf_lookup": retrieval["tfidf_lookup"],
        "city_to_tfidf_rows": city_to_tfidf_rows,
        "semantic_index": retrieval["semantic_index"],
        "semantic_lookup": retrieval["semantic_lookup"],
        "semantic_embeddings": semantic_embeddings,
        "semantic_city_lookup": semantic_city_lookup,
        "city_to_semantic_rows": city_to_semantic_rows,
        "city_reference": city_reference,
        "city_surface_to_canonical": city_surface_to_canonical,
        "approved_aliases_added": approved_aliases_added,
        "semantic_model": load_semantic_model(),
        "cross_encoder": load_cross_encoder_model(),
    }


def _validated_preferences(preferences: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep no-preference behavior identical to the verified Kaggle regression."""
    if preferences is None or preferences == {}:
        return None
    if not isinstance(preferences, dict):
        raise TypeError("preferences must be a dictionary or None.")

    allowed = set(build_user_preferences())
    unknown = sorted(set(preferences) - allowed)
    if unknown:
        raise ValueError(f"Unsupported preference fields: {unknown}")

    normalized = build_user_preferences(**preferences)
    for key in ("min_rating", "min_review_count", "max_entrance_fee_inr", "max_visit_time_hours"):
        value = normalized[key]
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be numeric or None.")
        if not (float("-inf") < float(value) < float("inf")):
            raise ValueError(f"{key} must be finite.")
        if float(value) < 0:
            raise ValueError(f"{key} cannot be negative.")
    if normalized["min_rating"] is not None and normalized["min_rating"] > 5:
        raise ValueError("min_rating cannot exceed 5.")
    if normalized["min_review_count"] is not None:
        count = normalized["min_review_count"]
        if int(count) != count:
            raise ValueError("min_review_count must be a whole number.")
    return normalized


def _empty_recommendations() -> pd.DataFrame:
    """Minimum schema required by the RAG context builder when no places match."""
    return pd.DataFrame(columns=["place_id", "recommendation_rank", "place", "city"])


def recommend_places(
    query: str,
    preferences: dict[str, Any] | None = None,
    top_k: int = 5,
    candidate_k: int = 10,
    max_review_chars: int | None = None,
    backend: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full validated pipeline for one user query.

    Returns a dictionary containing:
        recommendations: ranked pandas DataFrame
        explanations: structured evidence-backed explanation list
        rag_context: bounded, verified context for later LLM integration
        answer: deterministic fallback text (works without an LLM/API key)
        info: detected city, cleaned retrieval query, counts and safety audit

    `backend` optionally accepts a previously loaded resource dictionary, useful
    for integration tests. Normal application usage should omit it.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Enter a nonempty travel query.")
    query = query.strip()

    for name, value in (("top_k", top_k), ("candidate_k", candidate_k)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not (top_k <= candidate_k <= MAX_VALIDATED_CANDIDATES):
        raise ValueError(
            f"Require top_k <= candidate_k <= {MAX_VALIDATED_CANDIDATES} "
            "(the saved validated Cross-Encoder depth)."
        )

    prefs = _validated_preferences(preferences)
    resources = load_backend() if backend is None else backend

    saved_limit = int(resources["rag_policy"]["review_excerpt_max_chars"])
    if max_review_chars is None:
        max_review_chars = saved_limit
    if (
        isinstance(max_review_chars, bool)
        or not isinstance(max_review_chars, int)
        or not (1 <= max_review_chars <= saved_limit)
    ):
        raise ValueError(f"max_review_chars must be between 1 and {saved_limit}.")

    # Use the exact detector already used by production_retrieval_search.
    parsed = detect_query_city(query, resources["city_surface_to_canonical"])
    detected_city = parsed["detected_city"]
    retrieval_query = parsed["retrieval_query"]

    hybrid = production_retrieval_search(
        query=query,
        method="hybrid",
        city_surface_to_canonical=resources["city_surface_to_canonical"],
        tfidf_vectorizer=resources["tfidf_vectorizer"],
        tfidf_matrix=resources["tfidf_matrix"],
        tfidf_lookup=resources["tfidf_lookup"],
        city_to_tfidf_rows=resources["city_to_tfidf_rows"],
        semantic_model=resources["semantic_model"],
        semantic_index=resources["semantic_index"],
        semantic_lookup=resources["semantic_lookup"],
        semantic_embeddings=resources["semantic_embeddings"],
        semantic_city_lookup=resources["semantic_city_lookup"],
        city_to_semantic_rows=resources["city_to_semantic_rows"],
        top_k=candidate_k,
    )

    info: dict[str, Any] = {
        "query": query,
        "detected_city": detected_city,
        "matched_city_surface": parsed["matched_surface"],
        "retrieval_query": retrieval_query,
        "hybrid_candidate_count": len(hybrid),
        "cross_encoder_candidate_count": 0,
        "recommendation_count": 0,
        "metadata_audit_passed": True,
        "unsupported_metadata_claims": 0,
    }

    if hybrid.empty:
        results = _empty_recommendations()
        explanations: list[dict[str, Any]] = []
        info["status"] = "no_retrieval_candidates"
    else:
        # The Cross-Encoder receives the ORIGINAL query, as in Kaggle.
        ce = cross_encoder_rerank(
            query=query,
            candidates=hybrid,
            cross_encoder=resources["cross_encoder"],
            retrieval_documents=resources["retrieval_documents"],
            rerank_depth=candidate_k,
        )
        info["cross_encoder_candidate_count"] = len(ce)

        if ce.empty:
            results = _empty_recommendations()
            explanations = []
            info["status"] = "no_cross_encoder_candidates"
        else:
            results, recommendation_info = recommend_candidates(
                cross_encoder_candidates=ce,
                recommendation_features=resources["recommendation_features"],
                places_master=resources["places_master"],
                preferences=prefs,
                top_k=top_k,
            )
            if not isinstance(recommendation_info, dict):
                raise RuntimeError("Recommendation stage did not return an info dictionary.")
            info["recommendation_details"] = recommendation_info

            if results.empty:
                results = _empty_recommendations()
                explanations = []
                info["status"] = "no_candidates_after_preferences"
            else:
                # Strict geographic safety: check CANONICAL city, not display text.
                if detected_city is not None:
                    results = attach_explainability_geography(
                        results, resources["places_master"]
                    )
                    if not results["city_canonical"].eq(detected_city).all():
                        raise RuntimeError(
                            "City safety failure: a recommendation lies outside "
                            f"the requested city {detected_city!r}."
                        )

                explanations = explain_recommendations(
                    results=results,
                    query=query,
                    detected_city=detected_city,
                    places_master=resources["places_master"],
                )
                audit = audit_explanation_metadata(
                    results=results, explanations=explanations
                )
                info["metadata_audit_passed"] = bool(audit["passed"])
                info["unsupported_metadata_claims"] = int(
                    audit["unsupported_metadata_claims"]
                )
                if not info["metadata_audit_passed"]:
                    raise RuntimeError("Explainability metadata safety audit failed.")
                info["status"] = "ok"

    info["recommendation_count"] = len(results)
    context, _, _ = build_rag_context(
        query=query,
        results=results,
        explanations=explanations,
        recommendation_info=info,
        retrieval_documents=resources["retrieval_documents"],
        max_review_chars=max_review_chars,
    )
    answer = deterministic_rag_response(context)

    return {
        "query": query,
        "detected_city": detected_city,
        "retrieval_query": retrieval_query,
        "recommendations": results,
        "explanations": explanations,
        "rag_context": context,
        "answer": answer,
        "info": info,
    }


def backend_status() -> dict[str, Any]:
    """Show the loaded runtime inventory; useful for app diagnostics."""
    resources = load_backend()
    return {
        "catalog_places": len(resources["places_master"]),
        "retrieval_documents": len(resources["retrieval_documents"]),
        "recommendation_features": len(resources["recommendation_features"]),
        "canonical_cities": len(resources["city_reference"]),
        "rag_excerpt_max_chars": resources["rag_policy"]["review_excerpt_max_chars"],
        "models_loaded": True,
    }
