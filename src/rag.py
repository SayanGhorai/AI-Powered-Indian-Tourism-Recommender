
"""Grounded RAG context and deterministic fallback for the tourism recommender.

This module packages *already ranked* recommendations. Retrieval, Cross-Encoder,
recommendation, and explainability run upstream (in backend.py).
No LLM inference or model training happens here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROOT = (
    PROJECT_ROOT
    / "production_package"
    / "runtime"
    / "Indian-Tourism-AI-Recommender-Production"
)
RAG_POLICY_PATH = PRODUCTION_ROOT / "rag" / "rag_grounding_policy.json"

RAG_METADATA_FIELDS = (
    "attraction_type",
    "significance",
    "entrance_fee_inr",
    "visit_time_hours",
    "dslr_allowed",
    "airport_within_50km",
    "best_time_to_visit",
)


def load_rag_grounding_policy() -> dict[str, Any]:
    """Load the actual Kaggle RAG policy; do not silently replace it."""
    if not RAG_POLICY_PATH.is_file():
        raise FileNotFoundError(f"RAG policy not found: {RAG_POLICY_PATH}")

    with RAG_POLICY_PATH.open("r", encoding="utf-8") as file:
        policy = json.load(file)

    required = {
        "rag_role",
        "retrieval_architecture",
        "generation_location",
        "kaggle_llm_training",
        "grounding_rules",
        "default_top_k",
        "default_candidate_k",
        "review_excerpt_max_chars",
    }
    missing = sorted(required.difference(policy))
    if missing:
        raise RuntimeError(f"RAG policy is missing keys: {missing}")

    if policy["review_excerpt_max_chars"] != 900:
        raise RuntimeError("Unexpected saved RAG excerpt limit; inspect the policy.")
    return policy


def build_rag_document_lookup(retrieval_documents: pd.DataFrame) -> pd.DataFrame:
    """Index the saved retrieval documents by verified place_id."""
    if not isinstance(retrieval_documents, pd.DataFrame):
        raise TypeError("retrieval_documents must be a pandas DataFrame.")

    required = {"place_id", "retrieval_document"}
    missing = sorted(required.difference(retrieval_documents.columns))
    if missing:
        raise RuntimeError(f"Retrieval documents are missing columns: {missing}")

    lookup = retrieval_documents[["place_id", "retrieval_document"]].copy()
    if lookup["place_id"].isna().any():
        raise RuntimeError("Retrieval documents contain missing place_id values.")

    lookup["place_id"] = lookup["place_id"].astype(str)
    if lookup["place_id"].duplicated().any():
        raise RuntimeError("Retrieval documents contain duplicate place IDs.")

    return lookup.set_index("place_id", drop=False)


def build_review_evidence_excerpt(
    place_id: str,
    rag_document_lookup: pd.DataFrame,
    max_chars: int = 900,
) -> str | None:
    """Return the saved document's bounded prefix, preserving Kaggle truncation."""
    if not isinstance(max_chars, int) or max_chars < 1:
        raise ValueError("max_chars must be a positive integer.")

    place_id = str(place_id)
    if place_id not in rag_document_lookup.index:
        return None

    document = rag_document_lookup.at[place_id, "retrieval_document"]
    if pd.isna(document):
        return None

    document = str(document).strip()
    if not document:
        return None
    if len(document) <= max_chars:
        return document

    excerpt = document[:max_chars]
    if " " in excerpt:
        excerpt = excerpt.rsplit(" ", 1)[0]
    return excerpt.strip() + "..."


def build_rag_place_record(
    result_row: pd.Series,
    explanation_record: dict[str, Any],
    rag_document_lookup: pd.DataFrame,
    max_review_chars: int = 900,
) -> dict[str, Any]:
    """Construct a grounded record from an existing recommendation/explanation."""
    place_id = str(result_row["place_id"])
    if str(explanation_record["place_id"]) != place_id:
        raise RuntimeError(f"Explanation place_id mismatch for {place_id}.")

    evidence = explanation_record["evidence"]
    if str(evidence["place_id"]) != place_id:
        raise RuntimeError(f"Explanation evidence mismatch for {place_id}.")

    quality = evidence["quality"]
    sentiment = evidence["sentiment"]
    metadata = evidence["metadata"]
    preferences = evidence["preferences"]

    return {
        "place_id": place_id,
        "rank": int(result_row["recommendation_rank"]),
        "place": str(result_row["place"]),
        "city": str(result_row["city"]),
        "rating": {
            "mean": quality.get("rating_mean"),
            "review_count": quality.get("review_count"),
            "confidence_adjusted": quality.get("bayesian_rating"),
        },
        "sentiment": {
            "dominant": sentiment.get("dominant"),
            "index": sentiment.get("sentiment_index"),
            "support_tier": sentiment.get("support_tier"),
        },
        "tourism_metadata": {
            field: metadata.get(field) for field in RAG_METADATA_FIELDS
        },
        "preference_evidence": {
            "matched_count": preferences.get("matched_count", 0),
            "known_count": preferences.get("known_count", 0),
            "reasons": list(preferences.get("reasons") or []),
        },
        "explanation": explanation_record["explanation"],
        "review_evidence_excerpt": build_review_evidence_excerpt(
            place_id, rag_document_lookup, max_chars=max_review_chars
        ),
    }


def build_rag_context(
    query: str,
    results: pd.DataFrame,
    explanations: list[dict[str, Any]],
    recommendation_info: dict[str, Any],
    retrieval_documents: pd.DataFrame,
    max_review_chars: int = 900,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Package the *already computed* local pipeline output for RAG.

    Local adaptation of Kaggle Cell 18: the future backend passes results,
    explanations and metadata instead of calling Kaggle's global
    explainable_recommend(). Return shape matches the Kaggle function.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a nonempty string.")
    if not isinstance(results, pd.DataFrame):
        raise TypeError("results must be a pandas DataFrame.")
    if not isinstance(recommendation_info, dict):
        raise TypeError("recommendation_info must be a dictionary.")
    if not isinstance(explanations, list):
        raise TypeError("explanations must be a list.")
    if not isinstance(max_review_chars, int) or max_review_chars < 1:
        raise ValueError("max_review_chars must be a positive integer.")

    required = {"place_id", "recommendation_rank", "place", "city"}
    missing = sorted(required.difference(results.columns))
    if missing:
        raise RuntimeError(f"Recommendations are missing columns: {missing}")
    if results["place_id"].isna().any() or results["place_id"].astype(str).duplicated().any():
        raise RuntimeError("Recommendations contain missing or duplicate place IDs.")

    explanation_lookup: dict[str, dict[str, Any]] = {}
    for item in explanations:
        place_id = str(item["place_id"])
        if place_id in explanation_lookup:
            raise RuntimeError(f"Duplicate explanation for {place_id}.")
        explanation_lookup[place_id] = item

    result_ids = set(results["place_id"].astype(str))
    if result_ids != set(explanation_lookup):
        raise RuntimeError("Recommendation/explanation place IDs do not match.")

    rag_document_lookup = build_rag_document_lookup(retrieval_documents)
    detected_city = recommendation_info.get("detected_city")
    if detected_city is not None:
        for item in explanations:
            geo = item["evidence"].get("geographic_constraint")
            if (
                not isinstance(geo, dict)
                or geo.get("requested_city") != detected_city
                or geo.get("constraint_satisfied") is not True
            ):
                raise RuntimeError(
                    f"City constraint not verified for {item['place_id']}."
                )

    places = [
        build_rag_place_record(
            row,
            explanation_lookup[str(row["place_id"])],
            rag_document_lookup,
            max_review_chars=max_review_chars,
        )
        for _, row in results.iterrows()
    ]

    context = {
        "query": query,
        "detected_city": detected_city,
        "retrieval_query": recommendation_info.get("retrieval_query"),
        "result_count": len(places),
        "places": places,
        "grounding_policy": {
            "use_only_provided_places": True,
            "do_not_invent_place_facts": True,
            "do_not_infer_missing_metadata": True,
            "review_text_is_supporting_evidence": True,
            "rating_and_sentiment_are_correlated": True,
        },
    }
    return context, results, recommendation_info


def deterministic_rag_response(context: dict[str, Any]) -> str:
    """Return a grounded user-facing answer without any external LLM."""
    places = context.get("places", [])
    if not places:
        return "I could not find a sufficiently supported recommendation for that request."

    city = context.get("detected_city")
    if city:
        opening = f"Here are the strongest supported recommendations I found for {city.title()}:"
    else:
        opening = "Here are the strongest supported recommendations I found:"

    lines = [opening]
    for place in places:
        rating = place["rating"]["mean"]
        reviews = place["rating"]["review_count"]
        line = f"{place['rank']}. {place['place']} ({place['city']})"
        if rating is not None and reviews is not None:
            line += f" — {rating:.2f}/5 from {reviews:,} reviews"
        lines.append(line)

    lines.append(
        "These recommendations are based on retrieval relevance and verified "
        "project data; missing metadata has not been inferred."
    )
    return "\n".join(lines)
