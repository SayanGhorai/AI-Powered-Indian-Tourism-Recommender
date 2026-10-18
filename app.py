
"""Streamlit interface for AI-Powered Indian Tourism Recommender V2.

Run from the project root: streamlit run app.py
Inference uses the verified Kaggle production bundle through src.backend.
"""

from __future__ import annotations

import json
import math
from typing import Any

import streamlit as st

from src.backend import recommend_places


st.set_page_config(
    page_title="Indian Tourism AI | Discover Places",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
    .hero-kicker {font-size: .79rem; font-weight: 700; letter-spacing: .15em;
                  text-transform: uppercase; color: #1b927e; margin-bottom: .5rem;}
    .hero-title {font-size: clamp(2rem, 4vw, 3.1rem); font-weight: 780;
                 line-height: 1.12; letter-spacing: -.045em; margin: 0 0 .65rem;}
    .hero-subtitle {font-size: 1.02rem; opacity: .8; max-width: 760px;}
    .small-note {font-size: .85rem; opacity: .75;}
    div[data-testid="stMetric"] {padding: .35rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


def is_known(value: Any) -> bool:
    """True only for meaningful scalar values, not missing metadata."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip()) and value.strip().lower() not in {
            "nan", "none", "null", "<na>"
        }
    try:
        return not bool(math.isnan(float(value)))
    except (TypeError, ValueError, OverflowError):
        return True


def format_number(value: Any, decimals: int = 0) -> str:
    if not is_known(value):
        return "Not available"
    try:
        number = float(value)
        return f"{number:,.{decimals}f}"
    except (TypeError, ValueError, OverflowError):
        return str(value)


def metadata_display(metadata: dict[str, Any]) -> list[tuple[str, str]]:
    """Display ONLY fields supplied by the grounded context."""
    fields: list[tuple[str, str]] = []
    text_fields = {
        "attraction_type": "Attraction type",
        "significance": "Significance",
        "best_time_to_visit": "Recorded best time to visit",
    }
    for key, label in text_fields.items():
        value = metadata.get(key)
        if is_known(value):
            fields.append((label, str(value)))

    fee = metadata.get("entrance_fee_inr")
    if is_known(fee):
        fields.append(("Recorded entrance fee", f"₹{format_number(fee)}"))

    duration = metadata.get("visit_time_hours")
    if is_known(duration):
        fields.append(("Recorded visit duration", f"{format_number(duration, 1)} hours"))

    for key, label in (
        ("dslr_allowed", "DSLR photography allowed"),
        ("airport_within_50km", "Airport within 50 km"),
    ):
        value = metadata.get(key)
        if isinstance(value, bool):
            fields.append((label, "Yes" if value else "No"))
    return fields


def render_place(place: dict[str, Any]) -> None:
    """Render a recommendation using its verified RAG context only."""
    with st.container(border=True):
        left, right = st.columns([3, 1], vertical_alignment="top")
        with left:
            st.subheader(f"{place.get('rank', '—')}. {place.get('place', 'Unknown place')}")
            st.caption(f"📍 {place.get('city') or 'City not available'}")
        with right:
            rating = place.get("rating") or {}
            mean = rating.get("mean")
            count = rating.get("review_count")
            st.metric("Average rating", f"{format_number(mean, 2)}/5" if is_known(mean) else "N/A")
            st.caption(f"{format_number(count)} reviews" if is_known(count) else "Review count unavailable")

        explanation = place.get("explanation")
        if isinstance(explanation, str) and explanation.strip():
            st.markdown(f"**Why recommended:** {explanation}")
        elif isinstance(explanation, (list, tuple)):
            for reason in explanation[:4]:
                if reason:
                    st.write(f"• {reason}")

        preferences = place.get("preference_evidence") or {}
        reasons = preferences.get("reasons") or []
        if reasons:
            st.caption("Preference evidence: " + "; ".join(map(str, reasons)))

        metadata = metadata_display(place.get("tourism_metadata") or {})
        excerpt = place.get("review_evidence_excerpt")
        if metadata or is_known(excerpt):
            with st.expander("Review evidence and available tourism metadata"):
                if metadata:
                    st.markdown("**Recorded metadata**")
                    for label, value in metadata:
                        st.write(f"**{label}:** {value}")
                else:
                    st.caption("No additional structured tourism metadata is available for this place.")
                if is_known(excerpt):
                    st.markdown("**Supporting review-derived text**")
                    st.write(str(excerpt))
                    st.caption("Supporting text is not an independently verified fact about the attraction.")


with st.sidebar:
    st.title("🧭 Tourism AI")
    st.caption("Indian Tourism Recommender · V2")
    st.divider()
    st.markdown("**How recommendations are produced**")
    st.caption("City-aware TF-IDF + SBERT/FAISS → Reciprocal Rank Fusion → Cross-Encoder → preferences and quality tie-breaks → grounded explanations.")
    st.divider()
    st.markdown("**Saved production catalog**")
    st.write("15,239 places · 1,794 canonical cities")
    st.caption("Models are pretrained. Saved Kaggle artifacts are reused; no training is performed in this app.")
    st.divider()
    st.info("Tourism metadata is sparse. Unknown fees, visit times, or attraction categories are not inferred.")

st.markdown('<div class="hero-kicker">Explore India with evidence</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Find your next remarkable place.</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Describe what you want to experience. '
    'The recommender ranks places using review evidence, semantic relevance, '
    'and optional preferences.</div>',
    unsafe_allow_html=True,
)
st.write("")

with st.form("tourism_search_form"):
    query = st.text_input(
        "What kind of place would you like to visit?",
        value="historical places in Jaipur",
        placeholder="e.g., peaceful places in Kolkata or quiet beach for relaxation",
        help="Include a city in your query to activate strict city-first retrieval.",
    )

    settings_left, settings_right = st.columns(2)
    with settings_left:
        top_k = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)
    with settings_right:
        min_rating = st.slider(
            "Minimum average rating (optional)",
            min_value=0.0, max_value=5.0, value=0.0, step=0.5,
            help="Zero disables the rating constraint.",
        )

    with st.expander("Advanced preferences (optional)"):
        st.caption("The minimum rating and review count are hard filters. Sparse metadata is neutral when missing unless you explicitly enable hard constraints.")
        min_reviews = st.number_input(
            "Minimum review count", min_value=0, value=0, step=25
        )
        col_a, col_b = st.columns(2)
        with col_a:
            attraction_type = st.text_input(
                "Preferred attraction type", placeholder="e.g., Fort"
            )
            use_max_fee = st.checkbox("Set maximum recorded entrance fee")
            max_fee = st.number_input(
                "Maximum entrance fee (₹)", min_value=0.0,
                value=500.0, step=50.0,
                help="Used only when the checkbox above is selected.",
            )
            require_dslr = st.checkbox("Prefer DSLR photography allowed")
        with col_b:
            significance = st.text_input(
                "Preferred significance", placeholder="Optional metadata term"
            )
            use_max_duration = st.checkbox("Set maximum recorded visit duration")
            max_duration = st.number_input(
                "Maximum visit duration (hours)", min_value=0.0,
                value=3.0, step=0.5,
                help="Used only when the checkbox above is selected.",
            )
            require_airport = st.checkbox("Prefer airport within 50 km")
        sparse_hard = st.checkbox(
            "Treat known sparse-metadata mismatches as hard exclusions",
            value=False,
            help="Missing metadata remains neutral; known mismatches may be excluded.",
        )

    submitted = st.form_submit_button(
        "Find recommendations", type="primary", use_container_width=True
    )

st.caption("Try: historical places in Jaipur · quiet beach for relaxation · peaceful places in Kolkata")

if submitted:
    if not query.strip():
        st.warning("Please enter a travel query before searching.")
    else:
        preferences: dict[str, Any] = {}
        if min_rating > 0:
            preferences["min_rating"] = min_rating
        if min_reviews > 0:
            preferences["min_review_count"] = int(min_reviews)
        if attraction_type.strip():
            preferences["preferred_attraction_type"] = attraction_type.strip()
        if significance.strip():
            preferences["preferred_significance"] = significance.strip()
        if use_max_fee:
            preferences["max_entrance_fee_inr"] = float(max_fee)
        if use_max_duration:
            preferences["max_visit_time_hours"] = float(max_duration)
        if require_dslr:
            preferences["require_dslr_allowed"] = True
        if require_airport:
            preferences["require_airport_within_50km"] = True
        if sparse_hard:
            preferences["sparse_constraints_are_hard"] = True

        try:
            with st.spinner("Searching saved artifacts and ranking places… The first search may load pretrained models."):
                result = recommend_places(
                    query=query.strip(),
                    preferences=preferences or None,
                    top_k=top_k,
                    candidate_k=10,
                )
        except (ValueError, TypeError) as exc:
            st.error(f"Please check your search preferences: {exc}")
        except Exception as exc:
            st.error("The recommendation pipeline could not complete.")
            with st.expander("Technical details for local debugging"):
                st.exception(exc)
        else:
            st.session_state["tourism_result"] = result
            st.session_state["tourism_preferences"] = preferences

result = st.session_state.get("tourism_result")
if result is None:
    st.info("Enter a travel query and select **Find recommendations** to begin.")
else:
    st.divider()
    st.subheader("Your recommendations")
    info = result.get("info") or {}
    context = result.get("rag_context") or {}
    places = context.get("places") or []
    detected_city = result.get("detected_city")

    summary_a, summary_b, summary_c = st.columns(3)
    summary_a.metric("Recommended places", len(places))
    summary_b.metric("City constraint", detected_city.title() if detected_city else "All India")
    summary_c.metric("Candidates reranked", info.get("cross_encoder_candidate_count", 0))

    st.caption(f"Search: {result.get('query', '')}")
    if st.session_state.get("tourism_preferences"):
        st.caption("Applied preferences: " + ", ".join(
            f"{key}={value}" for key, value in st.session_state["tourism_preferences"].items()
        ))

    if not places:
        st.warning("No sufficiently supported places matched the current query and constraints. Try broader wording or fewer filters.")
    else:
        for place in places:
            render_place(place)

    st.subheader("Grounded recommendation summary")
    st.text(result.get("answer") or "No grounded response available.")
    st.caption("Deterministic, evidence-grounded response. No external generative LLM is used.")

    with st.expander("Technical provenance and grounding policy"):
        st.write("**Detected city:**", detected_city or "None (global search)")
        st.write("**Retrieval query:**", result.get("retrieval_query"))
        st.write("**Metadata safety audit:**", "Passed" if info.get("metadata_audit_passed") else "Not passed")
        st.json(context.get("grounding_policy") or {})

    st.download_button(
        "Download grounded evidence (JSON)",
        data=json.dumps(context, ensure_ascii=False, indent=2, default=str),
        file_name="tourism_recommendation_evidence.json",
        mime="application/json",
    )
