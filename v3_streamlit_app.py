"""V3 Tourism Recommender — Streamlit MVP.

Uses the existing V2/V3 production backend.
Does not invent route times, arrival times or attraction facts.
"""

from datetime import date, datetime, time

import pandas as pd
import streamlit as st

from src.retrieval import PLACES_MASTER_PATH
from src.v3_nationwide_selection import (
    build_destination_options,
    build_attraction_options,
)
from src.v3_planning_candidates import build_planning_candidates
from src.v3_planning_controller import run_v3_planning
from src.v3_recommendation_gallery import build_recommendation_gallery
from src.v3_attraction_media import discover_attraction_photo
from src.v3_nationwide_map_merge import build_nationwide_recommendation_map
from src.v3_itinerary_presenter import present_v3_itinerary
from src.v3_uncertainty_display import build_selection_uncertainty
from src.v3_map_identity import resolve_map_identity
from src.v3_conservative_schedule import assess_conservative_schedule
from src.v3_exclusion_presentation import group_presented_exclusions
from src.v3_agent_runtime import run_v3_agent_runtime


st.set_page_config(
    page_title="Indian Tourism AI Recommender",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    div[data-testid="stVerticalBlock"] {
        gap: 0.65rem;
    }

    div[data-testid="stForm"] {
        padding-top: 0.7rem;
        padding-bottom: 0.7rem;
    }

    div[data-testid="stForm"] h3 {
        margin-top: 0.25rem;
        margin-bottom: 0.25rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.025em;
    }

    h1 {
        font-size: clamp(1.9rem, 3vw, 2.6rem);
        font-weight: 750;
    }

    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.07);
        border: 1px solid rgba(128, 128, 128, 0.16);
        border-radius: 14px;
        padding: 16px;
    }

    div[data-testid="stForm"] {
        border-radius: 16px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



@st.cache_data(show_spinner=False)
def load_itinerary_rating_metadata():
    catalog = pd.read_parquet(
        PLACES_MASTER_PATH,
        columns=["place_id", "rating_mean", "review_count"],
    )

    if catalog["place_id"].isna().any():
        raise ValueError("Catalog contains missing place IDs.")

    if catalog["place_id"].duplicated().any():
        raise ValueError("Catalog contains duplicate place IDs.")

    return catalog.set_index("place_id").to_dict("index")



@st.cache_data(ttl=86400, max_entries=256, show_spinner=False)
def load_automatic_attraction_photo(place_id, place_name, city):
    """Best-effort photo lookup; never affects itinerary eligibility."""
    return discover_attraction_photo(place_id, place_name, city)


@st.cache_data(show_spinner=False)
def load_verified_attraction_photos():
    """Load only locally verified photograph records."""
    import csv
    from pathlib import Path

    registry = Path("assets/attractions/photo_registry.csv")

    if not registry.is_file():
        return {}

    records = {}

    with registry.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            if row.get("verification_status") != "VERIFIED_COMMONS_METADATA":
                continue

            place_id = str(row.get("place_id", "")).strip()
            filename = str(row.get("filename", "")).strip()

            if not place_id or not filename:
                continue

            if Path(filename).name != filename:
                continue

            image_path = registry.parent / filename

            if not image_path.is_file():
                continue

            if place_id in records:
                raise ValueError("Duplicate verified photo place_id.")

            records[place_id] = {
                "path": str(image_path),
                "creator": row.get("attribution_text") or row.get("creator", ""),
                "license": row.get("license", ""),
                "license_url": row.get("license_url", ""),
                "source_page": row.get("source_page", ""),
                "modification_note": row.get("modification_note", ""),
            }

    return records



def render_clickable_attraction_photo(
    photo_path, place_id, place_name, stop_index
):
    """Render a clickable thumbnail with a CSS-only enlarged overlay."""
    import base64
    import html
    from pathlib import Path

    image_path = Path(photo_path)
    data = image_path.read_bytes()

    if not data.startswith(bytes([255, 216, 255])):
        raise ValueError("Attraction photograph must be a JPEG.")

    encoded = base64.b64encode(data).decode("ascii")

    safe_id = "".join(
        character if character.isalnum() else "-"
        for character in str(place_id)
    )

    unique_id = f"{safe_id}-{int(stop_index)}"
    alt = html.escape(str(place_name), quote=True)
    image_src = "data:image/jpeg;base64," + encoded

    markup = f"""
    <style>
    #photo-thumb-{unique_id} {{
        display: block;
        width: 100%;
        cursor: zoom-in;
    }}

    #photo-thumb-{unique_id} img {{
        display: block;
        width: 100%;
        aspect-ratio: 4 / 3;
        object-fit: cover;
        border-radius: 8px;
    }}

    #photo-overlay-{unique_id} {{
        display: none;
        position: fixed;
        inset: 0;
        z-index: 999999;
        align-items: center;
        justify-content: center;
    }}

    #photo-overlay-{unique_id}:target {{
        display: flex;
    }}

    #photo-overlay-{unique_id} .photo-backdrop {{
        position: absolute;
        inset: 0;
        background: rgba(0, 0, 0, 0.92);
    }}

    #photo-overlay-{unique_id} .photo-large {{
        position: relative;
        z-index: 1;
        max-width: 92vw;
        max-height: 85vh;
        width: auto;
        height: auto;
        object-fit: contain;
        border-radius: 8px;
    }}

    #photo-overlay-{unique_id} .photo-close {{
        position: absolute;
        top: 16px;
        right: 24px;
        z-index: 2;
        color: white;
        font-size: 38px;
        line-height: 1;
        text-decoration: none;
        cursor: pointer;
    }}
    </style>

    <span id="photo-close-{unique_id}"></span>

    <a
        id="photo-thumb-{unique_id}"
        href="#photo-overlay-{unique_id}"
        aria-label="Enlarge photograph of {alt}"
        title="Click to enlarge photograph"
    >
        <img src="{image_src}" alt="{alt}" loading="lazy">
    </a>

    <div
        id="photo-overlay-{unique_id}"
        role="dialog"
        aria-label="Enlarged photograph of {alt}"
    >
        <a
            class="photo-backdrop"
            href="#photo-close-{unique_id}"
            aria-label="Close enlarged photograph"
        ></a>

        <img class="photo-large" src="{image_src}" alt="{alt}">

        <a
            class="photo-close"
            href="#photo-close-{unique_id}"
            aria-label="Close photograph"
        >&#215;</a>
    </div>
    """

    st.html(markup)


def build_v3_itinerary_pdf(
    presented,
    evidence_stops,
    planning_mode,
    planning_source,
    visit_date,
    start_time,
    schedule_assessment=None,
):
    """Export the displayed provisional plan without recalculating it."""
    import csv
    from html import escape
    from io import BytesIO
    from pathlib import Path

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        HRFlowable,
        CondPageBreak,
    )

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=43,
        leftMargin=43,
        topMargin=42,
        bottomMargin=42,
        title="Indian Tourism Recommender V3 - Itinerary",
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="V3Title",
            parent=styles["Title"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17365D"),
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="V3Body",
            parent=styles["BodyText"],
            leading=15,
            spaceAfter=7,
            wordWrap="CJK",
        )
    )

    story = []

    def paragraph(value, style="V3Body"):
        story.append(
            Paragraph(escape(str(value)), styles[style])
        )

    def heading(value):
        story.append(Spacer(1, 9))
        paragraph(value, "Heading2")

    approved = load_approved_evidence_for_display()

    with Path(
        "v3_data/attraction_locations.csv"
    ).open("r", encoding="utf-8-sig", newline="") as handle:
        locations = {
            row["place_id"]: row
            for row in csv.DictReader(handle)
            if row.get("verification_status") == "user_confirmed"
        }

    evidence_to_place = {
        evidence_id: record.get("place_id", "")
        for evidence_id, record in approved.items()
    }

    paragraph("Indian Tourism Recommender V3", "V3Title")
    paragraph("Evidence-grounded provisional itinerary")
    verified_destination = presented.get("city")
    if verified_destination:
        paragraph(
            "Destination: "
            + str(verified_destination).title()
        )

    paragraph("Visit date: " + str(visit_date))
    paragraph("Requested starting time: " + str(start_time))
    paragraph("Planning mode: " + str(planning_mode))
    paragraph("Planning source: " + str(planning_source))

    story.append(Spacer(1, 8))
    story.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=colors.HexColor("#CBD5E1"),
        )
    )

    heading("Itinerary summary")
    paragraph("Planned stops: " + str(presented["stop_count"]))

    subtotal = presented["known_admission_subtotal_inr"]
    paragraph(
        "Known admission subtotal: INR " + str(subtotal)
        if presented["stop_count"] > 0
        else "Known admission subtotal: Not applicable"
    )
    exclusion_groups = group_presented_exclusions(
        presented["held"]
    )

    paragraph(
        "Attractions requiring review: "
        + str(len(exclusion_groups["review_items"]))
    )
    paragraph(
        "Geographic entries excluded: "
        + str(len(exclusion_groups["geographic_exclusions"]))
    )
    paragraph(
        "Known schedule conflicts: "
        + str(len(exclusion_groups["schedule_conflicts"]))
    )

    if schedule_assessment is not None:
        if not isinstance(schedule_assessment, dict):
            raise TypeError("Invalid scheduling assessment.")

        scheduled_stops = schedule_assessment["stops"]

        if schedule_assessment["stop_count"] != len(evidence_stops):
            raise ValueError("Scheduling stop count mismatch.")

        if len(scheduled_stops) != len(evidence_stops):
            raise ValueError("Scheduling stop list mismatch.")

        if schedule_assessment["route_optimized"] is not False:
            raise ValueError("Unsupported route optimization claim.")

        if schedule_assessment["full_day_feasibility_verified"] is not False:
            raise ValueError("Unsupported full-day feasibility claim.")

        for index, (planned, scheduled) in enumerate(
            zip(evidence_stops, scheduled_stops), start=1
        ):
            if planned["place_id"] != scheduled["place_id"]:
                raise ValueError("Scheduling place ID mismatch.")

            if scheduled["stop_number"] != index:
                raise ValueError("Scheduling stop order mismatch.")

            if scheduled["visit_duration_minutes"] is not None:
                raise ValueError("Unverified visit duration.")

            if scheduled["travel_time_from_previous_minutes"] is not None:
                raise ValueError("Unverified travel time.")

            requested = scheduled["requested_start_time"]

            if index == 1:
                if requested is None:
                    raise ValueError("Missing requested first start.")
                if scheduled["arrival_status"] != "USER_REQUESTED_START":
                    raise ValueError("Invalid first arrival status.")
                if requested != planned.get("proposed_arrival_time"):
                    raise ValueError("First arrival mismatch.")
            else:
                if requested is not None:
                    raise ValueError("Unsupported later arrival.")
                if scheduled["arrival_status"] != "NOT_SCHEDULED":
                    raise ValueError("Invalid later arrival status.")
                if planned.get("proposed_arrival_time") is not None:
                    raise ValueError("Unsupported planned later arrival.")

        heading("Scheduling and travel feasibility")
        paragraph(
            "Scheduling assessment: "
            + schedule_assessment["schedule_assessment_status"]
            .replace("_", " ").title()
        )

        if not scheduled_stops:
            paragraph(
                "No evidence-supported stops are available for scheduling."
            )

        for scheduled in scheduled_stops:
            paragraph(
                "Stop " + str(scheduled["stop_number"])
                + " - " + str(scheduled["place"])
            )

            requested = scheduled["requested_start_time"]

            if requested is not None:
                paragraph("User-requested start: " + str(requested))
            else:
                paragraph("Arrival time: not scheduled")

            paragraph("Visit duration: unverified")

            if scheduled["stop_number"] > 1:
                paragraph(
                    "Travel time from previous stop: unverified"
                )

        for limitation in schedule_assessment["limitations"]:
            paragraph("- " + str(limitation))

    for index, stop in enumerate(evidence_stops, start=1):
        place = str(
            stop.get("place")
            or stop.get("place_name")
            or "Attraction"
        )

        story.append(CondPageBreak(230))
        heading("Stop " + str(index) + ": " + place)

        section = stop.get("visit_section")
        if section:
            paragraph(
                "Visit section: "
                + str(section).replace("_", " ").title()
            )

        arrival = stop.get("proposed_arrival_time")
        if index == 1 and arrival is not None:
            paragraph("Requested start: " + str(arrival))
        else:
            paragraph("Arrival: Not scheduled")

        admission = stop.get("admission_total_inr")
        paragraph(
            "Admission: INR " + str(admission)
            if admission is not None
            else "Admission: Not verified"
        )

        evidence_ids = (
            stop.get("supporting_evidence_ids")
            or stop.get("evidence_ids")
            or []
        )

        paragraph(
            "Supporting evidence records: "
            + str(len(evidence_ids))
        )

        place_ids = {
            evidence_to_place[evidence_id]
            for evidence_id in evidence_ids
            if evidence_id in evidence_to_place
            and evidence_to_place[evidence_id]
        }

        explicit_id = stop.get("place_id")
        if explicit_id:
            place_ids.add(str(explicit_id))

        if len(place_ids) == 1:
            place_id = next(iter(place_ids))
            location = locations.get(place_id)
            if location:
                paragraph(
                    "User-confirmed attraction map coordinates: "
                    + str(location["latitude"])
                    + ", "
                    + str(location["longitude"])
                )
            else:
                paragraph(
                    "Attraction-level map coordinates: "
                    "Not confirmed"
                )
        else:
            paragraph(
                "Attraction-level map coordinates: "
                "Not confirmed"
            )

        for evidence_id in evidence_ids:
            record = approved.get(evidence_id)

            if record is None:
                paragraph(
                    "Evidence not confirmed approved: "
                    + str(evidence_id)
                )
                continue

            label, value = describe_evidence_for_display(record)
            pdf_value = str(value).replace("₹", "INR ")
            paragraph(label + ": " + pdf_value)
            paragraph("Evidence ID: " + str(evidence_id))

            approval = record["approval"]
            paragraph(
                "Approval: "
                + str(approval.get("decision", "Unknown"))
                + " | Reviewed: "
                + str(approval.get("reviewed_on", "Unknown"))
            )

            source_url = record.get("source_url", "")
            if source_url.startswith(("https://", "http://")):
                safe_url = escape(source_url, quote=True)
                story.append(
                    Paragraph(
                        'Official source: <link href="'
                        + safe_url
                        + '">'
                        + escape(source_url)
                        + "</link>",
                        styles["V3Body"],
                    )
                )

    exclusion_sections = (
        ("Attractions requiring review", "review_items"),
        ("Geographic entries excluded", "geographic_exclusions"),
        ("Known schedule conflicts", "schedule_conflicts"),
    )

    for section_heading, section_key in exclusion_sections:
        items = exclusion_groups[section_key]
        if not items:
            continue

        heading(section_heading)
        for held in items:
            paragraph(
                str(held["place"])
                + " - "
                + str(held["reason"])
                + " - "
                + str(held["display_message"])
            )

    heading("Important planning limitations")
    for limitation in (
        "This is a provisional itinerary, not a verified full-day schedule.",
        "Travel times and route optimization have not been verified.",
        "Only the first stop may have a requested start time; later arrivals are unscheduled.",
        "Later-stop opening-hour feasibility must be checked at actual arrival times.",
        "Holiday exceptions and live ticket availability are not guaranteed.",
        "Known admission subtotal is not the total trip cost.",
        "Attraction map coordinates are user-confirmed, not route verification.",
    ):
        paragraph("- " + limitation)

    document.build(story)
    pdf_bytes = buffer.getvalue()

    if not pdf_bytes.startswith(b"%PDF-"):
        raise RuntimeError("PDF generation verification failed.")

    return pdf_bytes


def load_approved_evidence_for_display():
    """Read approved evidence metadata for UI presentation only."""
    import csv
    from pathlib import Path

    evidence_path = Path("v3_data/attraction_evidence.csv")
    ledger_path = Path("v3_data/evidence_approval_ledger.csv")

    with ledger_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        decisions = list(csv.DictReader(handle))

    latest_decisions = {}

    for row in decisions:
        evidence_id = row.get("evidence_id", "").strip()
        if evidence_id:
            latest_decisions[evidence_id] = row

    approved_ids = {
        evidence_id
        for evidence_id, row in latest_decisions.items()
        if row.get("decision", "").strip().lower() == "approved"
        and row.get("source_checked", "").strip().lower()
        in ("true", "1", "yes")
    }

    with evidence_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        records = list(csv.DictReader(handle))

    return {
        row["evidence_id"]: {
            **row,
            "approval": latest_decisions[row["evidence_id"]],
        }
        for row in records
        if row.get("evidence_id") in approved_ids
    }


def describe_evidence_for_display(record):
    """Format recorded facts without introducing new claims."""
    kind = record.get("evidence_type", "")
    category = record.get("category", "")
    value = record.get("value", "")
    unit = record.get("unit", "")

    category_label = category.replace("_", " ").title()

    if kind == "entrance_fee" and unit == "INR":
        return category_label, f"₹{value} per applicable ticket"

    if kind == "opening_hours":
        return (
            "Opening hours — " + category_label,
            f"{value} ({unit.replace('_', ' ')})",
        )

    if kind == "weekly_off":
        return "Regular closure — " + category_label, value

    return (
        kind.replace("_", " ").title() + " — " + category_label,
        f"{value} {unit}".strip(),
    )


with st.sidebar:
    st.markdown("## 🧭 Tourism AI")
    st.caption("Indian Tourism AI Recommender")

    st.divider()

    st.markdown("### About Project")
    st.write(
        "Discover Indian attractions using AI-powered retrieval "
        "and build provisional itineraries using approved "
        "planning evidence where available."
    )

    st.markdown("### Dataset")
    st.markdown("**15,239** attractions")
    st.markdown("**1,482,466** reviews")
    st.caption(
        "Nationwide recommendation catalog. Verified admission, "
        "hours and map coverage are more limited."
    )

    st.markdown("### Technology")
    st.caption(
        "Python · Streamlit · TF-IDF · Sentence Transformers · "
        "FAISS · Cross-Encoder · Evidence-grounded planning"
    )


st.title("Indian Tourism AI Recommender")
st.caption("Discover attractions. Plan your trip.")


@st.cache_data(show_spinner=False)
def load_catalog():
    return pd.read_parquet(PLACES_MASTER_PATH)


def positive_categories(values):
    return {
        name: int(count)
        for name, count in values.items()
        if int(count) > 0
    }


mode_label = st.radio(
    "Choose your planning mode",
    [
        "AI recommendations",
        "Plan selected attractions",
    ],
    horizontal=True,
)

mode = (
    "recommend"
    if mode_label == "AI recommendations"
    else "user_selected"
)
    
use_agent = st.checkbox(
    "Use AI agent (optional)",
    value=False,
    help=(
        "When OPENAI_API_KEY is configured, the model selects "
        "an approved planning tool. Otherwise the application "
        "uses the verified offline fallback."
    ),
)

if use_agent:
    st.caption(
        "Agent mode: OpenAI tool selection when configured; "
        "otherwise deterministic planning. "
        "API usage may incur charges."
    )



selected_city_canonical = "kolkata"

if mode == "user_selected":
    selection_catalog = load_catalog()

    destination_options = build_destination_options(
        selection_catalog
    )

    destination_keys = [
        item["city_canonical"]
        for item in destination_options
    ]

    destination_labels = {
        item["city_canonical"]: item["label"]
        for item in destination_options
    }

    selected_city_canonical = st.selectbox(
        "Choose your destination",
        options=destination_keys,
        index=destination_keys.index("kolkata"),
        format_func=lambda city: destination_labels[city],
        key="v3_nationwide_destination",
        help=(
            "Search by destination name. The number in each "
            "option shows catalog coverage, not the number "
            "of operationally verified attractions."
        ),
    )

    st.caption(
        "Catalog locations and attractions are provisional. "
        "Unverified geographic or operational information "
        "will not be treated as approved evidence."
    )


with st.form("v3_planning_form"):

    query = None
    selections = []
    visitors = {}

    if mode == "recommend":

        query = st.text_input(
            "Where would you like to explore?",
            value="museums and historical attractions in Kolkata",
            help=(
                "Describe the destination and interests, such as "
                "museums, history, nature or temples. Results use "
                "the existing V2 recommendation system."
            ),
        )

    left, right = st.columns(2)

    with left:
        visit_date = st.date_input(
            "Visit date",
            value=date.today(),
            min_value=date.today(),
        )

    with right:
        start_time = st.time_input(
            "Starting time",
            value=time(11, 0),
            step=1800,
        )

    if mode == "recommend":

        st.markdown("**Visitors**")

        c1, c2, c3 = st.columns(3)

        with c1:
            adults = st.number_input(
                "Adults",
                min_value=0,
                max_value=30,
                value=2,
                step=1,
            )

        with c2:
            under_18 = st.number_input(
                "Below 18",
                min_value=0,
                max_value=30,
                value=0,
                step=1,
            )

        with c3:
            foreign = st.number_input(
                "Foreign-national admission",
                min_value=0,
                max_value=30,
                value=0,
                step=1,
            )

        visitors = positive_categories({
            "adult_admission": adults,
            "below_18_admission": under_18,
            "foreign_national_admission": foreign,
        })

        with st.expander(
            "Advanced recommendation settings",
            expanded=False,
        ):
            top_k = st.slider(
                "Maximum recommendations",
                min_value=1,
                max_value=5,
                value=5,
            )

            st.caption(
                "Admission categories are attraction-specific. "
                "Unknown ticket prices will not be assumed."
            )

    else:

        if selected_city_canonical == "kolkata":
            st.markdown("#### Choose attractions")

            st.caption(
                "This initial interface supports two Kolkata "
                "attractions with approved operating evidence. "
                "These are user selections, not V2 rankings."
            )

            include_indian = st.checkbox(
                "Indian Museum (Jadu Ghar)",
                value=True,
            )

            if include_indian:
                st.markdown("**Indian Museum ticket categories**")

                c1, c2, c3 = st.columns(3)

                with c1:
                    indian_adults = st.number_input(
                        "Indian Museum — adults",
                        min_value=0,
                        max_value=30,
                        value=2,
                        step=1,
                    )

                with c2:
                    indian_under18 = st.number_input(
                        "Indian Museum — below 18",
                        min_value=0,
                        max_value=30,
                        value=1,
                        step=1,
                    )

                with c3:
                    indian_foreign = st.number_input(
                        "Indian Museum — foreign nationals",
                        min_value=0,
                        max_value=30,
                        value=0,
                        step=1,
                    )

                selections.append({
                    "place_id": "INPLACE_07020",
                    "visitors": positive_categories({
                        "adult_admission": indian_adults,
                        "below_18_admission": indian_under18,
                        "foreign_national_admission": indian_foreign,
                    }),
                })

            include_victoria = st.checkbox(
                "Victoria Memorial Hall",
                value=True,
            )

            if include_victoria:

                section_label = st.selectbox(
                    "Victoria Memorial section",
                    [
                        "Museum galleries",
                        "Gardens",
                    ],
                )

                section = (
                    "museum_galleries"
                    if section_label == "Museum galleries"
                    else "gardens"
                )

                if section == "museum_galleries":

                    st.caption(
                        "The approved museum-gallery ticket category "
                        "used here is for Indian nationals. "
                        "Other categories are not assumed."
                    )

                    count = st.number_input(
                        "Indian-national museum-gallery tickets",
                        min_value=0,
                        max_value=30,
                        value=2,
                        step=1,
                    )

                    section_visitors = positive_categories({
                        "indian_national_museum_galleries": count,
                    })

                else:

                    count = st.number_input(
                        "Garden single-entry tickets",
                        min_value=0,
                        max_value=30,
                        value=2,
                        step=1,
                    )

                    section_visitors = positive_categories({
                        "garden_daily_single_entry": count,
                    })

                selections.append({
                    "place_id": "INPLACE_07148",
                    "section": section,
                    "visitors": section_visitors,
                })


        st.markdown("#### Explore catalog attractions")

        city_options = build_attraction_options(
            selection_catalog,
            selected_city_canonical,
        )

        specialized_ids = (
            {"INPLACE_07020", "INPLACE_07148"}
            if selected_city_canonical == "kolkata"
            else set()
        )

        available = {
            item["place_id"]: item
            for item in city_options
            if item["place_id"] not in specialized_ids
        }

        planning_candidates = build_planning_candidates(
            selection_catalog,
            selected_city_canonical,
        )

        candidate_lookup = {
            item["place_id"]: item
            for item in planning_candidates["candidates"]
        }

        assert set(available).issubset(candidate_lookup)

        st.caption(
            "Select attractions for consideration, not "
            "confirmed itinerary stops. Identity, opening "
            "hours and admission still require verification."
        )

        additional_ids = st.multiselect(
            "Additional attractions to consider",
            options=list(available),
            format_func=lambda place_id: (
                available[place_id]["label"]
                + " | Dataset rating "
                + f"{candidate_lookup[place_id]["dataset_rating_mean"]:.2f}/5"
                + " | "
                + f"{candidate_lookup[place_id]["dataset_review_count"]:,} reviews"
            ),
            help=(
                "Unverified attractions may be held "
                "by the existing itinerary planner."
            ),
        )

        st.markdown("**Visitors for catalog attractions**")

        extra_c1, extra_c2, extra_c3 = st.columns(3)

        with extra_c1:
            extra_adults = st.number_input(
                "Catalog attraction adults",
                min_value=0,
                max_value=30,
                value=2,
                step=1,
            )

        with extra_c2:
            extra_under18 = st.number_input(
                "Catalog attraction below 18",
                min_value=0,
                max_value=30,
                value=0,
                step=1,
            )

        with extra_c3:
            extra_foreign = st.number_input(
                "Catalog attraction foreign nationals",
                min_value=0,
                max_value=30,
                value=0,
                step=1,
            )

        extra_visitors = positive_categories({
            "adult_admission": extra_adults,
            "below_18_admission": extra_under18,
            "foreign_national_admission": extra_foreign,
        })

        for place_id in additional_ids:
            selections.append({
                "place_id": place_id,
                "visitors": dict(extra_visitors),
            })

    submitted = st.form_submit_button(
        "Generate provisional itinerary",
        type="primary",
        use_container_width=True,
    )


with st.expander("Planning limitations", expanded=False):
    st.caption(
        "Plans are provisional. Verified admission evidence is used "
        "where available. Travel times, later arrival times, holiday "
        "exceptions and full-day feasibility are not guaranteed."
    )




if submitted:

    if mode == "recommend" and not query.strip():
        st.error("Enter a tourism query.")

    elif mode == "recommend" and not visitors:
        st.error("Specify at least one visitor.")

    elif mode == "user_selected" and not selections:
        st.error("Select at least one attraction.")

    elif (
        mode == "user_selected"
        and any(not item["visitors"] for item in selections)
    ):
        st.error(
            "Every selected attraction needs at least "
            "one applicable ticket."
        )

    else:

        try:

            with st.spinner(
                "Preparing your provisional itinerary and checking "
                "available approved evidence..."
            ):

                catalog = load_catalog()

                date_text = visit_date.isoformat()
                time_text = start_time.strftime("%H:%M")

                if use_agent:

                    if mode == "recommend":
                        agent_arguments = {
                            "query": query,
                            "visit_date": date_text,
                            "start_time": time_text,
                            "visitors": visitors,
                            "top_k": top_k,
                        }
                        agent_message = query

                    else:
                        agent_arguments = {
                            "city_canonical": selected_city_canonical,
                            "selections": selections,
                            "visit_date": date_text,
                            "start_time": time_text,
                        }
                        agent_message = (
                            "Plan my explicitly selected attractions "
                            f"in {selected_city_canonical} using available evidence."
                        )

                    agent_result = run_v3_agent_runtime(
                        {
                            "user_message": agent_message,
                            "mode": mode,
                            "arguments": agent_arguments,
                        },
                        catalog=catalog,
                        use_llm=True,
                    )

                    presented = {
                        "heading": agent_result["answer"].splitlines()[0],
                        "city": agent_result["city"],
                        "stop_count": agent_result["stop_count"],
                        "known_admission_subtotal_inr": (
                            agent_result["known_admission_subtotal_inr"]
                        ),
                        "held": agent_result["held"],
                        "answer": agent_result["answer"],
                    }

                    planning_mode = agent_result["planning_mode"]
                    planning_source = (
                        "V2_RANKED_RECOMMENDATIONS"
                        if mode == "recommend"
                        else "EXPLICIT_USER_SELECTION"
                    )
                    agent_status = agent_result["agent_status"]
                    llm_status = agent_result["llm_status"]
                    evidence_stops = agent_result["evidence_stops"]
                    schedule_assessment = agent_result["schedule_assessment"]

                else:

                    if mode == "recommend":

                        result = run_v3_planning(
                            mode="recommend",
                            query=query,
                            visit_date=date_text,
                            start_time=time_text,
                            visitors=visitors,
                            top_k=top_k,
                            candidate_k=10,
                            catalog=catalog,
                        )

                    else:

                        result = run_v3_planning(
                            mode="user_selected",
                            city_canonical=selected_city_canonical,
                            selections=selections,
                            visit_date=date_text,
                            start_time=time_text,
                            catalog=catalog,
                        )

                    presented = present_v3_itinerary(result)

                    planning_mode = result["mode"]
                    planning_source = result["source"]
                    agent_status = result["agent_status"]
                    llm_status = "OFFLINE_DIRECT_PLANNING"
                    evidence_stops = result["draft"]["stops"]
                    schedule_assessment = (
                        assess_conservative_schedule(result)
                    )

            st.success("Provisional itinerary generated")

            st.subheader(presented["heading"])

            if use_agent:
                if llm_status == "NO_API_KEY_FALLBACK":
                    st.caption(
                        "🟢 Offline planning mode — "
                        "verified planning tools are active."
                    )
                elif llm_status == "LLM_PROVIDER_ERROR_FALLBACK":
                    st.caption(
                        "🟡 AI service unavailable — "
                        "verified offline planning is active."
                    )
                else:
                    st.caption("Agent mode: " + llm_status)

            # Compact result summary — presentation only.
            summary_stops, summary_cost, summary_held = st.columns(3)

            subtotal = presented["known_admission_subtotal_inr"]

            with summary_stops:
                with st.container(border=True):
                    st.metric("Stops planned", presented["stop_count"])

            with summary_cost:
                with st.container(border=True):
                    st.metric(
                        "Known admission",
                        f"₹{subtotal}" if presented["stop_count"] > 0 else "N/A",
                    )

            with summary_held:
                with st.container(border=True):
                    st.metric("Needs checking", len(presented["held"]))

            st.caption(
                "Known admission is a ticket subtotal, "
                "not the total trip cost."
            )

            if mode == "user_selected" and not use_agent:
                uncertainty_records = build_selection_uncertainty(
                    selections,
                    date_text,
                    time_text,
                )

                with st.expander(
                    "Opening hours and admission verification",
                    expanded=False,
                ):
                    st.caption(
                        "These explanations use existing approved "
                        "evidence. They do not confirm bookings, "
                        "live opening hours or full-day feasibility."
                    )

                    for record in uncertainty_records:
                        place_id = record["place_id"]
                        selected_row = catalog.loc[
                            catalog["place_id"] == place_id
                        ]
                        assert len(selected_row) == 1

                        place_name = str(
                            selected_row.iloc[0]["place"]
                        )
                        section = record["section"]

                        if section:
                            place_name += (
                                " — "
                                + str(section).replace("_", " ").title()
                            )

                        st.markdown("**" + place_name + "**")
                        st.caption(
                            "Opening hours: "
                            + record["schedule_explanation"]
                        )
                        st.caption(
                            "Admission: "
                            + record["admission_explanation"]
                        )
                        st.caption(
                            "Planning: "
                            + record["planning_explanation"]
                        )

                        if record["admission_total_verified"]:
                            st.caption(
                                "Approved admission for selected "
                                "visitors: ₹"
                                + str(record["admission_total_inr"])
                            )
                        else:
                            st.caption(
                                "Complete admission total: unknown"
                            )

            if schedule_assessment is not None:
                with st.expander(
                    "Scheduling and travel feasibility",
                    expanded=False,
                ):
                    st.caption(
                        "Scheduling assessment: "
                        + schedule_assessment[
                            "schedule_assessment_status"
                        ].replace("_", " ").title()
                    )

                    if not schedule_assessment["stops"]:
                        st.info(
                            "No evidence-supported stops are "
                            "available for scheduling."
                        )

                    for scheduled_stop in (
                        schedule_assessment["stops"]
                    ):
                        st.markdown(
                            "**Stop "
                            + str(scheduled_stop["stop_number"])
                            + " — "
                            + str(scheduled_stop["place"])
                            + "**"
                        )

                        requested = scheduled_stop[
                            "requested_start_time"
                        ]

                        if requested is not None:
                            st.caption(
                                "User-requested start: "
                                + str(requested)
                            )
                        else:
                            st.caption(
                                "Arrival time: not scheduled"
                            )

                        st.caption(
                            "Visit duration: unverified"
                        )

                        if scheduled_stop["stop_number"] > 1:
                            st.caption(
                                "Travel time from previous stop: "
                                "unverified"
                            )

                    for limitation in (
                        schedule_assessment["limitations"]
                    ):
                        st.caption("• " + str(limitation))


            # UI21: V2-ranked discovery gallery, separate from itinerary approval.
            if mode == "recommend" and not use_agent:
                v2_output = result.get("v2_result") or {}
                recommendation_frame = v2_output.get("recommendations")

                if recommendation_frame is not None:
                    gallery_records = build_recommendation_gallery(
                        recommendation_frame,
                        photo_metadata=load_verified_attraction_photos(),
                        evidence_stops=evidence_stops,
                    )

                    st.markdown("### Recommended attractions")
                    st.caption(
                        "V2-ranked discovery results. An attraction can appear "
                        "here even when itinerary evidence needs checking."
                    )

                    if not gallery_records:
                        st.info("No matching attractions were returned.")

                    for gallery_item in gallery_records:
                        with st.container(border=True):
                            photo = gallery_item["photo"]
                            automatic_photo = None

                            if (
                                photo is None
                                and gallery_item["place"].casefold().strip()
                                != gallery_item["city"].casefold().strip()
                            ):
                                automatic_photo = load_automatic_attraction_photo(
                                    gallery_item["place_id"],
                                    gallery_item["place"],
                                    gallery_item["city"],
                                )

                            if photo is not None or automatic_photo is not None:
                                image_col, info_col = st.columns(
                                    [1.4, 4.6],
                                    gap="small",
                                    vertical_alignment="top",
                                )
                                with image_col:
                                    if photo is not None:
                                        render_clickable_attraction_photo(
                                            photo["path"],
                                            gallery_item["place_id"],
                                            gallery_item["place"],
                                            gallery_item["rank"],
                                        )
                                    else:
                                        st.image(
                                            automatic_photo["photo_url"],
                                            caption="Automatically matched",
                                            use_container_width=True,
                                        )
                            else:
                                info_col = st.container()

                            with info_col:
                                st.caption(
                                    f"RECOMMENDATION {gallery_item['rank']:02d}"
                                )
                                st.markdown(
                                    "**" + gallery_item["place"] + "**"
                                )
                                st.caption(gallery_item["city"])

                                rating = gallery_item["rating_mean"]
                                reviews = gallery_item["review_count"]

                                if rating is not None and reviews is not None:
                                    st.caption(
                                        f"★ {rating:.2f}/5 · "
                                        f"{reviews:,} dataset reviews"
                                    )

                                if photo is not None:
                                    st.caption(
                                        "Photo: " + str(photo["creator"])
                                        + " · " + str(photo["license"])
                                    )
                                elif automatic_photo is not None:
                                    st.caption(
                                        "Automatically matched photo: "
                                        + str(automatic_photo["creator"])
                                        + " · "
                                        + str(automatic_photo["license"])
                                    )
                                    st.link_button(
                                        "Photo source",
                                        automatic_photo["source_page"],
                                    )
                                    if automatic_photo.get("license_url"):
                                        st.link_button(
                                            "Image license",
                                            automatic_photo["license_url"],
                                        )
                                else:
                                    st.caption("Photo unavailable")

                                st.caption(gallery_item["planning_label"])


            # UI21: shared discovery map, separate from strict itinerary pins.
            if mode == "recommend" and not use_agent:
                if recommendation_frame is not None:
                    discovery_map = build_nationwide_recommendation_map(
                        gallery_records,
                        max_lookups=3,
                        max_http_requests=6,
                    )

                    st.markdown("### Recommended attraction locations")
                    st.caption(
                        "One shared map for V2-ranked attractions with "
                        "available attraction-level coordinates. "
                        "Wikidata candidate pins are provisional, "
                        "not verified itinerary stops."
                    )

                    map_markers = discovery_map["markers"]

                    if map_markers:
                        import pandas as pd

                        map_frame = pd.DataFrame([
                            {
                                "latitude": marker["latitude"],
                                "longitude": marker["longitude"],
                                "place": marker["place"],
                                "rank": marker["rank"],
                                "location_status": marker["location_status"],
                            }
                            for marker in map_markers
                        ])

                        st.map(
                            map_frame,
                            latitude="latitude",
                            longitude="longitude",
                            color="#DC2626",
                            size=85,
                            height=420,
                        )

                        st.caption(
                            f"{len(map_markers)} mapped attraction(s) "
                            f"out of {len(gallery_records)} recommendations."
                        )

                        for marker in map_markers:
                            st.caption(
                                f"{marker['rank']}. {marker['place']} "
                                f"— {marker['source']}"
                            )
                    else:
                        st.info(
                            "No attraction-level coordinates are currently "
                            "available for these recommendations."
                        )

            st.markdown("### Your itinerary")

            if planning_mode == "recommend":
                st.caption(
                    "Stops originate from V2-ranked recommendations. "
                    "Only evidence-supported stops are displayed."
                )
            else:
                st.caption(
                    "These attractions were selected by you, "
                    "not generated by V2 rankings."
                )

            if not evidence_stops:
                st.warning(
                    "No attractions currently qualify as "
                    "provisional itinerary stops."
                )

            rating_metadata = load_itinerary_rating_metadata()
            photo_metadata = load_verified_attraction_photos()

            for index, stop in enumerate(evidence_stops, start=1):
                if not isinstance(stop, dict):
                    raise TypeError("Invalid itinerary stop record.")

                place = str(stop["place"])
                section = stop.get("visit_section")
                arrival = stop.get("proposed_arrival_time")
                admission = stop.get("admission_total_inr")

                evidence_ids = (
                    stop.get("supporting_evidence_ids")
                    or stop.get("evidence_ids")
                    or []
                )

                with st.container(border=True):
                    stop_col, detail_col, cost_col = st.columns(
                        [1, 5, 2],
                        vertical_alignment="top",
                    )

                    with stop_col:
                        st.caption(f"STOP {index:02d}")

                    with detail_col:
                        photo_record = photo_metadata.get(
                            str(stop.get("place_id", ""))
                        )

                        if photo_record is not None:
                            image_col, info_col = st.columns(
                                [1.4, 4.6],
                                gap="small",
                                vertical_alignment="top",
                            )

                            with image_col:
                                render_clickable_attraction_photo(
                                    photo_record["path"],
                                    stop.get("place_id"),
                                    place,
                                    index,
                                )
                        else:
                            info_col = st.container()

                        with info_col:
                            st.markdown(f"**{place}**")

                            if photo_record is not None:
                                credit = str(photo_record["creator"])
                                license_name = str(photo_record["license"])
                                source_url = str(photo_record["source_page"])
                                license_url = str(photo_record["license_url"])

                                st.caption(
                                    f"Photo: {credit} · "
                                    f"[{license_name}]({license_url}) · "
                                    f"[Wikimedia Commons]({source_url}) · "
                                    "Resized thumbnail"
                                )


                            stop_place_id = stop.get("place_id")
                            rating_record = (
                                rating_metadata.get(str(stop_place_id))
                                if stop_place_id
                                else None
                            )

                            if rating_record is not None:
                                rating_value = rating_record.get("rating_mean")
                                review_value = rating_record.get("review_count")

                                if (
                                    pd.notna(rating_value)
                                    and pd.notna(review_value)
                                    and 0 <= float(rating_value) <= 5
                                    and int(review_value) > 0
                                ):
                                    st.caption(
                                        f"★ {float(rating_value):.2f}/5"
                                        f" · {int(review_value):,} dataset reviews"
                                    )

                            if planning_mode == "recommend":
                                st.caption("V2-ranked recommendation")
                            else:
                                st.caption("Selected by you")

                            if section:
                                st.caption(
                                    "Section: "
                                    + str(section).replace("_", " ").title()
                                )

                            if index == 1 and arrival is not None:
                                st.caption(
                                    "Requested start: " + str(arrival)
                                )
                            else:
                                st.caption("Arrival not scheduled")

                            st.markdown(
                                ":green[**Evidence-supported**] "
                                f"· {len(evidence_ids)} approved evidence record(s)"
                            )

                    with cost_col:
                        if admission is None:
                            st.caption("Admission not verified")
                        else:
                            st.markdown(f"**₹{admission}**")
                            st.caption("Known admission")

            st.markdown("### Download your itinerary")

            pdf_data = build_v3_itinerary_pdf(
                presented=presented,
                evidence_stops=evidence_stops,
                planning_mode=planning_mode,
                planning_source=planning_source,
                visit_date=date_text,
                start_time=time_text,
                schedule_assessment=(
                    schedule_assessment
                ),
            )

            st.download_button(
                label="Download itinerary PDF",
                data=pdf_data,
                file_name=(
                    "Indian_Tourism_V3_Itinerary_"
                    + str(date_text).replace("/", "-")
                    + ".pdf"
                ),
                mime="application/pdf",
                key="v3_itinerary_pdf_download",
            )

            st.caption(
                "The PDF contains the same provisional "
                "planning results and approved evidence "
                "references shown above."
            )

            # V3 attraction-level map: approved planning stops only.
            st.markdown("### Planned attraction locations")

            import csv
            from pathlib import Path
            import pandas as pd

            with Path(
                "v3_data/attraction_locations.csv"
            ).open("r", encoding="utf-8-sig", newline="") as handle:
                location_records = list(csv.DictReader(handle))

            with Path(
                "v3_data/attraction_evidence.csv"
            ).open("r", encoding="utf-8-sig", newline="") as handle:
                evidence_records = list(csv.DictReader(handle))

            locations_by_id = {}
            locations_by_name = {}

            for location in location_records:
                if location.get("verification_status") != "user_confirmed":
                    continue

                place_id = location.get("place_id", "").strip()
                place_name = location.get("place_name", "").strip()

                if not place_id or not place_name:
                    continue

                if place_id in locations_by_id:
                    raise ValueError(
                        "Duplicate attraction location: " + place_id
                    )

                lat = float(location["latitude"])
                lon = float(location["longitude"])

                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    raise ValueError(
                        "Invalid attraction coordinates: " + place_id
                    )

                locations_by_id[place_id] = {
                    "place_id": place_id,
                    "place_name": place_name,
                    "latitude": lat,
                    "longitude": lon,
                }

                normalized_name = place_name.casefold()
                locations_by_name.setdefault(
                    normalized_name, []
                ).append(place_id)

            evidence_to_place = {}

            for record in evidence_records:
                evidence_id = record.get("evidence_id", "").strip()
                place_id = record.get("place_id", "").strip()

                if evidence_id:
                    if evidence_id in evidence_to_place:
                        raise ValueError(
                            "Duplicate evidence ID: " + evidence_id
                        )
                    evidence_to_place[evidence_id] = place_id

            map_rows = []
            missing_locations = []

            for stop_number, planned_stop in enumerate(
                evidence_stops, start=1
            ):
                stop_name = str(
                    planned_stop.get("place")
                    or planned_stop.get("place_name")
                    or "Attraction"
                )

                location = resolve_map_identity(
                    planned_stop,
                    locations_by_id,
                    evidence_to_place,
                )

                if location is None:
                    missing_locations.append(stop_name)
                    continue

                map_rows.append({
                    "latitude": location["latitude"],
                    "longitude": location["longitude"],
                    "stop_number": stop_number,
                    "place": stop_name,
                })

            if map_rows:
                map_frame = pd.DataFrame(map_rows)

                st.map(
                    map_frame,
                    latitude="latitude",
                    longitude="longitude",
                    zoom=12,
                )

                for mapped_stop in map_rows:
                    st.caption(
                        "Stop "
                        + str(mapped_stop["stop_number"])
                        + ": "
                        + mapped_stop["place"]
                    )

                st.caption(
                    "Pins use user-confirmed attraction-level "
                    "Google Maps coordinates. No travel route, "
                    "travel time, or full-day feasibility "
                    "has been verified."
                )
            else:
                st.info(
                    "No planned stops have confirmed "
                    "attraction-level map coordinates."
                )

            if missing_locations:
                st.warning(
                    "Map coordinates are not yet verified for: "
                    + ", ".join(missing_locations)
                )

            exclusion_groups = group_presented_exclusions(
                presented["held"]
            )

            exclusion_sections = (
                ("Attractions requiring review", "review_items"),
                ("Geographic entries excluded", "geographic_exclusions"),
                ("Known schedule conflicts", "schedule_conflicts"),
            )

            for section_heading, section_key in exclusion_sections:
                items = exclusion_groups[section_key]
                if not items:
                    continue

                with st.expander(
                    f"{section_heading} ({len(items)})"
                ):
                    for item in items:
                        st.markdown(
                            "**" + str(item["place"]) + "**"
                        )
                        st.caption(
                            str(item["reason"]).replace("_", " ").title()
                        )
                        st.caption(item["display_message"])

            with st.expander(
                "Full planning details and limitations"
            ):
                st.text(presented["answer"])

            st.caption(
                "No route optimization or full-day "
                "schedule verification has been performed."
            )

            with st.expander("Planning evidence and source"):

                st.write("Planning mode:", planning_mode)
                st.write("Source:", planning_source)
                st.write("Agent execution:", agent_status)
                st.write("LLM status:", llm_status)

                for stop in evidence_stops:

                    st.write(
                        stop.get("place")
                        or stop.get("place_name")
                        or "Attraction",
                        "—",
                        stop.get("visit_section") or "General visit",
                    )

                    evidence_ids = (
                        stop.get("supporting_evidence_ids")
                        or stop.get("evidence_ids")
                        or []
                    )

                    approved_records = (
                        load_approved_evidence_for_display()
                    )

                    if not evidence_ids:
                        st.caption(
                            "No supporting evidence IDs supplied."
                        )

                    for evidence_id in evidence_ids:
                        record = approved_records.get(evidence_id)

                        if record is None:
                            st.warning(
                                "Evidence not confirmed approved: "
                                + str(evidence_id)
                            )
                            continue

                        title, value = (
                            describe_evidence_for_display(record)
                        )

                        st.markdown(f"**{title}**")
                        st.write(value)

                        source_name = record.get(
                            "source_name", "Recorded source"
                        )
                        source_url = record.get("source_url", "")
                        reviewed_on = record[
                            "approval"
                        ].get("reviewed_on", "")

                        if source_url.startswith(
                            ("https://", "http://")
                        ):
                            st.link_button(
                                "Official source: " + source_name,
                                source_url,
                            )
                        else:
                            st.caption(
                                "Source: " + source_name
                            )

                        st.caption(
                            "Evidence ID: "
                            + str(evidence_id)
                            + " | Reviewed: "
                            + str(reviewed_on)
                        )

                        with st.expander(
                            "Evidence review and history — "
                            + str(evidence_id)
                        ):
                            approval = record["approval"]

                            st.success(
                                "Approved evidence — "
                                "source checked"
                            )

                            st.caption(
                                "Approval date: "
                                + str(
                                    approval.get(
                                        "reviewed_on", "Not recorded"
                                    )
                                )
                            )

                            st.markdown(
                                "**Original evidence note "
                                "(historical)**"
                            )

                            st.write(
                                record.get("notes")
                                or "No original note recorded."
                            )

                            st.caption(
                                "This note reflects the original "
                                "evidence record. The approval ledger "
                                "records the current review decision."
                            )

                            review_notes = approval.get(
                                "review_notes", ""
                            )

                            if review_notes:
                                st.markdown(
                                    "**Approval review notes**"
                                )
                                st.write(review_notes)

        except Exception as exc:
            st.error(
                "The itinerary could not be generated. "
                "No unverified fallback itinerary was created."
            )
            with st.expander("Technical error"):
                st.code(
                    "Planning failed safely. Technical details are not "
                    "displayed because they may contain sensitive "
                    "information. Please review your selections "
                    "and try again."
                )

st.divider()

st.caption(
    "V3 MVP | V2 rankings remain unchanged | "
    "Optional agent runtime available | "
    "Live LLM requires OPENAI_API_KEY"
)
