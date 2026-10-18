"""V3 Tourism Recommender — Streamlit MVP.

Uses the existing V2/V3 production backend.
Does not invent route times, arrival times or attraction facts.
"""

from datetime import date, datetime, time

import pandas as pd
import streamlit as st

from src.retrieval import PLACES_MASTER_PATH
from src.v3_planning_controller import run_v3_planning
from src.v3_itinerary_presenter import present_v3_itinerary
from src.v3_agent_runtime import run_v3_agent_runtime


st.set_page_config(
    page_title="Indian Tourism AI — V3",
    page_icon="🧭",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3 {
        letter-spacing: -0.025em;
    }
    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.07);
        border: 1px solid rgba(128, 128, 128, 0.16);
        border-radius: 14px;
        padding: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def build_v3_itinerary_pdf(
    presented,
    evidence_stops,
    planning_mode,
    planning_source,
    visit_date,
    start_time,
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
    paragraph(
        "Attractions needing checking: "
        + str(len(presented["held"]))
    )

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

    if presented["held"]:
        heading("Attractions needing checking")
        for held in presented["held"]:
            paragraph(
                str(held.get("place", "Attraction"))
                + " - "
                + str(
                    held.get("selection_status", "UNVERIFIED")
                )
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


st.title("🧭 Explore India, with evidence you can trust")
st.caption(
    "AI-powered attraction discovery and evidence-grounded "
    "itinerary planning | Indian Tourism Recommender V3"
)

st.info(
    "Plans are provisional. Verified admission evidence is used "
    "where available. Travel times, later arrival times, holiday "
    "exceptions and full-day feasibility are not guaranteed."
)


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
        "otherwise deterministic planning. API usage may incur charges."
    )


with st.form("v3_planning_form"):

    left, right = st.columns(2)

    with left:
        visit_date = st.date_input(
            "Visit date",
            value=date.today(),
            min_value=date.today(),
        )

    with right:
        start_time = st.time_input(
            "Requested starting time",
            value=time(11, 0),
            step=1800,
        )

    query = None
    selections = []
    visitors = {}

    if mode == "recommend":

        st.subheader("Find attractions")

        query = st.text_input(
            "What would you like to explore?",
            value="museums and historical attractions in Kolkata",
        )

        top_k = st.slider(
            "Maximum recommendations",
            min_value=1,
            max_value=5,
            value=5,
        )

        st.subheader("Visitor categories")

        st.caption(
            "Specify ticket categories explicitly. "
            "Admission categories are attraction-specific; "
            "unknown prices will not be assumed."
        )

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

    else:

        st.subheader("Choose attractions")

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

    submitted = st.form_submit_button(
        "Generate provisional itinerary",
        type="primary",
        use_container_width=True,
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
                "Checking recommendations and approved evidence..."
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
                            "city_canonical": "kolkata",
                            "selections": selections,
                            "visit_date": date_text,
                            "start_time": time_text,
                        }
                        agent_message = (
                            "Plan my explicitly selected attractions "
                            "in Kolkata using verified evidence."
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
                            city_canonical="kolkata",
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

            m1, m2, m3 = st.columns(3)

            m1.metric(
                "Stops planned",
                presented["stop_count"],
            )

            subtotal = presented[
                "known_admission_subtotal_inr"
            ]

            m2.metric(
                "Known admission cost",
                (
                    f"₹{subtotal}"
                    if presented["stop_count"] > 0
                    else "Not applicable"
                ),
            )

            m3.metric(
                "Needs checking",
                len(presented["held"]),
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
                    heading_col, time_col = st.columns([3, 2])

                    with heading_col:
                        st.caption(f"STOP {index:02d}")
                        st.markdown(f"#### {place}")

                        if section:
                            st.caption(
                                "Section: "
                                + str(section).replace("_", " ").title()
                            )

                    with time_col:
                        if index == 1 and arrival is not None:
                            st.markdown(
                                f"**Requested start: {arrival}**"
                            )
                        else:
                            st.caption("Arrival not scheduled")

                    fee_col, evidence_col = st.columns(2)

                    with fee_col:
                        if admission is None:
                            st.markdown(
                                "**Admission: Not verified**"
                            )
                        else:
                            st.markdown(
                                f"**Admission: ₹{admission}**"
                            )

                    with evidence_col:
                        st.caption(
                            f"Supporting evidence: {len(evidence_ids)} "
                            "record(s)"
                        )

            st.markdown("### Download your itinerary")

            pdf_data = build_v3_itinerary_pdf(
                presented=presented,
                evidence_stops=evidence_stops,
                planning_mode=planning_mode,
                planning_source=planning_source,
                visit_date=date_text,
                start_time=time_text,
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

                supporting_ids = (
                    planned_stop.get("supporting_evidence_ids")
                    or planned_stop.get("evidence_ids")
                    or []
                )

                matched_ids = {
                    evidence_to_place[evidence_id]
                    for evidence_id in supporting_ids
                    if evidence_id in evidence_to_place
                    and evidence_to_place[evidence_id]
                }

                explicit_id = planned_stop.get("place_id")

                if explicit_id:
                    matched_ids.add(str(explicit_id))

                if not matched_ids and not supporting_ids:
                    matched_ids = set(
                        locations_by_name.get(
                            stop_name.strip().casefold(), []
                        )
                    )

                if len(matched_ids) != 1:
                    missing_locations.append(stop_name)
                    continue

                place_id = next(iter(matched_ids))
                location = locations_by_id.get(place_id)

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

            if presented["held"]:
                with st.expander(
                    "Attractions needing checking "
                    f"({len(presented['held'])})"
                ):
                    for item in presented["held"]:
                        place = item.get("place", "Attraction")
                        reason = item.get(
                            "selection_status", "UNVERIFIED"
                        )
                        st.write(
                            f"**{place}** — "
                            + str(reason).replace("_", " ").title()
                        )

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
                    f"{type(exc).__name__}: {exc}"
                )

st.divider()

st.caption(
    "V3 MVP | V2 rankings remain unchanged | "
    "Optional agent runtime available | "
    "Live LLM requires OPENAI_API_KEY"
)
