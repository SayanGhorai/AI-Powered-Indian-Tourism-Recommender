"""Grounded user-facing presentation of V3 itinerary drafts.

Only describes information supplied by the planning controller.
Does not generate attraction facts, routes, or arrival times.
"""

from decimal import Decimal, InvalidOperation


def _format_inr(value):
    if value is None:
        return "not verified"

    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError("Invalid admission amount.")

    if not amount.is_finite() or amount < 0:
        raise ValueError("Invalid admission amount.")

    if amount == amount.to_integral_value():
        return f"₹{amount:,.0f}"

    return f"₹{amount:,.2f}"


def present_v3_itinerary(controller_result):
    """Produce a readable, conservative itinerary response."""

    if not isinstance(controller_result, dict):
        raise TypeError(
            "Controller result must be a dictionary."
        )

    mode = controller_result.get("mode")

    if mode not in ("recommend", "user_selected"):
        raise ValueError("Unsupported planning mode.")

    draft = controller_result.get("draft")

    if not isinstance(draft, dict):
        raise ValueError("Missing itinerary draft.")

    stops = draft.get("stops", [])

    if not isinstance(stops, list):
        raise TypeError("Draft stops must be a list.")

    if draft.get("route_optimized") is not False:
        raise ValueError(
            "Cannot present an unverified routing claim."
        )

    if draft.get("full_day_feasibility_verified") is not False:
        raise ValueError(
            "Cannot present an unverified full-day schedule."
        )

    city = controller_result.get("city")

    if not isinstance(city, str) or not city.strip():
        raise ValueError("Missing canonical city.")

    city_display = city.replace("_", " ").title()

    if mode == "recommend":
        heading = (
            f"Recommended itinerary draft — {city_display}"
        )
        source_note = (
            "Attractions originate from V2-ranked recommendations."
        )
        held = draft.get("held_recommendations", [])
    else:
        heading = (
            f"Your selected itinerary draft — {city_display}"
        )
        source_note = (
            "Attractions were explicitly selected by the user; "
            "they are not claimed to be V2 recommendations."
        )
        held = draft.get("held_selections", [])

    if not isinstance(held, list):
        raise TypeError("Held attractions must be a list.")

    lines = [heading, "", source_note, ""]

    if not stops:
        lines.append(
            "No attraction currently qualifies as a "
            "conditional itinerary stop."
        )
    else:
        lines.append("Provisional stops:")

    stop_summaries = []

    for index, stop in enumerate(stops, start=1):
        if not isinstance(stop, dict):
            raise TypeError("Invalid stop record.")

        name = str(stop["place"])
        section = stop.get("visit_section")
        arrival = stop.get("proposed_arrival_time")
        admission = _format_inr(
            stop.get("admission_total_inr")
        )

        display_name = name

        if section:
            display_name += (
                " — " + str(section).replace("_", " ")
            )

        if index == 1 and arrival is not None:
            arrival_text = (
                f"requested start {arrival}"
            )
        else:
            arrival_text = "arrival not scheduled"

        text = (
            f"{index}. {display_name} "
            f"| {arrival_text} "
            f"| admission: {admission}"
        )

        lines.append(text)

        stop_summaries.append({
            "place_id": stop["place_id"],
            "place": name,
            "section": section,
            "arrival": arrival,
            "admission_display": admission,
            "description": text,
        })

    lines.append("")

    subtotal = draft.get(
        "known_admission_subtotal_inr"
    )

    if stops:
        lines.append(
            "Known admission subtotal: "
            + _format_inr(subtotal)
        )
        lines.append(
            "This is not the total trip cost."
        )
    else:
        lines.append(
            "No selected-stop admission subtotal "
            "is applicable."
        )

    all_costs_known = draft.get(
        "all_selected_admission_costs_known"
    )

    if stops and all_costs_known is not True:
        lines.append(
            "Some selected admission costs remain unknown."
        )

    held_summaries = []

    if held:
        lines.extend([
            "",
            "Attractions not included:",
        ])

        for item in held:
            if not isinstance(item, dict):
                raise TypeError("Invalid held attraction.")

            name = str(item["place"])

            reason = str(
                item.get(
                    "selection_status",
                    "UNVERIFIED",
                )
            )

            text = f"- {name}: {reason}"

            lines.append(text)

            held_summaries.append({
                "place_id": item["place_id"],
                "place": name,
                "reason": reason,
                "description": text,
            })

    limitations = [
        "This is a provisional plan, not a verified "
        "full-day schedule.",
        "Travel times and route optimization "
        "have not been verified.",
        "Only the first stop may have a requested "
        "start time; later arrivals are unscheduled.",
        "Later-stop opening-hour feasibility "
        "must be checked at actual arrival times.",
        "Admission information is based on "
        "currently approved evidence, not live "
        "ticket availability.",
    ]

    lines.extend([
        "",
        "Planning limitations:",
    ])

    for limitation in limitations:
        lines.append("- " + limitation)

    return {
        "mode": mode,
        "heading": heading,
        "city": city,
        "source_note": source_note,
        "stop_count": len(stops),
        "stops": stop_summaries,
        "held": held_summaries,
        "known_admission_subtotal_inr": subtotal,
        "all_selected_admission_costs_known": (
            all_costs_known
        ),
        "limitations": limitations,
        "answer": "\n".join(lines),
        "grounding_status": "DETERMINISTIC_V3_OUTPUT",
        "llm_generated": False,
        "route_optimized": False,
        "full_day_feasibility_verified": False,
    }
