"""Section-aware admission budgeting with approved evidence only."""

from decimal import Decimal, InvalidOperation

from src.v3_attraction_profiles import (
    get_approved_attraction_profile,
)


SECTION_FEE_CATEGORIES = {
    "museum_galleries": {
        "indian_national_museum_galleries",
    },
    "gardens": {
        "garden_daily_single_entry",
    },
}


def _calculate_section_profile_budget(
    profile,
    place_id,
    section,
    visitors,
):
    """Internal calculation; profile injection supports testing."""

    if section not in SECTION_FEE_CATEGORIES:
        raise ValueError(
            f"Unsupported visitor section: {section}"
        )

    if not isinstance(visitors, dict) or not visitors:
        raise ValueError(
            "Visitors must be a nonempty category-count dictionary."
        )

    allowed = SECTION_FEE_CATEGORIES[section]

    for category, count in visitors.items():
        if category not in allowed:
            raise ValueError(
                f"Category {category!r} is not valid "
                f"for section {section!r}."
            )

        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
        ):
            raise ValueError(
                f"Invalid visitor count: {category}"
            )

    total_visitors = sum(visitors.values())

    if total_visitors <= 0:
        raise ValueError(
            "At least one visitor is required."
        )

    result = {
        "place_id": place_id,
        "section": section,
        "status": "UNKNOWN",
        "currency": "INR",
        "total_visitors": total_visitors,
        "total_inr": None,
        "known_subtotal_inr": "0",
        "breakdown": [],
        "missing_categories": [],
        "supporting_evidence_ids": [],
        "limitations": [
            "Admission prices may change.",
            "Visitor eligibility must be verified.",
            "Additional ticket categories and "
            "exemptions may apply.",
            "Travel, food and optional charges "
            "are excluded.",
        ],
    }

    if profile is not None and (
        profile.get("place_id") != place_id
    ):
        raise ValueError(
            "Profile identity does not match place ID."
        )

    fees = (
        profile.get("entrance_fees", [])
        if profile is not None
        else []
    )

    subtotal = Decimal("0")

    for category, count in visitors.items():

        if count == 0:
            continue

        matching = [
            fee
            for fee in fees
            if fee.get("category") == category
        ]

        if len(matching) != 1:
            result["missing_categories"].append(
                category
            )
            result["breakdown"].append({
                "category": category,
                "visitors": count,
                "status": "UNKNOWN",
                "unit_price_inr": None,
                "subtotal_inr": None,
                "evidence_id": None,
            })
            continue

        fee = matching[0]

        try:
            amount = Decimal(
                str(fee["value"])
            )
        except (
            InvalidOperation,
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"Invalid approved fee: {category}"
            ) from exc

        if not amount.is_finite() or amount < 0:
            raise ValueError(
                f"Invalid fee amount: {category}"
            )

        line_total = amount * count
        subtotal += line_total

        result["breakdown"].append({
            "category": category,
            "visitors": count,
            "status": "PRICED",
            "unit_price_inr": str(amount),
            "subtotal_inr": str(line_total),
            "evidence_id": fee["evidence_id"],
        })

        result[
            "supporting_evidence_ids"
        ].append(fee["evidence_id"])

    result["known_subtotal_inr"] = str(
        subtotal
    )

    if result["missing_categories"]:
        result["status"] = "PARTIAL_UNKNOWN"
    else:
        result["status"] = "COMPLETE"
        result["total_inr"] = str(
            subtotal
        )

    return result


def calculate_section_admission_budget(
    place_id,
    section,
    visitors,
):
    """Public calculation using approved attraction profiles."""

    if not isinstance(place_id, str) or not place_id:
        raise ValueError("A valid place ID is required.")

    profile = get_approved_attraction_profile(
        place_id
    )

    return _calculate_section_profile_budget(
        profile=profile,
        place_id=place_id,
        section=section,
        visitors=visitors,
    )
