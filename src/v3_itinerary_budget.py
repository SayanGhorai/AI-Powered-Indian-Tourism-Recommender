"""Conservative admission-budget calculations for V3 itineraries.

Only approved attraction profiles may supply admission prices.
Unknown prices are never interpreted as zero.
"""

from decimal import Decimal, InvalidOperation

from src.v3_attraction_profiles import (
    get_approved_attraction_profile,
)


VISITOR_CATEGORIES = (
    "adult_admission",
    "below_18_admission",
    "foreign_national_admission",
)


def calculate_admission_budget(place_id, visitors):
    """Calculate admission cost for explicit visitor categories.

    Each visitor must be assigned to exactly one category.
    The caller is responsible for choosing the applicable
    published category; overlapping eligibility is not inferred.
    """

    if not isinstance(place_id, str) or not place_id.strip():
        raise ValueError("A valid place ID is required.")

    if not isinstance(visitors, dict):
        raise TypeError("Visitors must be a category-count dictionary.")

    if not visitors:
        raise ValueError("At least one visitor category is required.")

    unknown_categories = set(visitors) - set(VISITOR_CATEGORIES)

    if unknown_categories:
        raise ValueError(
            f"Unsupported visitor categories: "
            f"{sorted(unknown_categories)}"
        )

    for category, count in visitors.items():
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
        ):
            raise ValueError(
                f"Invalid visitor count for {category}"
            )

    if sum(visitors.values()) == 0:
        raise ValueError("At least one visitor is required.")

    profile = get_approved_attraction_profile(place_id)

    result = {
        "place_id": place_id,
        "status": "UNKNOWN",
        "currency": "INR",
        "total_visitors": sum(visitors.values()),
        "total_inr": None,
        "known_subtotal_inr": "0",
        "breakdown": [],
        "missing_categories": [],
        "supporting_evidence_ids": [],
        "limitations": [
            "Admission only; travel, food and photography excluded.",
            "Published admission prices may change.",
            "Visitor eligibility must be selected correctly.",
        ],
    }

    if profile is None:
        result["missing_categories"] = [
            category
            for category, count in visitors.items()
            if count > 0
        ]
        return result

    fees_by_category = {}

    for fee in profile["entrance_fees"]:
        category = fee["category"]

        if category not in VISITOR_CATEGORIES:
            continue

        try:
            amount = Decimal(str(fee["value"]))
        except InvalidOperation as exc:
            raise ValueError(
                f"Invalid approved fee for {category}"
            ) from exc

        if not amount.is_finite() or amount < 0:
            raise ValueError(
                f"Invalid approved fee amount for {category}"
            )

        fees_by_category.setdefault(category, []).append(
            (amount, fee["evidence_id"])
        )

    subtotal = Decimal("0")

    for category, count in visitors.items():
        if count == 0:
            continue

        matching = fees_by_category.get(category, [])

        # Multiple price records are ambiguous even if identical.
        # Do not silently choose one.
        if len(matching) != 1:
            result["missing_categories"].append(category)
            result["breakdown"].append({
                "category": category,
                "visitors": count,
                "status": "UNKNOWN",
                "unit_price_inr": None,
                "subtotal_inr": None,
                "evidence_id": None,
            })
            continue

        unit_price, evidence_id = matching[0]
        line_total = unit_price * count
        subtotal += line_total

        result["breakdown"].append({
            "category": category,
            "visitors": count,
            "status": "PRICED",
            "unit_price_inr": str(unit_price),
            "subtotal_inr": str(line_total),
            "evidence_id": evidence_id,
        })

        result["supporting_evidence_ids"].append(evidence_id)

    result["known_subtotal_inr"] = str(subtotal)

    if result["missing_categories"]:
        result["status"] = "PARTIAL_UNKNOWN"
        result["total_inr"] = None
    else:
        result["status"] = "COMPLETE"
        result["total_inr"] = str(subtotal)

    return result
