"""Build itinerary-ready attraction profiles from approved V3 evidence.

Only records returned by the existing approval resolver are eligible.
Unknown information remains unknown rather than being inferred.
"""

from collections import defaultdict

from src.v3_evidence_resolver import (
    load_current_approved_evidence,
)


def build_approved_attraction_profiles():
    """Return approved attraction profiles indexed by place ID."""

    resolved = load_current_approved_evidence()

    if resolved.get("status") != "PASS":
        raise RuntimeError("Evidence resolver did not pass.")

    profiles = {}

    for item in resolved["approved_records"]:
        evidence = item["evidence"]
        approval = item["approval"]

        if approval["decision"] != "approved":
            raise ValueError("Unapproved evidence in resolver output.")

        if approval["source_checked"] is not True:
            raise ValueError("Evidence source was not checked.")

        place_id = evidence["place_id"]

        if place_id not in profiles:
            profiles[place_id] = {
                "place_id": place_id,
                "city_canonical": evidence["city_canonical"],
                "entrance_fees": [],
                "opening_hours": [],
                "weekly_off": [],
                "visit_duration": [],
                "coordinates": {},
                "evidence_ids": [],
                "sources": [],
            }

        profile = profiles[place_id]

        if profile["city_canonical"] != evidence["city_canonical"]:
            raise ValueError(
                f"Conflicting city identity for {place_id}"
            )

        evidence_type = evidence["evidence_type"]

        detail = {
            "category": evidence["category"],
            "value": evidence["value"],
            "unit": evidence["unit"],
            "evidence_id": evidence["evidence_id"],
        }

        if evidence_type == "entrance_fee":
            detail["amount_inr"] = float(evidence["value"])
            profile["entrance_fees"].append(detail)

        elif evidence_type == "opening_hours":
            profile["opening_hours"].append(detail)

        elif evidence_type == "weekly_off":
            profile["weekly_off"].append(detail)

        elif evidence_type == "visit_duration":
            detail["hours"] = float(evidence["value"])
            profile["visit_duration"].append(detail)

        elif evidence_type in (
            "attraction_latitude",
            "attraction_longitude",
        ):
            profile["coordinates"][evidence_type] = detail

        else:
            raise ValueError(
                f"Unsupported approved evidence type: {evidence_type}"
            )

        profile["evidence_ids"].append(evidence["evidence_id"])

        source = {
            "name": evidence["source_name"],
            "url": evidence["source_url"],
            "observed_on": evidence["observed_on"],
        }

        if source not in profile["sources"]:
            profile["sources"].append(source)

    for profile in profiles.values():
        profile["approved_evidence_count"] = len(
            profile["evidence_ids"]
        )

        profile["has_verified_coordinates"] = all(
            key in profile["coordinates"]
            for key in (
                "attraction_latitude",
                "attraction_longitude",
            )
        )

        profile["has_approved_opening_hours"] = bool(
            profile["opening_hours"]
        )

        profile["has_approved_entrance_fees"] = bool(
            profile["entrance_fees"]
        )

    return profiles


def get_approved_attraction_profile(place_id):
    """Return an approved profile or None when unavailable."""

    return build_approved_attraction_profiles().get(place_id)
