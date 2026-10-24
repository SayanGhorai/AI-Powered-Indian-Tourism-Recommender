"""Strict attraction-level map identity resolution.

Never assigns map coordinates from attraction names or city centers.
Requires a catalog place ID and consistent evidence identity.
"""


def resolve_map_identity(
    planned_stop,
    locations_by_id,
    evidence_to_place,
):
    """Return a confirmed location or None when identity is unsafe."""

    if not isinstance(planned_stop, dict):
        raise TypeError("Planned stop must be a dictionary.")

    if not isinstance(locations_by_id, dict):
        raise TypeError("Location lookup must be a dictionary.")

    if not isinstance(evidence_to_place, dict):
        raise TypeError("Evidence lookup must be a dictionary.")

    explicit_id = planned_stop.get("place_id")

    if not isinstance(explicit_id, str) or not explicit_id.strip():
        return None

    explicit_id = explicit_id.strip()

    supporting_ids = (
        planned_stop.get("supporting_evidence_ids")
        or planned_stop.get("evidence_ids")
        or []
    )

    if not isinstance(supporting_ids, (list, tuple)):
        raise TypeError("Evidence IDs must be a list or tuple.")

    for evidence_id in supporting_ids:
        if not isinstance(evidence_id, str) or not evidence_id:
            raise ValueError("Invalid supporting evidence ID.")

        linked_place = evidence_to_place.get(evidence_id)

        if linked_place is None:
            return None

        if linked_place != explicit_id:
            return None

    location = locations_by_id.get(explicit_id)

    if location is None:
        return None

    if location.get("place_id") != explicit_id:
        return None

    return {
        "place_id": explicit_id,
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "place_name": location.get("place_name"),
        "identity_status": "STRICT_ID_MATCH",
    }
