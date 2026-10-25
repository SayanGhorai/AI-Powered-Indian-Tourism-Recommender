"""Presentation-only records for V2-ranked attraction galleries."""

import pandas as pd


def build_recommendation_gallery(
    recommendations,
    photo_metadata=None,
    evidence_stops=None,
):
    """Preserve V2 row order without promoting held attractions."""

    if not isinstance(recommendations, pd.DataFrame):
        raise TypeError("Recommendations must be a DataFrame.")

    required = {"place_id", "place", "city"}
    missing = required - set(recommendations.columns)

    if missing:
        raise ValueError(
            f"Missing recommendation fields: {sorted(missing)}"
        )

    photos = photo_metadata if photo_metadata is not None else {}
    stops = evidence_stops if evidence_stops is not None else []

    if not isinstance(photos, dict):
        raise TypeError("Photo metadata must be a dictionary.")

    if not isinstance(stops, (list, tuple)):
        raise TypeError("Evidence stops must be a list or tuple.")

    approved_ids = {
        str(stop["place_id"])
        for stop in stops
        if isinstance(stop, dict) and stop.get("place_id")
    }

    records = []

    for index, (_, row) in enumerate(
        recommendations.iterrows(), start=1
    ):
        place_id = str(row["place_id"]).strip()
        place = str(row["place"]).strip()
        city = str(row["city"]).strip()

        if not place_id or not place or not city:
            raise ValueError("Recommendation identity is incomplete.")

        rating = row.get("rating_mean")
        reviews = row.get("review_count")

        rating = (
            float(rating)
            if pd.notna(rating) and 0 <= float(rating) <= 5
            else None
        )

        reviews = (
            int(reviews)
            if pd.notna(reviews) and int(reviews) >= 0
            else None
        )

        records.append({
            "rank": index,
            "place_id": place_id,
            "place": place,
            "city": city,
            "rating_mean": rating,
            "review_count": reviews,
            "photo": photos.get(place_id),
            "is_itinerary_stop": place_id in approved_ids,
            "planning_label": (
                "Included in provisional itinerary"
                if place_id in approved_ids
                else "Not included in provisional itinerary"
            ),
        })

    return records
