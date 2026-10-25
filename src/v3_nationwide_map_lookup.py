"""Bounded Wikidata coordinate discovery for attraction previews.

Results are unverified discovery candidates, never itinerary evidence.
"""

import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
import time

API = "https://www.wikidata.org/w/api.php"

USER_AGENT = (
    "IndianTourismRecommenderV3/0.1 "
    "(educational portfolio geographic previews)"
)


def _normalize(value):
    return " ".join(
        re.findall(r"[a-z0-9]+", str(value).casefold())
    )



_SUCCESS_CACHE = OrderedDict()
_CACHE_TTL_SECONDS = 86400
_CACHE_MAX_ENTRIES = 512


def clear_coordinate_cache():
    """Clear successful discovery results."""
    _SUCCESS_CACHE.clear()


def _cache_key(place_id, place_name, city):
    return (
        str(place_id).strip(),
        str(place_name).strip().casefold(),
        str(city).strip().casefold(),
    )


def _cached_success(key):
    entry = _SUCCESS_CACHE.get(key)

    if entry is None:
        return None

    expires_at, result = entry

    if time.monotonic() >= expires_at:
        del _SUCCESS_CACHE[key]
        return None

    _SUCCESS_CACHE.move_to_end(key)
    return dict(result)


def _store_success(key, result):
    _SUCCESS_CACHE[key] = (
        time.monotonic() + _CACHE_TTL_SECONDS,
        dict(result),
    )

    _SUCCESS_CACHE.move_to_end(key)

    while len(_SUCCESS_CACHE) > _CACHE_MAX_ENTRIES:
        _SUCCESS_CACHE.popitem(last=False)


class RequestBudget:
    """Count actual HTTP attempts within one lookup batch."""

    def __init__(self, maximum=6):
        if (
            not isinstance(maximum, int)
            or isinstance(maximum, bool)
            or not 0 <= maximum <= 10
        ):
            raise ValueError("Invalid HTTP request budget.")

        self.maximum = maximum
        self.used = 0
        self.rate_limited = False

    def consume(self):
        if self.rate_limited or self.used >= self.maximum:
            return False

        self.used += 1
        return True



def _request(parameters, timeout=6, budget=None):
    if budget is not None and not budget.consume():
        raise RuntimeError("HTTP request budget exhausted.")

    url = API + "?" + urllib.parse.urlencode(parameters)

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
        ) as response:
            if response.status != 200:
                raise RuntimeError("Wikidata API error.")
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 429 and budget is not None:
            budget.rate_limited = True
        raise


def _valid_coordinates(latitude, longitude):
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        return False

    return (
        math.isfinite(lat)
        and math.isfinite(lon)
        and -90 <= lat <= 90
        and -180 <= lon <= 180
    )


def discover_candidate_coordinate(place_id, place_name, city, budget=None):
    """Find one conservative attraction-level Wikidata candidate."""

    if not all(
        isinstance(value, str) and value.strip()
        for value in (place_id, place_name, city)
    ):
        return None

    key = _cache_key(place_id, place_name, city)
    cached = _cached_success(key)

    if cached is not None:
        return cached

    try:
        search = _request({
            "action": "wbsearchentities",
            "search": place_name,
            "language": "en",
            "limit": 3,
            "format": "json",
        }, budget=budget)

        matches = search.get("search", [])

        exact = []

        for item in matches:
            if (
                _normalize(item.get("label", ""))
                != _normalize(place_name)
            ):
                continue

            description = (
                " " + _normalize(
                    item.get("description", "")
                ) + " "
            )

            city_phrase = " " + _normalize(city) + " "

            if city_phrase in description:
                exact.append(item)

        if len(exact) != 1:
            return None

        candidate = exact[0]
        qid = candidate.get("id", "")

        if not re.fullmatch(r"Q[1-9][0-9]*", qid):
            return None

        details = _request({
            "action": "wbgetentities",
            "ids": qid,
            "props": "claims",
            "format": "json",
        }, budget=budget)

        entity = details.get("entities", {}).get(qid, {})

        statements = entity.get(
            "claims", {}
        ).get("P625", [])

        coordinates = []

        for statement in statements:
            value = (
                statement.get("mainsnak", {})
                .get("datavalue", {})
                .get("value", {})
            )

            if not isinstance(value, dict):
                continue

            lat = value.get("latitude")
            lon = value.get("longitude")

            if _valid_coordinates(lat, lon):
                coordinates.append((float(lat), float(lon)))

        if len(coordinates) != 1:
            return None

        result = {
            "place_id": place_id,
            "place": place_name,
            "city": city,
            "latitude": coordinates[0][0],
            "longitude": coordinates[0][1],
            "source": "Wikidata candidate " + qid,
            "location_status": "WIKIDATA_CANDIDATE",
            "wikidata_id": qid,
        }

        _store_success(key, result)
        return dict(result)

    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
    ):
        return None


def discover_recommendation_coordinates(
    gallery_records,
    max_lookups=3,
    max_http_requests=6,
):
    """Best-effort lookup for a bounded number of recommendations.

    The limit controls attraction lookup attempts, not raw HTTP calls.
    Results are unverified discovery candidates.
    """

    if not isinstance(gallery_records, (list, tuple)):
        raise TypeError("Gallery records must be a list or tuple.")

    if (
        not isinstance(max_lookups, int)
        or isinstance(max_lookups, bool)
        or not 0 <= max_lookups <= 5
    ):
        raise ValueError("max_lookups must be an integer from 0 to 5.")

    budget = RequestBudget(maximum=max_http_requests)

    found = []
    missing = []
    attempts = 0

    for item in gallery_records:
        if not isinstance(item, dict):
            raise TypeError("Gallery item must be a dictionary.")

        place_id = str(item.get("place_id", "")).strip()
        place = str(item.get("place", "")).strip()
        city = str(item.get("city", "")).strip()

        if not place_id or not place or not city:
            raise ValueError("Incomplete attraction identity.")

        if place.casefold() == city.casefold():
            missing.append(place_id)
            continue

        cached = _cached_success(
            _cache_key(place_id, place, city)
        )

        if cached is not None:
            found.append({
                **cached,
                "rank": item.get("rank"),
            })
            continue

        if (
            attempts >= max_lookups
            or budget.rate_limited
            or budget.used >= budget.maximum
        ):
            missing.append(place_id)
            continue

        attempts += 1

        result = discover_candidate_coordinate(
            place_id,
            place,
            city,
            budget=budget,
        )

        if result is None:
            missing.append(place_id)
            continue

        found.append({
            **result,
            "rank": item.get("rank"),
        })

    return {
        "markers": found,
        "missing_place_ids": missing,
        "lookup_attempts": attempts,
        "lookup_limit": max_lookups,
        "http_requests_used": budget.used,
        "http_request_limit": budget.maximum,
        "rate_limited": budget.rate_limited,
    }
