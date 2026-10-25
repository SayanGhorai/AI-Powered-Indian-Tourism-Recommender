"""Best-effort Wikimedia Commons photos for attraction previews.

This module provides presentation metadata only.
It never approves itinerary evidence or geographic identity.
"""

import html
import json
import re
import urllib.parse
import urllib.request
from urllib.parse import urlsplit
from functools import lru_cache

COMMONS_API = "https://commons.wikimedia.org/w/api.php"

USER_AGENT = (
    "IndianTourismRecommenderV3/0.1 "
    "(educational portfolio attraction previews)"
)

STOPWORDS = {
    "the", "of", "in", "at", "and", "a", "an",
    "museum", "fort", "palace", "temple",
}



def _safe_https_url(value, allowed_hosts):
    """Require HTTPS and an exact approved hostname."""
    if not isinstance(value, str):
        return False

    try:
        parsed = urlsplit(value)
    except ValueError:
        return False

    return (
        parsed.scheme == "https"
        and parsed.hostname in allowed_hosts
        and parsed.username is None
        and parsed.password is None
        and parsed.port in (None, 443)
        and bool(parsed.path)
    )



def _tokens(value):
    return {
        token
        for token in re.findall(
            r"[a-z0-9]+",
            str(value).casefold(),
        )
        if token not in STOPWORDS
    }


def _plain_text(value):
    text = re.sub(r"<[^>]*>", " ", str(value or ""))
    return " ".join(html.unescape(text).split())


def _request_commons(query, timeout=8):
    parameters = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": 5,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size|mime",
        "iiurlwidth": 400,
        "format": "json",
    }

    url = COMMONS_API + "?" + urllib.parse.urlencode(parameters)

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=timeout,
    ) as response:
        if response.status != 200:
            raise RuntimeError("Commons API returned an error.")
        return json.load(response)


@lru_cache(maxsize=256)
def discover_attraction_photo(place_id, place_name, city):
    """Return an automatically matched photo or None."""

    if not all(
        isinstance(value, str) and value.strip()
        for value in (place_id, place_name, city)
    ):
        return None

    query = f"{place_name.strip()} {city.strip()}"

    try:
        response = _request_commons(query)
    except (
        OSError,
        ValueError,
        RuntimeError,
    ):
        return None

    pages = response.get("query", {}).get("pages", {})

    if not isinstance(pages, dict):
        return None

    name_tokens = _tokens(place_name)

    if not name_tokens:
        return None

    candidates = []

    for page in pages.values():
        title = str(page.get("title", ""))

        if not title.startswith("File:"):
            continue

        info_list = page.get("imageinfo") or []

        if not info_list:
            continue

        info = info_list[0]

        if info.get("mime") not in ("image/jpeg", "image/png"):
            continue

        thumbnail = info.get("thumburl")
        source_page = info.get("descriptionurl")

        if not thumbnail or not source_page:
            continue

        title_tokens = _tokens(title)

        overlap = len(name_tokens & title_tokens)

        if overlap < max(1, len(name_tokens) // 2):
            continue

        metadata = info.get("extmetadata") or {}

        def field(key):
            return _plain_text(
                metadata.get(key, {}).get("value", "")
            )

        license_name = field("LicenseShortName")
        license_url = _plain_text(
            metadata.get("LicenseUrl", {}).get("value", "")
        )

        creator = field("Artist") or field("Credit")

        if not license_name or not creator:
            continue

        if not (
            _safe_https_url(
                thumbnail,
                {"thumb.wikimedia.org", "upload.wikimedia.org"},
            )
            and _safe_https_url(
                source_page,
                {"commons.wikimedia.org"},
            )
        ):
            continue

        if license_url and not _safe_https_url(
            license_url,
            {"creativecommons.org", "www.creativecommons.org"},
        ):
            license_url = ""

        candidates.append((
            overlap,
            {
                "place_id": place_id,
                "photo_url": thumbnail,
                "source_page": source_page,
                "creator": creator[:240],
                "license": license_name[:100],
                "license_url": license_url,
                "verification_status": "AUTOMATIC_COMMONS_MATCH",
                "match_label": "Automatically matched photo",
            },
        ))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)

    return candidates[0][1]
