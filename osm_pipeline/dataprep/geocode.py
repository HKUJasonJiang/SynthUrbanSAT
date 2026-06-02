"""Free geocoding via Nominatim (OpenStreetMap).

Public function:
    geocode(query: str) -> list[dict]  # each dict has lat, lon, display_name

Respect Nominatim usage policy: 1 request/sec, descriptive User-Agent.
"""
from __future__ import annotations

import time
from typing import List

import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "ProcedureOSM/0.1 (research; https://github.com/local)"
_LAST_CALL_TS = 0.0


def geocode(query: str, limit: int = 5, timeout: int = 15) -> List[dict]:
    """Return up to `limit` matches for `query`. Empty list on failure."""
    global _LAST_CALL_TS
    # rate-limit: ≥ 1 second between calls
    delay = 1.05 - (time.time() - _LAST_CALL_TS)
    if delay > 0:
        time.sleep(delay)
    try:
        r = requests.get(
            NOMINATIM_URL,
            params={"q": query, "format": "json", "limit": str(limit)},
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:  # noqa: BLE001
        print(f"[geocode] failed: {e}")
        return []
    finally:
        _LAST_CALL_TS = time.time()
    out = []
    for item in data:
        try:
            out.append({
                "lat": float(item["lat"]),
                "lon": float(item["lon"]),
                "display_name": item.get("display_name", ""),
                "type": item.get("type", ""),
            })
        except (KeyError, ValueError):
            continue
    return out
