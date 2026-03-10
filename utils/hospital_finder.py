"""Hospital finder using OpenStreetMap Overpass API."""

from typing import List, Dict
import math
import requests


OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def get_nearby_hospitals(
    lat: float,
    lon: float,
    radius_m: int = 5000,
    limit: int = 10,
    timeout: int = 25,
) -> List[Dict[str, float]]:
    """
    Query the Overpass API for nearby hospitals.

    Args:
        lat: Latitude.
        lon: Longitude.

    Returns:
        List of dicts: {name, lat, lon, distance_km}
    """
    query = f"""
[out:json][timeout:{timeout}];
(
  node["amenity"="hospital"](around:{radius_m},{lat},{lon});
  way["amenity"="hospital"](around:{radius_m},{lat},{lon});
  relation["amenity"="hospital"](around:{radius_m},{lat},{lon});
);
out center;
"""
    headers = {"User-Agent": "Retina-Vitals/1.0 (hackathon demo)"}
    try:
        resp = requests.post(OVERPASS_URL, data={"data": query}, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    results: List[Dict[str, float]] = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name", "Hospital")
        if "lat" in el and "lon" in el:
            hlat, hlon = el["lat"], el["lon"]
        else:
            center = el.get("center")
            if not center:
                continue
            hlat, hlon = center.get("lat"), center.get("lon")
            if hlat is None or hlon is None:
                continue

        dist_km = _haversine_km(lat, lon, hlat, hlon)
        results.append({"name": name, "lat": hlat, "lon": hlon, "distance_km": dist_km})

    results.sort(key=lambda r: r["distance_km"])
    return results[: max(1, limit)]
