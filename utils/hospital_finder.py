"""Helpers for user location, nearby hospital/clinic lookup, and folium map generation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import folium
import requests
import streamlit as st

try:
    from streamlit_geolocation import streamlit_geolocation
except Exception:
    streamlit_geolocation = None


OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


def get_user_location(
    default_lat: float = 28.6139,
    default_lon: float = 77.2090,
    key_prefix: str = "location",
) -> Tuple[float, float]:
    """
    Get user location from browser geolocation when available,
    with manual lat/lon fallback for desktop or blocked GPS.
    """
    st.caption("Location access: browser GPS (if available) or manual coordinates.")

    geo_lat: Optional[float] = None
    geo_lon: Optional[float] = None
    if streamlit_geolocation is not None:
        geo_data = streamlit_geolocation(key=f"{key_prefix}_geo")
        if geo_data:
            geo_lat = geo_data.get("latitude")
            geo_lon = geo_data.get("longitude")
    else:
        st.caption("`streamlit-geolocation` is not installed; using manual location input.")

    lat = float(geo_lat) if geo_lat is not None else float(default_lat)
    lon = float(geo_lon) if geo_lon is not None else float(default_lon)

    if geo_lat is not None and geo_lon is not None:
        st.success("Using browser geolocation.")
        return lat, lon

    st.info("GPS not available. Enter your coordinates manually.")
    manual_lat = st.number_input("Latitude", value=lat, format="%.6f", key=f"{key_prefix}_lat")
    manual_lon = st.number_input("Longitude", value=lon, format="%.6f", key=f"{key_prefix}_lon")
    return float(manual_lat), float(manual_lon)


def get_nearby_hospitals(
    lat: float,
    lon: float,
    radius_m: int = 5000,
    limit: int = 5,
    timeout: int = 25,
) -> List[Dict[str, float]]:
    """
    Query nearby hospitals/clinics using OSM Overpass API.

    Returns a list of dicts: {name, kind, lat, lon, distance_km}
    """
    query = f"""
[out:json][timeout:{timeout}];
(
  node["amenity"~"hospital|clinic"](around:{radius_m},{lat},{lon});
  way["amenity"~"hospital|clinic"](around:{radius_m},{lat},{lon});
  relation["amenity"~"hospital|clinic"](around:{radius_m},{lat},{lon});
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
    for element in data.get("elements", []):
        tags = element.get("tags", {})
        kind = tags.get("amenity", "hospital")
        name = tags.get("name", kind.title())

        if "lat" in element and "lon" in element:
            hlat, hlon = element["lat"], element["lon"]
        else:
            center = element.get("center")
            if not center:
                continue
            hlat, hlon = center.get("lat"), center.get("lon")
            if hlat is None or hlon is None:
                continue

        dist_km = _haversine_km(lat, lon, hlat, hlon)
        results.append(
            {
                "name": name,
                "kind": kind,
                "lat": hlat,
                "lon": hlon,
                "distance_km": dist_km,
            }
        )

    results.sort(key=lambda row: row["distance_km"])
    return results[: max(1, limit)]


def build_hospital_map(
    user_lat: float,
    user_lon: float,
    facilities: List[Dict[str, float]],
    zoom_start: int = 13,
) -> folium.Map:
    """Create a folium map with user and facility markers."""
    m = folium.Map(location=[user_lat, user_lon], zoom_start=zoom_start)
    folium.Marker(
        location=[user_lat, user_lon],
        popup="Your Location",
        icon=folium.Icon(color="blue", icon="user"),
    ).add_to(m)

    for facility in facilities:
        color = "red" if facility.get("kind") == "hospital" else "green"
        popup = (
            f'{facility["name"]} ({facility["kind"]}) - '
            f'{facility["distance_km"]:.2f} km'
        )
        folium.Marker(
            location=[facility["lat"], facility["lon"]],
            popup=popup,
            icon=folium.Icon(color=color, icon="plus-sign"),
        ).add_to(m)
    return m


def generate_hospital_map_html(
    user_lat: float,
    user_lon: float,
    facilities: List[Dict[str, float]],
    output_path: str = "hospital_map.html",
) -> str:
    """Generate and save a folium HTML map file with facility markers."""
    map_obj = build_hospital_map(user_lat, user_lon, facilities)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    map_obj.save(str(output))
    return str(output.resolve())
