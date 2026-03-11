import os
import sys
import time

import cv2
import joblib
import numpy as np
import streamlit as st
from scipy.signal import find_peaks
from streamlit_folium import st_folium
import folium

# Ensure utils is importable when running from project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.feature_extractor import (
    append_signal_sample,
    center_roi,
    extract_features,
    init_signal_buffer,
    mean_green_from_roi,
)
from utils.hospital_finder import get_nearby_hospitals
from utils.signal_processor import apply_filter


APP_TITLE = "Retina-Vitals"
MODEL_PATH = os.path.join("models", "bp_predictor.pkl")
VITAL_THRESHOLDS = {
    "bp": {
        "warning": {"sbp_high": 130, "dbp_high": 85},
        "critical": {"sbp_high": 140, "dbp_high": 90},
    },
    "hr": {
        "warning": {"high": 100, "low": 55},
        "critical": {"high": 120, "low": 45},
    },
    "hb": {
        "warning": {"low": 10.0},
        "critical": {"low": 8.0},
    },
}


def try_load_model(path: str):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def estimate_hb(mean_val: float, std_dev: float) -> float:
    """Return a demo Hb estimate derived from camera signal statistics."""
    hb = 8.0 + (mean_val / 255.0) * 8.0 + (std_dev * 0.2)
    return float(np.clip(hb, 7.0, 18.0))


def triage_patient(predictions: dict) -> str:
    """Return triage label from vital predictions: NORMAL, WARNING, EMERGENCY."""
    sbp = float(predictions.get("sbp", 0.0))
    dbp = float(predictions.get("dbp", 0.0))
    hr = float(predictions.get("hr", 0.0))
    hb = float(predictions.get("hb", 0.0))

    bp_critical = (
        sbp > VITAL_THRESHOLDS["bp"]["critical"]["sbp_high"]
        or dbp > VITAL_THRESHOLDS["bp"]["critical"]["dbp_high"]
    )
    hr_critical = (
        hr > VITAL_THRESHOLDS["hr"]["critical"]["high"]
        or hr < VITAL_THRESHOLDS["hr"]["critical"]["low"]
    )
    hb_critical = hb < VITAL_THRESHOLDS["hb"]["critical"]["low"]
    if bp_critical or hr_critical or hb_critical:
        return "EMERGENCY"

    bp_warning = (
        sbp > VITAL_THRESHOLDS["bp"]["warning"]["sbp_high"]
        or dbp > VITAL_THRESHOLDS["bp"]["warning"]["dbp_high"]
    )
    hr_warning = (
        hr > VITAL_THRESHOLDS["hr"]["warning"]["high"]
        or hr < VITAL_THRESHOLDS["hr"]["warning"]["low"]
    )
    hb_warning = hb < VITAL_THRESHOLDS["hb"]["warning"]["low"]
    if bp_warning or hr_warning or hb_warning:
        return "WARNING"

    return "NORMAL"


def render_hospital_map(lat: float, lon: float, radius_m: int):
    m = folium.Map(location=[lat, lon], zoom_start=13)
    folium.Marker(
        location=[lat, lon],
        popup="Your Location",
        icon=folium.Icon(color="blue", icon="user"),
    ).add_to(m)
    hospitals = get_nearby_hospitals(lat, lon, radius_m=radius_m)
    if not hospitals:
        st.info("No hospitals found (or Overpass API unavailable).")
    for hospital in hospitals:
        folium.Marker(
            location=[hospital["lat"], hospital["lon"]],
            popup=f'{hospital["name"]} ({hospital["distance_km"]:.1f} km)',
            icon=folium.Icon(color="red", icon="plus-sign"),
        ).add_to(m)
    st_folium(m, height=350)


def render_results_card(sbp: float, dbp: float, hr: float, hb: float, triage: str):
    emergency = triage == "EMERGENCY"
    accent = "#b91c1c" if emergency else "#166534"
    bg = "#fee2e2" if emergency else "#dcfce7"
    border = "#ef4444" if emergency else "#22c55e"
    st.markdown(
        f"""
        <div style="border:2px solid {border}; background:{bg}; border-radius:14px; padding:18px;">
            <div style="font-size:30px; font-weight:800; color:{accent};">Final Output</div>
            <div style="font-size:44px; font-weight:800; color:{accent};">BP: {sbp:.0f}/{dbp:.0f}</div>
            <div style="font-size:44px; font-weight:800; color:{accent};">HR: {hr:.0f} bpm</div>
            <div style="font-size:44px; font-weight:800; color:{accent};">Hb: {hb:.1f} g/dL</div>
            <div style="font-size:24px; font-weight:700; color:{accent}; margin-top:8px;">Triage: {triage}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.write("Contactless pulse estimation demo using the green channel.")

    with st.expander("Instructions", expanded=True):
        st.markdown(
            """
- Ensure good lighting and keep your face/hand steady in the camera view.
- Toggle **Run Camera** to begin.
- If the model file is missing, prediction will be disabled but the signal graph will still render.
            """
        )

    model = try_load_model(MODEL_PATH)
    if model is None:
        st.warning("Model not found or failed to load. Place a trained model at `models/bp_predictor.pkl`.")

    with st.sidebar:
        st.header("Location Settings")
        default_lat = 28.6139
        default_lon = 77.2090
        loc_lat = st.number_input("Latitude", value=default_lat, format="%.6f")
        loc_lon = st.number_input("Longitude", value=default_lon, format="%.6f")
        radius_m = st.slider("Search Radius (m)", min_value=1000, max_value=10000, value=5000, step=500)

    run = st.checkbox("Run Camera", key="run_camera")
    frame_window = st.image([])
    st.caption("Live PPG Signal (Green intensity over time)")
    chart_placeholder = st.empty()
    status_placeholder = st.empty()
    result_placeholder = st.empty()
    advice_placeholder = st.empty()
    map_placeholder = st.empty()

    # Keep the latest 30 seconds of signal samples for real-time plotting.
    sampling_rate_hz = 30
    signal_buffer = init_signal_buffer(fs=sampling_rate_hz, duration_sec=30)
    min_processing_samples = 200

    if run:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            st.error("Unable to access webcam (index 0).")
            return

        fs = sampling_rate_hz  # assumed frame rate (Hz)

        try:
            while True:
                if not st.session_state.get("run_camera", False):
                    break
                ret, frame = camera.read()
                if not ret:
                    status_placeholder.error("Camera frame not available.")
                    break

                # Center ROI (100x100) and mean green-channel extraction.
                x0, y0, x1, y1 = center_roi(frame, box_size=100)
                g_val = mean_green_from_roi(frame, (x0, y0, x1, y1))
                append_signal_sample(signal_buffer, g_val)
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

                # Render frame
                frame_window.image(frame, channels="BGR")
                chart_placeholder.line_chart(np.array(signal_buffer, dtype=np.float32))

                # Process once we have a stable minimum number of samples.
                if len(signal_buffer) >= min_processing_samples:
                    raw_sig = np.array(signal_buffer, dtype=np.float32)
                    clean_sig = apply_filter(raw_sig, 0.5, 4.0, fs)

                    peaks, _ = find_peaks(clean_sig, distance=20)
                    if len(peaks) > 5:
                        features = extract_features(clean_sig, fs)
                        hr = features["hr_bpm"]
                        std_dev = features["std_dev"]
                        mean_val = features["mean_val"]

                        status_placeholder.info(
                            f"HR: {hr:.1f} bpm | STD: {std_dev:.2f} | Mean: {mean_val:.2f}"
                        )

                        if model is not None:
                            prediction = model.predict([[hr, std_dev, mean_val]])
                            sbp = float(prediction[0][0])
                            dbp = float(prediction[0][1])
                            hb = estimate_hb(mean_val, std_dev)
                            triage = triage_patient({"sbp": sbp, "dbp": dbp, "hr": hr, "hb": hb})

                            with result_placeholder.container():
                                render_results_card(sbp, dbp, hr, hb, triage)

                            with advice_placeholder.container():
                                if triage == "EMERGENCY":
                                    st.error("Please visit the nearest facility immediately.")
                                elif triage == "WARNING":
                                    st.warning("Health risk detected. Rest and consult a clinician soon.")
                                else:
                                    st.success("Vitals are in a relatively stable range.")

                            if triage == "EMERGENCY":
                                with map_placeholder.container():
                                    render_hospital_map(loc_lat, loc_lon, radius_m)
                            else:
                                map_placeholder.empty()
                        else:
                            st.info("Prediction disabled (no model loaded).")

                    else:
                        status_placeholder.info("Waiting for stable peaks...")

                # Allow Streamlit to process UI events
                time.sleep(0.01)
        finally:
            camera.release()
    else:
        st.info("Camera is off. Toggle 'Run Camera' to start.")


if __name__ == "__main__":
    main()
