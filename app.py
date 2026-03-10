import os
import sys
import time
from collections import deque

import cv2
import joblib
import numpy as np
import streamlit as st
from scipy.signal import find_peaks
from streamlit_folium import st_folium
import folium

# Ensure utils is importable when running from project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.feature_extractor import extract_features
from utils.hospital_finder import get_nearby_hospitals
from utils.signal_processor import apply_filter


APP_TITLE = "Retina-Vitals"
MODEL_PATH = os.path.join("models", "bp_predictor.pkl")


def try_load_model(path: str):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def main():
    st.set_page_config(page_title=APP_TITLE, layout="centered")
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
    chart_placeholder = st.empty()
    status_placeholder = st.empty()

    # Signal buffer keeps the most recent N samples
    buffer_size = 200
    signal_buffer = deque(maxlen=buffer_size)

    if run:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            st.error("Unable to access webcam (index 0).")
            return

        fs = 30  # assumed frame rate (Hz)

        try:
            while True:
                if not st.session_state.get("run_camera", False):
                    break
                ret, frame = camera.read()
                if not ret:
                    status_placeholder.error("Camera frame not available.")
                    break

                # Define ROI (center square)
                h, w = frame.shape[:2]
                size = min(h, w) // 3
                x0 = w // 2 - size // 2
                y0 = h // 2 - size // 2
                roi = frame[y0 : y0 + size, x0 : x0 + size]

                # Extract green channel intensity
                g_val = float(np.mean(roi[:, :, 1]))
                signal_buffer.append(g_val)

                # Render frame
                frame_window.image(frame, channels="BGR")

                # Process only when buffer is full
                if len(signal_buffer) == buffer_size:
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
                            st.success(f"Predicted BP: {sbp:.0f}/{dbp:.0f}")
                            if sbp > 140 or dbp > 90:
                                st.warning("High Blood Pressure Detected.")
                                st.info("Advice: Reduce salt intake, rest, and consult a clinician.")
                                m = folium.Map(location=[loc_lat, loc_lon], zoom_start=13)
                                hospitals = get_nearby_hospitals(loc_lat, loc_lon, radius_m=radius_m)
                                if not hospitals:
                                    st.info("No hospitals found (or Overpass API unavailable).")
                                for h in hospitals:
                                    folium.Marker(
                                        location=[h["lat"], h["lon"]],
                                        popup=f'{h[\"name\"]} ({h[\"distance_km\"]:.1f} km)',
                                        icon=folium.Icon(color="red", icon="plus-sign"),
                                    ).add_to(m)
                                st_folium(m, height=350)
                        else:
                            st.info("Prediction disabled (no model loaded).")

                        chart_placeholder.line_chart(clean_sig)
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
