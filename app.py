import os
import sys
import time
from collections import deque

import av
import joblib
import numpy as np
import streamlit as st
from scipy.ndimage import zoom
from scipy.signal import butter, find_peaks, sosfiltfilt
from streamlit_folium import st_folium
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
import folium

# Ensure utils is importable when running from project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.feature_extractor import (
    append_signal_sample,
    center_roi,
    extract_ppg_features,
    init_signal_buffer,
    mean_green_from_roi,
)
from utils.hospital_finder import get_nearby_hospitals
from utils.signal_processor import process_signal


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


def render_hospital_map(lat: float, lon: float, radius_m: int, key: str):
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
    st_folium(m, height=350, key=key)


class PPGVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.signal_buffer = None
        self.box_size = 100
        self.heatmap_enabled = False
        self.heatmap_grid = 64
        self.heatmap_alpha = 0.5
        self.heatmap_window_sec = 6.0
        self.heatmap_update_every = 3
        self.heatmap_fs = 30
        self.heatmap_buffer = None
        self.heatmap_mask_bgr = None
        self._heatmap_frame_count = 0
        self._heatmap_config = None

    @staticmethod
    def _thermal_colormap(norm_map: np.ndarray) -> np.ndarray:
        """Map [0,1] values to a thermal-style RGB colormap."""
        anchors = np.array(
            [
                [0, 0, 128],
                [0, 255, 255],
                [255, 255, 0],
                [255, 0, 0],
            ],
            dtype=np.float32,
        )
        x = np.clip(norm_map, 0.0, 1.0) * 3.0
        idx = np.floor(x).astype(np.int32)
        idx = np.clip(idx, 0, 2)
        t = (x - idx)[..., None]
        c0 = anchors[idx]
        c1 = anchors[idx + 1]
        rgb = c0 + (c1 - c0) * t
        return rgb.astype(np.uint8)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        height, width = image.shape[:2]
        x0, y0, x1, y1 = center_roi(image, box_size=self.box_size)
        g_val = mean_green_from_roi(image, (x0, y0, x1, y1))
        if self.signal_buffer is not None:
            append_signal_sample(self.signal_buffer, g_val)

        if self.heatmap_enabled:
            window_frames = max(30, int(self.heatmap_window_sec * self.heatmap_fs))
            cfg = (self.heatmap_grid, window_frames, self.heatmap_fs)
            if self._heatmap_config != cfg:
                self.heatmap_buffer = deque(maxlen=window_frames)
                self.heatmap_mask_bgr = None
                self._heatmap_frame_count = 0
                self._heatmap_config = cfg

            green = image[:, :, 1].astype(np.float32)
            scale_y = self.heatmap_grid / float(height)
            scale_x = self.heatmap_grid / float(width)
            small = zoom(green, (scale_y, scale_x), order=1)
            small = np.clip(small, 0.0, 255.0)

            if self.heatmap_buffer is not None:
                self.heatmap_buffer.append(small)
                self._heatmap_frame_count += 1

                if (
                    len(self.heatmap_buffer) >= max(30, int(0.8 * window_frames))
                    and self._heatmap_frame_count % self.heatmap_update_every == 0
                ):
                    data = np.stack(self.heatmap_buffer, axis=0)
                    data = data - data.mean(axis=0, keepdims=True)
                    try:
                        sos = butter(2, [0.7, 3.0], btype="bandpass", fs=self.heatmap_fs, output="sos")
                        filtered = sosfiltfilt(sos, data, axis=0)
                        strength = np.std(filtered, axis=0)
                    except ValueError:
                        strength = np.std(data, axis=0)

                    lo = np.percentile(strength, 5)
                    hi = np.percentile(strength, 95)
                    norm = (strength - lo) / (hi - lo + 1e-6)
                    norm = np.clip(norm, 0.0, 1.0)
                    heatmap_rgb = self._thermal_colormap(norm)
                    heatmap_full = zoom(heatmap_rgb, (height / heatmap_rgb.shape[0], width / heatmap_rgb.shape[1], 1), order=1)
                    heatmap_full = np.clip(heatmap_full, 0.0, 255.0).astype(np.uint8)
                    self.heatmap_mask_bgr = heatmap_full[:, :, ::-1]

            if self.heatmap_mask_bgr is not None:
                alpha = np.clip(self.heatmap_alpha, 0.0, 1.0)
                image = (image.astype(np.float32) * (1.0 - alpha) + self.heatmap_mask_bgr.astype(np.float32) * alpha).astype(
                    np.uint8
                )

        # Draw ROI rectangle in-place (BGR).
        image[y0:y0 + 2, x0:x1] = (0, 255, 0)
        image[y1 - 2:y1, x0:x1] = (0, 255, 0)
        image[y0:y1, x0:x0 + 2] = (0, 255, 0)
        image[y0:y1, x1 - 2:x1] = (0, 255, 0)

        return av.VideoFrame.from_ndarray(image, format="bgr24")


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

    if "calibrated" not in st.session_state:
        st.session_state.calibrated = False
        st.session_state.sbp_offset = 0.0
        st.session_state.dbp_offset = 0.0
        st.session_state.calibration_mode = False

    with st.expander("Instructions", expanded=True):
        st.markdown(
            """
- Ensure good lighting and keep your face/hand steady in the camera view.
- If using a phone, turn on the rear flash (torch) for stable illumination.
- Toggle **Run Camera** to begin.
- If the model file is missing, prediction will be disabled but the signal graph will still render.
            """
        )

    model = try_load_model(MODEL_PATH)
    if model is None:
        st.warning("Model not found or failed to load. Place a trained model at `models/bp_predictor.pkl`.")
    else:
        expected_features = 5
        model_feature_count = getattr(model, "n_features_in_", None)
        if model_feature_count is not None and model_feature_count != expected_features:
            st.warning(
                f"Model expects {model_feature_count} features, but the app now computes {expected_features}. "
                "Please retrain the model."
            )
            model = None

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Calibrate System"):
            st.session_state.calibration_mode = True
    with col2:
        if st.session_state.calibrated:
            st.success(
                "Calibrated | SBP offset: {:.1f} | DBP offset: {:.1f}".format(
                    st.session_state.sbp_offset, st.session_state.dbp_offset
                )
            )
        else:
            st.warning("Not calibrated (using generic model)")

    with st.sidebar:
        st.header("Location Settings")
        default_lat = 28.6139
        default_lon = 77.2090
        loc_lat = st.number_input("Latitude", value=default_lat, format="%.6f")
        loc_lon = st.number_input("Longitude", value=default_lon, format="%.6f")
        radius_m = st.slider("Search Radius (m)", min_value=1000, max_value=10000, value=5000, step=500)
        st.header("Camera Settings")
        request_torch = st.checkbox("Request rear flash (torch)", value=False)
        reset_buffer = st.button("Reset signal buffer")
        st.header("Perfusion Heatmap")
        heatmap_enabled = st.checkbox("Enable heatmap overlay", value=False)
        heatmap_alpha = st.slider("Heatmap opacity", min_value=0.1, max_value=0.9, value=0.55, step=0.05)
        heatmap_grid = st.slider("Heatmap grid size", min_value=32, max_value=96, value=64, step=8)
        heatmap_window = st.slider("Heatmap window (sec)", min_value=3.0, max_value=10.0, value=6.0, step=0.5)
        heatmap_update_every = st.slider("Heatmap update (frames)", min_value=1, max_value=6, value=3, step=1)
    run = st.checkbox("Run Camera", key="run_camera")
    st.caption("Live PPG Signal (Green intensity over time)")
    chart_placeholder = st.empty()
    status_placeholder = st.empty()
    result_placeholder = st.empty()
    advice_placeholder = st.empty()
    map_placeholder = st.empty()

    # Keep the latest 30 seconds of signal samples for real-time plotting.
    sampling_rate_hz = 30
    if "signal_buffer" not in st.session_state:
        st.session_state["signal_buffer"] = init_signal_buffer(fs=sampling_rate_hz, duration_sec=30)
    signal_buffer = st.session_state["signal_buffer"]
    window_sec = 8
    window_samples = int(sampling_rate_hz * window_sec)
    min_processing_samples = max(200, window_samples)

    if "run_camera_prev" not in st.session_state:
        st.session_state["run_camera_prev"] = False
    if reset_buffer or (run and not st.session_state["run_camera_prev"]):
        st.session_state["signal_buffer"] = init_signal_buffer(fs=sampling_rate_hz, duration_sec=30)
        signal_buffer = st.session_state["signal_buffer"]
    st.session_state["run_camera_prev"] = run

    if run:
        fs = sampling_rate_hz  # assumed frame rate (Hz)

        video_constraints = {"facingMode": {"ideal": "environment"}}
        if request_torch:
            video_constraints["advanced"] = [{"torch": True}]

        ctx = webrtc_streamer(
            key="ppg_webrtc",
            video_processor_factory=PPGVideoProcessor,
            media_stream_constraints={"video": video_constraints, "audio": False},
            desired_playing_state=True,
        )
        if ctx.video_processor:
            ctx.video_processor.signal_buffer = signal_buffer
            ctx.video_processor.heatmap_enabled = heatmap_enabled
            ctx.video_processor.heatmap_alpha = heatmap_alpha
            ctx.video_processor.heatmap_grid = heatmap_grid
            ctx.video_processor.heatmap_window_sec = heatmap_window
            ctx.video_processor.heatmap_update_every = heatmap_update_every
            ctx.video_processor.heatmap_fs = sampling_rate_hz

        if ctx.state.playing:
            while ctx.state.playing and st.session_state.get("run_camera", False):
                chart_placeholder.line_chart(np.array(signal_buffer, dtype=np.float32))

                if len(signal_buffer) >= min_processing_samples:
                    raw_sig = np.array(signal_buffer, dtype=np.float32)[-window_samples:]
                    clean_sig = process_signal(raw_sig, fs)

                    peaks, _ = find_peaks(clean_sig, distance=20)
                    if len(peaks) > 5:
                        hr, std_dev, skewness, kurtosis, mean_val = extract_ppg_features(clean_sig, fs)

                        status_placeholder.info(
                            "HR: {:.1f} bpm | STD: {:.2f} | Skew: {:.2f} | Kurt: {:.2f} | Mean: {:.2f}".format(
                                hr, std_dev, skewness, kurtosis, mean_val
                            )
                        )

                        if model is not None:
                            prediction = model.predict([[hr, std_dev, skewness, kurtosis, mean_val]])
                            sbp = float(prediction[0][0])
                            dbp = float(prediction[0][1])
                            st.session_state.current_raw_sbp = sbp
                            st.session_state.current_raw_dbp = dbp

                            sbp_cal = sbp - st.session_state.sbp_offset
                            dbp_cal = dbp - st.session_state.dbp_offset
                            hb = estimate_hb(mean_val, std_dev)
                            triage = triage_patient({"sbp": sbp_cal, "dbp": dbp_cal, "hr": hr, "hb": hb})

                            with result_placeholder.container():
                                render_results_card(sbp_cal, dbp_cal, hr, hb, triage)

                            with advice_placeholder.container():
                                if triage == "EMERGENCY":
                                    st.error("Please visit the nearest facility immediately.")
                                elif triage == "WARNING":
                                    st.warning("Health risk detected. Rest and consult a clinician soon.")
                                else:
                                    st.success("Vitals are in a relatively stable range.")

                            if "last_map_render" not in st.session_state:
                                st.session_state["last_map_render"] = 0.0
                            if "last_map_params" not in st.session_state:
                                st.session_state["last_map_params"] = None
                            if "map_key_counter" not in st.session_state:
                                st.session_state["map_key_counter"] = 0

                            if triage == "EMERGENCY":
                                now = time.time()
                                params = (round(loc_lat, 6), round(loc_lon, 6), int(radius_m))
                                should_render = (
                                    st.session_state["last_map_params"] != params
                                    or now - st.session_state["last_map_render"] > 5.0
                                )
                                if should_render:
                                    st.session_state["map_key_counter"] += 1
                                    with map_placeholder.container():
                                        render_hospital_map(
                                            loc_lat,
                                            loc_lon,
                                            radius_m,
                                            key=f"hospital_map_{st.session_state['map_key_counter']}",
                                        )
                                    st.session_state["last_map_render"] = now
                                    st.session_state["last_map_params"] = params
                            else:
                                map_placeholder.empty()
                        else:
                            st.info("Prediction disabled (no model loaded).")

                    else:
                        status_placeholder.info("Waiting for stable peaks...")

                time.sleep(0.05)

        if st.session_state.calibration_mode:
            st.subheader("Calibration Required")
            st.info("1. Place finger on camera. 2. Wait for reading. 3. Enter cuff BP.")
            if "current_raw_sbp" in st.session_state and "current_raw_dbp" in st.session_state:
                st.metric("App Reading (Raw SBP)", f"{st.session_state.current_raw_sbp:.0f}")
                st.metric("App Reading (Raw DBP)", f"{st.session_state.current_raw_dbp:.0f}")
                cuff_sbp = st.number_input("Enter cuff SBP", min_value=70, max_value=220, value=120)
                cuff_dbp = st.number_input("Enter cuff DBP", min_value=40, max_value=140, value=80)
                if st.button("Save Calibration"):
                    st.session_state.sbp_offset = st.session_state.current_raw_sbp - cuff_sbp
                    st.session_state.dbp_offset = st.session_state.current_raw_dbp - cuff_dbp
                    st.session_state.calibrated = True
                    st.session_state.calibration_mode = False
                    st.rerun()
            else:
                st.warning("Waiting for a live reading to calibrate.")
    else:
        st.info("Camera is off. Toggle 'Run Camera' to start.")


if __name__ == "__main__":
    main()
