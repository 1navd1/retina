"""Train BP + Hb predictor from PPG waveform data inside archive.zip."""

import argparse
import ast
import io
import os
import re
import zipfile

import joblib
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_ARCHIVE_PATH = os.path.join(ROOT_DIR, "archive.zip")
DEFAULT_MODEL_OUT_PATH = os.path.join(ROOT_DIR, "models", "bp_predictor.pkl")


def parse_args():
    parser = argparse.ArgumentParser(description="Train BP/Hb predictor from PPG waveforms.")
    parser.add_argument(
        "--archive",
        default=DEFAULT_ARCHIVE_PATH,
        help="Path to archive.zip, CSV/XLSX, or HDF5 file.",
    )
    parser.add_argument("--member", default=None, help="Optional CSV/XLSX member inside a zip.")
    parser.add_argument("--ppg-key", default=None, help="HDF5 dataset key for PPG waveforms.")
    parser.add_argument("--labels-key", default=None, help="HDF5 dataset key for labels array.")
    parser.add_argument("--sbp-key", default=None, help="HDF5 dataset key for SBP labels.")
    parser.add_argument("--dbp-key", default=None, help="HDF5 dataset key for DBP labels.")
    parser.add_argument("--hb-key", default=None, help="HDF5 dataset key for Hb labels.")
    parser.add_argument("--ppg-channel", type=int, default=0, help="MAT channel index for PPG.")
    parser.add_argument("--bp-channel", type=int, default=1, help="MAT channel index for ABP.")
    parser.add_argument("--waveform-col", default=None, help="Column containing waveform arrays.")
    parser.add_argument("--id-col", default=None, help="Record/subject identifier for long format.")
    parser.add_argument("--sbp-col", default=None, help="Systolic BP column name.")
    parser.add_argument("--dbp-col", default=None, help="Diastolic BP column name.")
    parser.add_argument("--hb-col", default=None, help="Hemoglobin column name.")
    parser.add_argument("--model-out", default=DEFAULT_MODEL_OUT_PATH, help="Output model path.")
    return parser.parse_args()


def _find_column(df: pd.DataFrame, candidates):
    lowered = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _is_list_like_string(value: object) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    return (text.startswith("[") and text.endswith("]")) or ("," in text) or (" " in text)


def _parse_waveform_cell(value: object):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.startswith("[") and text.endswith("]"):
            return np.asarray(ast.literal_eval(text), dtype=float)
        parts = [p for p in re.split(r"[ ,;]+", text) if p]
        return np.asarray([float(p) for p in parts], dtype=float)
    return None


def load_tabular_dataframe(input_path: str, member: Optional[str]) -> pd.DataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    if input_path.lower().endswith(".zip"):
        with zipfile.ZipFile(input_path) as archive:
            members = [m for m in archive.namelist() if m.lower().endswith((".csv", ".xlsx"))]
            if not members:
                raise ValueError("Archive contains no CSV/XLSX files.")

            target = member or members[0]
            if target not in archive.namelist():
                raise ValueError(f"Member not found in archive: {target}")

            payload = archive.read(target)
            if target.lower().endswith(".csv"):
                return pd.read_csv(io.BytesIO(payload))
            return pd.read_excel(io.BytesIO(payload))

    if input_path.lower().endswith(".csv"):
        return pd.read_csv(input_path)
    if input_path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(input_path)

    raise ValueError("Unsupported tabular format. Use CSV/XLSX or a zip containing them.")


def _list_hdf5_datasets(h5):
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("h5py is required to read HDF5 files. Install with `pip install h5py`.") from exc

    datasets = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets.append(name)

    h5.visititems(visitor)
    return datasets


def load_hdf5_arrays(
    input_path: str,
    ppg_key: Optional[str],
    labels_key: Optional[str],
    sbp_key: Optional[str],
    dbp_key: Optional[str],
    hb_key: Optional[str],
):
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("h5py is required to read HDF5 files. Install with `pip install h5py`.") from exc

    with h5py.File(input_path, "r") as h5:
        dataset_names = _list_hdf5_datasets(h5)
        if not dataset_names:
            raise ValueError("No datasets found in the HDF5 file.")

        if not ppg_key:
            ppg_key = next((n for n in dataset_names if "ppg" in n.lower()), None)
        if not ppg_key:
            raise ValueError("Provide --ppg-key to locate waveform data in the HDF5 file.")

        ppg = h5[ppg_key][...]

        labels = None
        if labels_key:
            labels = h5[labels_key][...]
        else:
            if not sbp_key:
                sbp_key = next((n for n in dataset_names if "sbp" in n.lower() or "systolic" in n.lower()), None)
            if not dbp_key:
                dbp_key = next((n for n in dataset_names if "dbp" in n.lower() or "diastolic" in n.lower()), None)
            if not hb_key:
                hb_key = next((n for n in dataset_names if "hb" in n.lower() or "hemoglobin" in n.lower()), None)

            if not sbp_key or not dbp_key:
                raise ValueError("Provide --sbp-key and --dbp-key to locate labels in the HDF5 file.")

            sbp = h5[sbp_key][...]
            dbp = h5[dbp_key][...]
            if hb_key:
                hb = h5[hb_key][...]
                labels = np.column_stack([sbp, dbp, hb])
            else:
                labels = np.column_stack([sbp, dbp])

        return ppg, labels


def build_samples_from_waveforms(waveforms: np.ndarray, labels: np.ndarray):
    if waveforms is None or labels is None:
        raise ValueError("Waveforms and labels are required.")

    waveforms = np.asarray(waveforms)
    if waveforms.ndim == 1 and waveforms.size > 0 and isinstance(waveforms[0], (list, tuple, np.ndarray)):
        waveforms = [np.asarray(w, dtype=float) for w in waveforms]
    elif waveforms.ndim >= 2:
        waveforms = [np.asarray(w, dtype=float).ravel() for w in waveforms]
    else:
        raise ValueError("Unsupported waveform array shape.")

    rows = []
    for waveform in waveforms:
        feats = compute_features(waveform)
        if feats is None:
            continue
        rows.append(feats)

    labels = np.asarray(labels, dtype=float)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
    if len(rows) != labels.shape[0]:
        raise ValueError("Waveform count does not match label count.")

    return np.asarray(rows, dtype=float), labels


def load_mat_dataset(dataset_dir: str, ppg_channel: int, bp_channel: int):
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    mat_files = sorted(
        f for f in os.listdir(dataset_dir) if f.lower().endswith(".mat")
    )
    if not mat_files:
        raise ValueError("No .mat files found in the dataset directory.")

    rows = []
    labels = []

    for filename in mat_files:
        mat = sio.loadmat(os.path.join(dataset_dir, filename))
        if "p" not in mat:
            continue
        records = mat["p"]
        if records.size == 0:
            continue

        for idx in range(records.shape[1]):
            record = records[0, idx]
            if not isinstance(record, np.ndarray):
                continue
            data = record
            if data.ndim != 2:
                continue

            if data.shape[0] <= data.shape[1]:
                channels_first = True
            else:
                channels_first = False

            if channels_first:
                if ppg_channel >= data.shape[0] or bp_channel >= data.shape[0]:
                    continue
                ppg = data[ppg_channel]
                bp = data[bp_channel]
            else:
                if ppg_channel >= data.shape[1] or bp_channel >= data.shape[1]:
                    continue
                ppg = data[:, ppg_channel]
                bp = data[:, bp_channel]

            feats = compute_features(ppg)
            if feats is None:
                continue
            rows.append(feats)
            labels.append([float(np.max(bp)), float(np.min(bp))])

    if not rows:
        raise ValueError("No training samples were generated from MAT files.")

    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=float)


def compute_features(waveform: np.ndarray):
    waveform = np.asarray(waveform, dtype=float)
    waveform = waveform[np.isfinite(waveform)]
    if waveform.size < 2:
        return None

    std_dev = float(np.std(waveform))
    kurt = float(kurtosis(waveform, fisher=False, bias=False))
    fft_vals = np.fft.rfft(waveform - np.mean(waveform))
    fft_power = float(np.sum(np.abs(fft_vals) ** 2) / len(fft_vals))
    return [std_dev, kurt, fft_power]


def build_samples(
    df: pd.DataFrame,
    waveform_col: Optional[str],
    id_col: Optional[str],
    sbp_col: Optional[str],
    dbp_col: Optional[str],
    hb_col: Optional[str],
):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    sbp_col = sbp_col or _find_column(df, ["sbp", "systolic bp", "systolic blood pressure(mmHg)"])
    dbp_col = dbp_col or _find_column(df, ["dbp", "diastolic bp", "diastolic blood pressure(mmHg)"])
    hb_col = hb_col or _find_column(df, ["hb", "hemoglobin", "hemoglobin(g/dl)"])

    if not sbp_col or not dbp_col:
        raise ValueError(
            "Could not locate SBP/DBP columns. Provide --sbp-col and --dbp-col with the correct names."
        )

    include_hb = bool(hb_col)

    if waveform_col and waveform_col in df.columns:
        sample_value = df[waveform_col].dropna().iloc[0]
        if _is_list_like_string(sample_value) or isinstance(sample_value, (list, tuple, np.ndarray)):
            rows = []
            labels = []
            for _, row in df.iterrows():
                waveform = _parse_waveform_cell(row[waveform_col])
                if waveform is None:
                    continue
                feats = compute_features(waveform)
                if feats is None:
                    continue
                rows.append(feats)
                label = [row[sbp_col], row[dbp_col]]
                if include_hb:
                    label.append(row[hb_col])
                labels.append(label)
            return np.asarray(rows, dtype=float), np.asarray(labels, dtype=float)

    ppg_cols = [c for c in df.columns if c.lower().startswith("ppg_")]
    if len(ppg_cols) >= 3:
        rows = []
        labels = []
        ppg_cols_sorted = sorted(ppg_cols, key=lambda c: int(re.findall(r"\d+", c)[0]) if re.findall(r"\d+", c) else c)
        for _, row in df.iterrows():
            waveform = row[ppg_cols_sorted].to_numpy(dtype=float)
            feats = compute_features(waveform)
            if feats is None:
                continue
            rows.append(feats)
            label = [row[sbp_col], row[dbp_col]]
            if include_hb:
                label.append(row[hb_col])
            labels.append(label)
        return np.asarray(rows, dtype=float), np.asarray(labels, dtype=float)

    ppg_value_col = _find_column(df, ["ppg_value", "ppg", "ppg_signal"])
    if ppg_value_col:
        id_col = id_col or _find_column(df, ["subject_id", "subject id", "record_id", "record id", "patient_id"])
        if not id_col:
            raise ValueError("Long-format data requires an id column. Provide --id-col.")
        rows = []
        labels = []
        for _, group in df.groupby(id_col):
            waveform = group[ppg_value_col].to_numpy(dtype=float)
            feats = compute_features(waveform)
            if feats is None:
                continue
            rows.append(feats)
            label = [group[sbp_col].iloc[0], group[dbp_col].iloc[0]]
            if include_hb:
                label.append(group[hb_col].iloc[0])
            labels.append(label)
        return np.asarray(rows, dtype=float), np.asarray(labels, dtype=float)

    raise ValueError(
        "Could not infer waveform columns. Provide --waveform-col, use PPG_* columns, "
        "or include a long-format column like ppg_value with an id column."
    )


def main():
    args = parse_args()
    if os.path.isdir(args.archive):
        X, y = load_mat_dataset(
            args.archive,
            ppg_channel=args.ppg_channel,
            bp_channel=args.bp_channel,
        )
    elif args.archive.lower().endswith((".hdf5", ".h5")):
        waveforms, labels = load_hdf5_arrays(
            args.archive,
            ppg_key=args.ppg_key,
            labels_key=args.labels_key,
            sbp_key=args.sbp_key,
            dbp_key=args.dbp_key,
            hb_key=args.hb_key,
        )
        X, y = build_samples_from_waveforms(waveforms, labels)
    else:
        df = load_tabular_dataframe(args.archive, args.member)
        X, y = build_samples(
            df,
            waveform_col=args.waveform_col,
            id_col=args.id_col,
            sbp_col=args.sbp_col,
            dbp_col=args.dbp_col,
            hb_col=args.hb_col,
        )

    if X.size == 0:
        raise ValueError("No samples were generated. Check waveform parsing and labels.")

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=200, random_state=42)),
        ]
    )
    pipeline.fit(X, y)

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(pipeline, args.model_out)
    print(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    main()
