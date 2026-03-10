"""Train a BP predictor model from PPG feature data."""

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.feature_extractor import extract_features

DEFAULT_RAW_DATA_PATH = os.path.join("training", "raw_data", "ppg_data.csv")
DEFAULT_MODEL_OUT_PATH = os.path.join("models", "bp_predictor.pkl")


# Expected CSV schema (example):
# timestamp,ppg_value,sbp,dbp
# 0.00,0.12,120,80
# 0.02,0.15,120,80
# ...


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw data not found at {path}. Place a CSV in training/raw_data/ppg_data.csv"
        )
    return pd.read_csv(path)


def build_feature_rows(
    df: pd.DataFrame,
    ppg_col: str,
    sbp_col: str,
    dbp_col: str,
    fs: float = 50.0,
    window_sec: float = 8.0,
):
    window_size = int(fs * window_sec)
    ppg = df[ppg_col].values
    sbp = df[sbp_col].values
    dbp = df[dbp_col].values

    X = []
    y = []

    for i in range(0, len(ppg) - window_size, window_size):
        segment = ppg[i : i + window_size]
        feats = extract_features(segment, fs)
        X.append([feats["hr_bpm"], feats["std_dev"], feats["mean_val"]])

        # Use the last BP values in the window as target
        y.append([sbp[i + window_size - 1], dbp[i + window_size - 1]])

    return np.array(X), np.array(y)


def parse_args():
    parser = argparse.ArgumentParser(description="Train BP predictor from PPG data.")
    parser.add_argument("--input", default=DEFAULT_RAW_DATA_PATH, help="Path to CSV file.")
    parser.add_argument("--ppg-col", default="ppg_value", help="Column name for PPG signal.")
    parser.add_argument("--sbp-col", default="sbp", help="Column name for systolic BP.")
    parser.add_argument("--dbp-col", default="dbp", help="Column name for diastolic BP.")
    parser.add_argument("--fs", type=float, default=50.0, help="Sampling rate (Hz).")
    parser.add_argument("--window-sec", type=float, default=8.0, help="Window size in seconds.")
    parser.add_argument("--model-out", default=DEFAULT_MODEL_OUT_PATH, help="Output model path.")
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_data(args.input)
    X, y = build_feature_rows(df, args.ppg_col, args.sbp_col, args.dbp_col, fs=args.fs, window_sec=args.window_sec)

    if len(X) == 0:
        raise ValueError("No training samples were generated. Check data length.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(model, args.model_out)

    print(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    main()
