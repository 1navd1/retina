# Retina-Vitals

Retina-Vitals is a hackathon demo that estimates pulse from the webcam green channel and (optionally) predicts blood pressure using a trained ML model.

## Quickstart

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the app:

```bash
streamlit run app.py
```

## Training the Model

1. Place a CSV file at `training/raw_data/ppg_data.csv` with columns:

- `ppg_value`
- `sbp`
- `dbp`

2. Train:

```bash
python training/train_model.py
```

This will write `models/bp_predictor.pkl`.

### Kaggle MIMIC-III PPG Dataset

If you use the Kaggle dataset `mimiciiippgall`, download it from Kaggle and point the trainer to the correct CSV and column names:

```bash
python training/train_model.py --input training/raw_data/YOUR_FILE.csv --ppg-col <PPG_COL> --sbp-col <SBP_COL> --dbp-col <DBP_COL>
```

## Notes

This project is a **demo** for hackathon use only. It is **not** a medical device and does not provide clinical-grade measurements.

The hospital map uses the OpenStreetMap Overpass API and requires internet access.
