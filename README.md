# Kaggle BPM Prediction – Pipeline and Submission

This repository contains a complete workflow to build submissions for a Kaggle Playground competition where the goal is to predict a song’s Beats-Per-Minute (BPM).

## Overview
- Target: `BeatsPerMinute`
- Metric: RMSE
- Inputs: `data/train.csv`, `data/test.csv`
- Outputs (submissions): CSVs in `submissions/`

## Project Structure
- `data/` – train/test and sample submission
- `data_docs/` – dataset artifacts (shapes, missingness, describes, correlations)
- `models_out/` – out-of-fold diagnostics (if generated)
- `submissions/` – generated submission CSV files
- `bpm_prediction.ipynb` – full E2E notebook (EDA, FE, CatBoost/LGBM/XGBoost baselines, blending, Optuna)
- `xgb_pipeline.ipynb` – compact XGBoost-only pipeline (EDA, FE, XGB CV/seed ensemble, optional random search)

## Workflow (high level)
1. EDA
   - Shapes, schema, missingness
   - Target distribution and correlation inspection
   - Optional correlation heatmap (top features)
2. Feature Engineering (FE)
   - Baseline FE: duration transforms; bounded logits; interaction terms (Energy×Loudness, Rhythm×Energy, AcousticQuality×*); Z-scores
   - Enhanced FE (toggle): additional nonlinearities, quantile ranks, more interactions (Mood, Energy–Loudness hub, Rhythm/Live), winsorization for stability
3. Modeling
   - Primary model: XGBoost (hist)
   - K-fold CV with early stopping (patience ~200, cap 20k rounds)
   - Seed ensemble: train across multiple random seeds and average predictions
   - Optional random search: sample XGB hyperparameters, score via CV, pick best
4. Submission
   - Write `ID/BeatsPerMinute` CSVs to `submissions/`

## How to Run (local)
1. Create env (Python 3.11+ recommended) and install deps:
   - pandas, numpy, matplotlib, seaborn, xgboost, scikit-learn, scipy
2. Place data under `data/` (train.csv, test.csv)
3. Open `xgb_pipeline.ipynb` and run top-to-bottom:
   - EDA cells (optional)
   - Toggle `USE_ENHANCED_FE` (default: True)
   - Toggle `USE_RANDOM_SEARCH` (default: True) and set `N_TRIALS_RS`
   - Run CV sanity cell, then seed ensemble cell
   - Output: `submissions/submission_xgb_pipeline.csv` or `submission_xgb_pipeline_random.csv`

## Kaggle Notebook Settings
- Accelerator: GPU (optional, speeds up XGBoost on GPU builds) or CPU
- Data sources: attach competition dataset
- Runtime: enable internet only if you need package installs

## Repro Tips
- Fix seeds list for seed ensemble
- Keep FE toggles consistent between CV and final training
- Clip predictions to plausible BPM range only if it improves OOF (not enabled by default)

## Notes
- LightGBM/CatBoost baselines and blending exist in `bpm_prediction.ipynb` for comparison but XGBoost-only pipeline (`xgb_pipeline.ipynb`) is the streamlined path for submission.
- Data artifacts (`data_docs/`) help with quick inspection and tracking data invariants.

## License
This project is for Kaggle competition participation and educational use.
