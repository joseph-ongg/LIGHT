# LIGHT — Microlensing Parameter Inference

Predicts Paczyński microlensing parameters from partial light curves using an LSTM + χ² optimizer pipeline.

## Overview

Gravitational microlensing events are often observed mid-event — before the peak is known. This project fits the five Paczyński parameters (t0, tE, u0, fs, I_bl) from a cutoff curve using two stages:

1. **LSTM** — a 3-layer LSTM reads the partial curve and produces an initial parameter estimate
2. **χ² optimizer** — scipy L-BFGS-B refines the LSTM guess by minimizing χ² with a learned prior

## Parameters

| Parameter | Description |
|-----------|-------------|
| t0 (Tmax) | Time of peak magnification (HJD) |
| tE (tau) | Einstein crossing time (days) |
| u0 (umin) | Minimum impact parameter |
| fs (fbl) | Source flux fraction (blending) |
| I_bl | Baseline magnitude |

## Results

The LSTM recovers t0 and I_bl with near-perfect accuracy (R² = 0.997, 0.998) but cannot disentangle the degenerate tE–u0–fs parameters from a partial curve. The χ² optimizer fits the Paczyński model exactly but drifts without an informed prior (R² = −0.52). Combining them yields R² = 0.81.

## Files

| File | Description |
|------|-------------|
| `trainer.py` | LSTM model definition and training loop |
| `curve_to_params.py` | χ² optimizer and inference pipeline |
| `prepare_data.py` | Runs inference on test set, saves `testdata.npz` |
| `data_prep.py` | Dataset loading and preprocessing |
| `gd_fit.py` | Gradient-descent baseline fitter |
| `plot_gd_fit.py` | Regenerate baseline plot from saved CSV |
| `total_loss.py` | Evaluate overall loss on test set |

## Usage

**Train the model:**
```bash
python trainer.py
```

**Prepare test data (runs LSTM + optimizer on test set):**
```bash
python prepare_data.py
```

**Run inference on a single event:**
```bash
python curve_to_params.py --year 2011 --cid 0133 --plot True
```

## Data

Training and testing data should be organized as:
```
training/<year>/curve_<year>_<id>.dat
training/<year>/params_<year>_<id>.dat
testing/<year>/curve_<year>_<id>.dat
testing/<year>/params_<year>_<id>.dat
```

## Requirements

```
torch
numpy
scipy
matplotlib
```
