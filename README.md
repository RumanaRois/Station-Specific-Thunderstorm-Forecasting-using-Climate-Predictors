# Thunderstorm Frequency Time Series Analysis

R scripts for analyzing monthly thunderstorm frequency (and related climate
variables) at three stations in Bangladesh: **Sylhet**, **Sreemangal**, and
**Mymensingh**. Each station script runs STL/X11 decomposition and compares
several forecasting models (ARIMA, ETS, ANN, SVR, XGBoost, LSTM — with and
without seasonal decomposition).

## Repository structure

```
.
├── S1_Dataset.xlsx                        # Source data (all three stations, one sheet)
├── load_station_data.R                    # Shared helper: extracts one station's data
├── Sylhet_TS_R.R                           # Sylhet: decomposition + forecasting models
├── Sreemangal_TS_R.R                       # Sreemangal: decomposition + forecasting models
├── Mymensingh_TS_R.R                       # Mymensingh: decomposition + forecasting models
├── All_Stations_Combine_Plot.R             # Combined exploratory plot (all 3 stations)
└── TS_Forecasting_with_Climate_Predictors.py  # Hybrid STL + ML/DL + residual-model forecasting with climate predictors
```

## Data

`S1_Dataset.xlsx` contains a single sheet, `Data`, with three side-by-side
9-column blocks (one per station):

| Station | Year | Month | Monthly_TSF | Avg_Temp | Avg_RH | Avg_CloudC | Avg_Rainfall | Avg_APressure |
|---|---|---|---|---|---|---|---|---|

`load_station_data.R` reads this sheet and returns a tidy data frame for a
single station (renaming columns to `Y`, `M`, `MT`, ... to match the rest of
the analysis code).

## Requirements

```r
install.packages(c(
  "forecast", "e1071", "xgboost", "tensorflow", "keras3", "seasonal",
  "Metrics", "ggplot2", "dplyr", "lubridate", "tidyr", "reshape2", "fpp2",
  "readxl", "openxlsx", "patchwork", "viridis", "RColorBrewer", "reticulate"
))
```

TensorFlow/Keras (used for the LSTM models) require a configured Python
environment via `reticulate`. See the commented setup block at the top of
the LSTM section in each station script.

## Usage

Open this folder as your R working directory (e.g. an RStudio Project at
the repo root, or `setwd()` to wherever you cloned the repo), then run any
of the station scripts directly:

```r
source("Sylhet_TS_R.R")
```

Each script loads its data via the shared `load_station_data.R` helper, so
no absolute file paths are required — just make sure `S1_Dataset.xlsx` and
`load_station_data.R` stay in the same directory as the station scripts.

Outputs (error-metric spreadsheets, figures) are written to the current
working directory using relative filenames, e.g. `Sylhet_Errors.xlsx`,
`Figure1_AllStations.tiff`.

## Forecasting with climate predictors (Python)

`TS_Forecasting_with_Climate_Predictors.py` is a complementary Python
analysis that forecasts a Discomfort Index (DI) at 3-hourly resolution
using climate predictors — Rainfall, Cloud cover, Atmospheric pressure,
Wind, and Season — as exogenous features.

It builds a grid of hybrid models by combining:

- **Base learners** (fit on the STL-seasonally-adjusted series plus
  climate/calendar/lag features): ANN, GRU, LSTM, Decision Tree, Random
  Forest, SVR, XGBoost, and Facebook Prophet.
- **Residual correctors** (fit on the base learner's training residuals):
  ETS (Holt-Winters), SARIMAX, and TBATS.

Each combination (e.g. `STL + LSTM + SARIMAX`) is scored on a held-out test
split with MSE, RMSE, MAE, MAPE, MASE, and R².

**Inputs expected** (place alongside the script, not included in this
repo): `S1_data.xlsx`, `DI_data.xlsx`, and `DI_data_LSTM.xlsx`, each with a
`DateTime` column plus `DI`, `Rainfall`, `Cloud`, `Atmosphere`, `Wind` (and
`Season` for the latter two). This is a separate, higher-frequency dataset
from `S1_Dataset.xlsx` used by the R scripts above — don't mix them up.

**Requirements:**

```bash
pip install pandas numpy scikit-learn statsmodels tbats prophet tensorflow xgboost openpyxl
```

Run with:

```bash
python TS_Forecasting_with_Climate_Predictors.py
```
