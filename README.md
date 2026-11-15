# Spatial Dependency Modelling in Multivariate Time Series

Traffic speed forecasting on METR-LA (207 sensors, 5-minute steps).
Step 2: models that forecast each sensor from its own history only, without the road graph.
They are the baseline that the graph-based models (Step 3) must beat.

| Model | Who |
|---|---|
| Linear, LSTM, TCN | Raluca |
| MLP, GRU, LightGBM | Eduard |

Training and evaluation use [BasicTS 1.1](https://github.com/GestaltCogTeam/BasicTS).

## Setup

Python 3.12. From the repo root:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate         # Linux / macOS
```

Install PyTorch first (the command depends on your machine):

```bash
# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
# NVIDIA GPU: use the command for your CUDA version from https://pytorch.org/get-started/locally/
```

Then the rest:

```bash
pip install -r requirements.txt
pip install "setuptools>=70,<82"
```

The last line is needed because `basicts` installs setuptools 59.5.0, which does not work on Python 3.12.
pip prints a dependency-conflict warning afterwards; it is harmless.

Check: `python -c "import torch, basicts; print(torch.__version__, basicts.__version__)"`

## Data

METR-LA is included in `data/METR-LA` in BasicTS format:
`{train,val,test}_data.npy` (speed, shape `[time, 207]`), `{train,val,test}_timestamps.npy`
(time of day, day of week), `adj_mx.pkl` (sensor graph), `graph_sensor_locations.csv`.
Split 70/10/20 by time. A value of 0 means missing.

## Repository structure

```
data/METR-LA/      dataset
notebooks/         …
src/data/          …
src/model/         …
src/utils/         …
steps/             task descriptions for each step
```

## How to run

*(added when the first model trains)*
