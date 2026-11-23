# Spatial Dependency Modelling in Multivariate Time Series

Traffic speed forecasting on METR-LA (207 sensors, 5-minute steps).
Step 2: models that forecast each sensor from its own history only, without the road graph.
They are the baseline that the graph-based models (Step 3) must beat.

| Model | Who |
|---|---|
| Linear, LSTM, TCN, encoder-decoder LSTM | Raluca |
| MLP, GRU, encoder-decoder GRU, LightGBM | Eduard |

Training and evaluation use [BasicTS 1.1](https://github.com/GestaltCogTeam/BasicTS).
All neural models share one set of training rules (`make_cfg` in `src/utils/config.py`):
12 steps in, 12 out; z-score scaling with training statistics; missing values (0) handled as in DCRNN;
early stopping on validation MAE; MAE / RMSE / MAPE at 15, 30 and 60 minutes; seed 42.

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

Training uses the GPU if there is one, otherwise the CPU.

## Data

METR-LA is included in `data/METR-LA` in BasicTS format:
`{train,val,test}_data.npy` (speed, shape `[time, 207]`), `{train,val,test}_timestamps.npy`
(time of day, day of week), `adj_mx.pkl` (sensor graph), `graph_sensor_locations.csv`.
Split 70/10/20 by time. A value of 0 means missing.

## Repository structure

```
data/METR-LA/                          dataset
notebooks/                             data analysis (Step 1), results (Step 2)
src/data/                              data loading and analysis helpers
src/model/basic_model/                 simple rules from Step 1 (last value, weekly average)
src/model/baseline_temporal_models/    Step 2 models, one shared model for all sensors
src/utils/config.py                    shared training settings (make_cfg)
src/utils/runner.py                    train one model from the command line
src/utils/tuning/                      hyperparameter tuning (Optuna): lgbm_tuning.py, nn_tuning.py
steps/                                 task descriptions for each step
checkpoints/                           training output (not in git)
```

## How to run

All commands from the repo root.

**Train one model** (`--tod` / `--dow` add time of day / day of week, `--epochs` is the maximum, default 10):

```bash
python -m src.utils.runner --model TCN --tod --dow --epochs 100
python -m src.utils.runner --model LGBM --tod
```

Models: `Linear`, `LSTM`, `TCN`, `EncoderDecoderLSTM`, `MLP`, `GRU`, `EncoderDecoderGRU`, `LGBM`.
Results go to `checkpoints/<model>/...`: `test_metrics.json`, training log, best checkpoint.

**Tune the neural models** (`src/utils/tuning/nn_tuning.py`): 10 tries per model, 10 epochs each,
inputs speed + time of day + day of week, chosen on validation MAE (the test set is not used).
The first try is the current default settings. Tries run 3 at a time; an interrupted run continues where it stopped.

```bash
python -m src.utils.tuning.nn_tuning tune --models Linear TCN LSTM   # the search
python -m src.utils.tuning.nn_tuning summary                         # table of all tries (+ CSV)
python -m src.utils.tuning.nn_tuning final --models TCN LSTM         # full training with the best settings + test
python -m src.utils.tuning.nn_tuning final --models TCN --seed 1     # the same with another seed
```

Tuning output: `checkpoints/tuning/` (`optuna.db`, `<model>_trials.csv`, worker logs);
final runs: `checkpoints/final/<model>/<tuned|defaults>_seed<seed>/`.
