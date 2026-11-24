"""Hyperparameter tuning of the neural models with Optuna (Step 2, task 5).

Same rules for every model:
  - the same budget: N tries per model (default 10, as for LGBM), each trained for 10 epochs;
  - the same inputs: 12 steps of speed + time of day + day of week (best inputs from task 4);
  - decisions on the VALIDATION MAE only; the test set is never evaluated during tuning;
  - the first try of every model is its current default settings, so the gain from tuning is measured directly.
Tries that are clearly worse than the median after 3 epochs are stopped early (pruned) to save time.
All tries are stored in checkpoints/tuning/optuna.db, so an interrupted run continues where it stopped.

Run from the repo root:
    python -m src.utils.tuning.nn_tuning tune --models Linear TCN LSTM          # search (e.g. overnight)
    python -m src.utils.tuning.nn_tuning summary                                # table of all tries
    python -m src.utils.tuning.nn_tuning final --models TCN LSTM                # full training, best settings
    python -m src.utils.tuning.nn_tuning final --models TCN --seed 1            # the same with another seed
    python -m src.utils.tuning.nn_tuning final --models LSTM --defaults         # full training, default settings
"""
import argparse
import os
import subprocess
import sys
import time

import optuna
import torch
from basicts import BasicTSLauncher
from basicts.runners.callback import BasicTSCallback, EarlyStopping
from optuna.trial import TrialState

from src.model.baseline_temporal_models.encoder_decoder_lstm import EncoderDecoderLSTM, EncoderDecoderLSTMConfig
from src.model.baseline_temporal_models.linear import Linear, LinearConfig
from src.model.baseline_temporal_models.lstm import LSTM, LSTMConfig
from src.model.baseline_temporal_models.tcn import TCN, TCNConfig
from src.utils.config import make_cfg

ROOT = os.environ.get('NN_TUNING_ROOT', 'checkpoints')       # set to another folder for test runs
TUNING_DIR, FINAL_DIR = f'{ROOT}/tuning', f'{ROOT}/final'
STORAGE_URL = f'sqlite:///{TUNING_DIR}/optuna.db'
FEATURES = dict(use_tod=True, use_dow=True, input_len=12)    # same inputs for every model
TRAIN_KEYS = {'lr', 'batch_size'}                             # training settings; everything else goes to the model
COUNTED = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING)   # tries that use up the budget


# ---------------------------------------------------------------- search spaces ----------------------------------
# Learning rate: the same range for every model. Batch size: fixed at 64 for every model (make_cfg),
# because 10-epoch tries are biased on it (a bigger batch makes fewer updates in the same 10 epochs).

def space_linear(t):
    t.suggest_float('lr', 3e-4, 3e-3, log=True)      # tuned; Linear has no size settings to tune besides it


def space_lstm(t):
    t.suggest_categorical('hidden_size', [32, 64, 128])
    if t.suggest_int('num_layers', 1, 2) > 1:
        t.suggest_float('dropout', 0.0, 0.3)
    t.suggest_float('lr', 3e-4, 3e-3, log=True)


def space_tcn(t):
    t.suggest_categorical('channels', [16, 32, 64])
    t.suggest_categorical('kernel_size', [2, 3])
    t.suggest_int('num_layers', 4, 5)            # with kernel 2: sees 16 / 32 steps back, always >= 12
    t.suggest_float('lr', 3e-4, 3e-3, log=True)


def space_encoder_decoder_lstm(t):
    space_lstm(t)
    t.suggest_categorical('teacher_forcing', [False, True])


MODELS = {   # name: (model, config, search space, current default settings = first try)
    'Linear': (Linear, LinearConfig, space_linear, dict(lr=1e-3)),
    'LSTM': (LSTM, LSTMConfig, space_lstm, dict(hidden_size=64, num_layers=1, lr=1e-3)),
    'TCN': (TCN, TCNConfig, space_tcn, dict(channels=32, kernel_size=2, num_layers=4, lr=1e-3)),
    'EncoderDecoderLSTM': (EncoderDecoderLSTM, EncoderDecoderLSTMConfig, space_encoder_decoder_lstm,
                           dict(hidden_size=64, num_layers=1, teacher_forcing=False, lr=1e-3)),
}


# ---------------------------------------------------------------- building and training --------------------------

def build(name, params, **overrides):
    """Settings (a flat dict, as Optuna stores them) -> (BasicTS config, number of model parameters)."""
    model, config_cls = MODELS[name][:2]
    model_config = config_cls(**FEATURES, **{k: v for k, v in params.items() if k not in TRAIN_KEYS})
    n_params = sum(p.numel() for p in model(model_config).parameters())
    train = {k: v for k, v in params.items() if k in TRAIN_KEYS}
    return make_cfg(model, model_config, use_timestamps=True, **train, **overrides), n_params


class OptunaReport(BasicTSCallback):
    """After every epoch: give the validation MAE to Optuna and stop training if Optuna prunes the try."""
    def __init__(self, trial):
        self.trial, self.best, self.epochs, self.pruned = trial, float('inf'), 0, False

    def on_validate_end(self, runner, train_step=None, train_epoch=None, **kwargs):
        val_mae = runner.meter_pool.get_value('val/MAE')
        self.epochs = train_epoch
        self.best = min(self.best, val_mae)
        self.trial.report(val_mae, step=train_epoch)
        if self.trial.should_prune():
            self.pruned = True
            runner.should_training_stop = True


def objective(trial, name, epochs):
    MODELS[name][2](trial)                                   # choose the settings of this try
    report = OptunaReport(trial)
    cfg, n_params = build(
        name, trial.params, num_epochs=epochs,
        callbacks=[EarlyStopping(patience=5), report],
        eval_after_train=False, save_results=False,          # no test set during tuning
        ckpt_save_dir=f'{TUNING_DIR}/{name}/trial_{trial.number:02d}')
    start = time.time()
    BasicTSLauncher.launch_training(cfg)
    trial.set_user_attr('n_params', n_params)
    trial.set_user_attr('minutes', round((time.time() - start) / 60, 1))
    trial.set_user_attr('epochs', report.epochs)
    if report.pruned:
        raise optuna.TrialPruned()
    return report.best                                       # best validation MAE of this try


# ---------------------------------------------------------------- commands ---------------------------------------

def storage():
    os.makedirs(TUNING_DIR, exist_ok=True)
    # several processes write to the same file: wait up to 60 s for the lock instead of failing
    return optuna.storages.RDBStorage(STORAGE_URL, engine_kwargs={'connect_args': {'timeout': 60}})


def load_study(name, store, worker_id=0):
    return optuna.create_study(
        study_name=name, storage=store, direction='minimize', load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42 + worker_id, n_startup_trials=5, constant_liar=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3))


def n_counted(study):
    return len(study.get_trials(deepcopy=False, states=COUNTED))


def tune(args):
    store = storage()
    if args.worker_id is None:                               # main process: prepare the studies, start the workers
        for name in args.models:
            study = load_study(name, store)
            for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
                store.set_trial_state_values(t._trial_id, TrialState.FAIL)   # left over from an interrupted run
            if not study.trials:
                study.enqueue_trial(MODELS[name][3])         # first try = current default settings
        threads = max(1, (os.cpu_count() or 1) // args.workers)
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), PYTHONIOENCODING='utf-8')
        workers = []
        for i in range(args.workers):
            log = open(f'{TUNING_DIR}/worker_{i}.log', 'a', encoding='utf-8')
            cmd = [sys.executable, '-m', 'src.utils.tuning.nn_tuning', 'tune', '--models', *args.models,
                   '--trials', str(args.trials), '--epochs', str(args.epochs), '--worker-id', str(i),
                   '--threads', str(threads)]
            workers.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env))
        print(f'{args.workers} workers x {threads} threads, logs in {TUNING_DIR}/worker_*.log')
        optuna.logging.set_verbosity(optuna.logging.WARNING)    # progress below; details in the worker logs
        seen = set()
        while any(w.poll() is None for w in workers):        # print every try as soon as it finishes
            time.sleep(60)
            seen = print_new_trials(args.models, store, seen)
        print_new_trials(args.models, store, seen)
        print('workers finished with exit codes', [w.returncode for w in workers])
        summary(args)
        return

    torch.set_num_threads(args.threads)                      # worker process: tune the models one after another
    for name in args.models:
        study = load_study(name, store, args.worker_id)
        if n_counted(study) >= args.trials:
            continue                                         # budget of this model already used up
        try:
            study.optimize(lambda t: objective(t, name, args.epochs), gc_after_trial=True,
                           callbacks=[optuna.study.MaxTrialsCallback(args.trials, states=COUNTED)])
        except Exception as e:                               # a broken model must not stop the other models
            print(f'!!! {name} stopped with an error: {e!r}', flush=True)


def print_new_trials(names, store, seen):
    for name in names:
        for t in load_study(name, store).get_trials(deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED)):
            if (name, t.number) not in seen:
                seen.add((name, t.number))
                if t.state == TrialState.PRUNED:
                    value = f'stopped at epoch {t.user_attrs.get("epochs")}'
                else:
                    value = f'val MAE {t.value:.4f}'
                print(f'{time.strftime("%H:%M")}  {name:18s} try {t.number:2d}  {value:20s} {t.params}', flush=True)
    return seen


def summary(args):
    store = storage()
    for name in args.models:
        study = load_study(name, store)
        df = study.trials_dataframe(attrs=('number', 'state', 'value', 'params', 'user_attrs'))
        if df.empty:
            continue
        df.columns = [c.removeprefix('params_').removeprefix('user_attrs_') for c in df.columns]
        df = df.rename(columns={'value': 'val_MAE'})
        df.loc[df.state == 'PRUNED', 'val_MAE'] = float('nan')   # stopped early: no final validation MAE
        df.to_csv(f'{TUNING_DIR}/{name}_trials.csv', index=False)
        print(f'\n=== {name} ({len(df)} tries, saved to {TUNING_DIR}/{name}_trials.csv)')
        print(df.round(4).to_string(index=False))
        done = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if done:
            default = study.trials[0]
            best = min(done, key=lambda t: t.value)
            print(f'default settings: val MAE {default.value}  ->  best: {best.value:.4f} (try {best.number}) {best.params}')


def final(args):
    store = storage()
    for name in args.models:
        if args.defaults:
            params, tag = MODELS[name][3], 'defaults'
        else:
            params, tag = load_study(name, store).best_params, 'tuned'
        print(f'{name}: full training with {tag} settings {params}, seed {args.seed}')
        cfg, _ = build(name, params, seed=args.seed, num_epochs=args.max_epochs,
                       ckpt_save_dir=f'{FINAL_DIR}/{name}/{tag}_seed{args.seed}')
        BasicTSLauncher.launch_training(cfg)                # early stopping (patience 5), then the test set


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('tune', help='search the best settings')
    p.add_argument('--models', nargs='+', choices=list(MODELS), default=['Linear', 'TCN', 'LSTM'])
    p.add_argument('--trials', type=int, default=10, help='tries per model (the tuning budget)')
    p.add_argument('--epochs', type=int, default=10, help='epochs per try')
    p.add_argument('--workers', type=int, default=3, help='tries trained at the same time')
    p.add_argument('--worker-id', type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument('--threads', type=int, default=None, help=argparse.SUPPRESS)
    p.set_defaults(func=tune)
    p = sub.add_parser('summary', help='table of all tries')
    p.add_argument('--models', nargs='+', choices=list(MODELS), default=list(MODELS))
    p.set_defaults(func=summary)
    p = sub.add_parser('final', help='full training with the best settings, then the test set')
    p.add_argument('--models', nargs='+', choices=list(MODELS), required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-epochs', type=int, default=100)
    p.add_argument('--defaults', action='store_true', help='use the default settings instead of the tuned ones')
    p.set_defaults(func=final)
    args = parser.parse_args()
    args.func(args)
