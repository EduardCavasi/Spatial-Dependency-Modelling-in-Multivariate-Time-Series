import optuna
from typing import Dict

from src.model.baseline_temporal_models.lgbm import LGBM, LGBMConfig

def objective(trial: optuna.Trial):
    max_depth = trial.suggest_int(
        "max_depth", 3, 10
    )

    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators", 200, 1000, step=200
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.1, log=True
        ),

        "max_depth": max_depth,

        "num_leaves": trial.suggest_int(
            "num_leaves", 8, min(64, 2 ** max_depth)
        ),

        "min_child_samples": trial.suggest_int(
            "min_child_samples", 100, 3000, log=True
        ),

        "feature_fraction": trial.suggest_float(
            "feature_fraction", 0.7, 1.0
        ),

        "bagging_fraction": trial.suggest_float(
            "bagging_fraction", 0.7, 1.0
        ),

        "bagging_freq": trial.suggest_int(
            "bagging_freq", 1, 5
        ),

        "lambda_l1": trial.suggest_float(
            "lambda_l1", 1e-5, 10, log=True
        ),

        "lambda_l2": trial.suggest_float(
            "lambda_l2", 1e-5, 100, log=True
        ),

        "min_gain_to_split": trial.suggest_float(
            "min_gain_to_split", 0.0, 1.0
        ),

        "max_bin": trial.suggest_categorical(
            "max_bin", [63, 127]
        ),
    }

    return train_and_evaluate(params)

def train_and_evaluate(params: Dict):
    path="data/METR-LA"
    lgbm = LGBM(LGBMConfig(**params, use_tod=True))
    lgbm.fit(*lgbm.windows(path, "train"), *lgbm.windows(path, "val"))
    results = lgbm.evaluate(*lgbm.windows(path, "val"))
    return results["all"]["MAE"]

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42)
)

study.optimize(
    objective, n_trials=10
)

best_params = study.best_params
print(best_params)
best_model = LGBM(LGBMConfig(**best_params, use_tod=True))
best_model.run(folder_name="checkpoints/LGBM/tuning")