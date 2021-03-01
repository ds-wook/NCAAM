from optuna.trial import Trial
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from data.dataset import tourney
from data.fea_eng import train_test_dataset


tourney, test = train_test_dataset(tourney, stage=True)

X = tourney.drop(["Season", "TeamID1", "TeamID2", "result"], axis=1)
y = tourney["result"]
s = tourney["Season"]

X_test = test.drop(["ID", "Season", "TeamID1", "TeamID2"], axis=1)


def objective(trial: Trial) -> float:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 20000,
        "learning_rate": 0.05,
        "random_state": 42,
        "num_leaves": trial.suggest_int("num_leaves", 10, 100),
        # "max_depth": trial.suggest_int("max_depth", 3, 12),
        # "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 1.0),
        # "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.4, 1.0),
        "subsample": trial.suggest_uniform("subsample", 0.4, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
        # "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
    }
    model = LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=100,
        verbose=False,
    )
    preds = model.predict_proba(X_valid, num_iteration=model.best_iteration_)[:, 1]
    loss = log_loss(y_valid, preds)
    return loss
