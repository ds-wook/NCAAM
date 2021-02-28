from typing import Any
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss


def group_kfold_model(
    model: Any,
    n_fold: int,
    group: pd.Series,
    train: pd.DataFrame,
    target: pd.Series,
    test: pd.DataFrame,
) -> np.ndarray:
    folds = GroupKFold(n_splits=n_fold, random_state=42)
    splits = folds.split(train, target, group)
    y_preds = np.zeros(test.shape[0])
    oof_preds = np.zeros(train.shape[0])

    for fold_n, (train_index, valid_index) in enumerate(splits):
        model_name = model.__class__.__name__
        X_train, X_valid = train.iloc[train_index], train.iloc[valid_index]
        y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]
        evals = [(X_train, y_train), (X_valid, y_valid)]
        if model_name == "LGBMClassifier":
            model.fit(
                X_train,
                y_train,
                eval_set=evals,
                early_stopping_rounds=100,
                eval_metric="logloss",
                verbose=20,
            )
            oof_preds[valid_index] = model.predict_proba(X_valid)[:, 1]
            y_preds += model.predict_proba(test)[:, 1] / n_fold
        else:
            model.fit(
                X_train,
                y_train,
                eval_set=evals,
                early_stopping_rounds=100,
                verbose=100,
            )
            oof_preds[valid_index] = model.predict(X_valid)
            y_preds += model.predict(test) / n_fold
        del X_train, X_valid, y_train, y_valid
    scores = log_loss(target, oof_preds)
    print(f"OOF Score: {scores: .5f}")
    return y_preds
