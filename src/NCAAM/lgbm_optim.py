import argparse

import optuna
import joblib

from optim.optuna_optim import objective


if __name__ == "__main__":
    parse = argparse.ArgumentParser("Optimize!")
    parse.add_argument("--n_trials", type=int, default=50)
    parse.add_argument("--name", type=str, default="params.pkl")
    args = parse.parse_args()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials)
    print("Best Score:", study.best_value)
    print("Best trial:", study.best_trial.params)
    joblib.dump(study.best_trial.params, "../../params/" + args.name)
