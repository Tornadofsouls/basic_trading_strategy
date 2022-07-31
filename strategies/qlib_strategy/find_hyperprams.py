
import qlib
import pandas as pd
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict
import optuna
from myalpha158 import MyAlpha158
import fire

def get_dataset():
  data_handler_config = {
      "start_time": "2001-01-01",
      "end_time": "2018-01-01",
      "fit_start_time": "2001-01-01",
      "fit_end_time": "2014-01-01",
      "instruments": "csi300",
      "benchmark": "SH000300"
  }    
  dataset_config = {
      "class": "DatasetH",
      "module_path": "qlib.data.dataset",
      "kwargs": {
          "handler": MyAlpha158(**data_handler_config),
          "segments": {
              "train": (data_handler_config["fit_start_time"], data_handler_config["fit_end_time"]),
              "valid": ("2014-01-01", "2018-01-01"),
          },
      },
  }

  return init_instance_by_config(dataset_config)

dataset = None

def objective(trial):
    task = {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                #"colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1),
                "learning_rate": trial.suggest_uniform("learning_rate", 0, 1),
                #"subsample": trial.suggest_uniform("subsample", 0, 1),
                "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 1e4),
                "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 1e4),
                "max_depth": 10,
                "num_leaves": trial.suggest_int("num_leaves", 2, 1024),
                "feature_fraction": trial.suggest_uniform("feature_fraction", 0.4, 1.0),
                "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
                #"min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            },
        },
    }
    evals_result = dict()
    model = init_instance_by_config(task["model"])
    model.fit(dataset, evals_result=evals_result)
    return min(evals_result["valid"]["l2"])

def optimize_model_hyperparam():
  global dataset

  provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
  if not exists_qlib_data(provider_uri):
      print(f"Qlib data is not found in {provider_uri}")
      return
  qlib.init(provider_uri=provider_uri, region=REG_CN)

  dataset = get_dataset()
  
  optuna.create_study(study_name="LGBM_158_with_benchmark", direction="minimize", storage="sqlite:///optuna.db.sqlite3", load_if_exists=True)
  study = optuna.Study(study_name="LGBM_158_with_benchmark", storage="sqlite:///optuna.db.sqlite3")
  study.optimize(objective, n_trials=2000)

if __name__ == "__main__":
  fire.Fire(optimize_model_hyperparam)