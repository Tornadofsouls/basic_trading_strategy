
import qlib
import pandas as pd
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict
import optuna
from optuna.samplers import RandomSampler
from myalpha158 import MyAlpha158
import fire
import time

dataset_cache = {}

def get_dataset(days_ahead=2):
  data_handler_config = {
      "start_time": "2001-01-01",
      "end_time": "2014-01-05",
      "instruments": "csi300",
      "benchmark": "SH000300",
      "days_ahead": days_ahead
  }
  data_handler_config["fit_start_time"] = data_handler_config["start_time"]
  data_handler_config["fit_end_time"] = "2014-01-01"
  dataset_config = {
      "class": "DatasetH",
      "module_path": "qlib.data.dataset",
      "kwargs": {
          "handler": MyAlpha158(**data_handler_config),
          "segments": {
              "train": (data_handler_config["fit_start_time"], data_handler_config["fit_end_time"]),
              "valid": (data_handler_config["fit_end_time"], data_handler_config["end_time"]),
          },
      },
  }

  return init_instance_by_config(dataset_config)

def get_model_ir(model, model_params):
    test_data_handler_config = {
        "start_time": "2014-01-01",
        "end_time": "2020-01-01",
        "instruments": "csi300",
        "benchmark": "SH000300",
        "days_ahead": 2
    }
    test_data_handler_config["fit_start_time"] = test_data_handler_config["start_time"]
    test_data_handler_config["fit_end_time"] = test_data_handler_config["end_time"]

    cache_key = f"test"
    if cache_key not in dataset_cache:
        test_handler = MyAlpha158(**test_data_handler_config)
        test_dataset = init_instance_by_config({
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": test_handler,
                    "segments": {
                        "test": (test_data_handler_config["start_time"], test_data_handler_config["end_time"]),
                        
                    },
                },
            })
        dataset_cache[cache_key] = test_dataset
    test_dataset = dataset_cache[cache_key]

    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "model": model,
                "dataset": test_dataset,
                "topk": 20,
                "n_drop": 20
            },
        },
        "backtest": {
            "start_time": test_data_handler_config["start_time"],
            "end_time": test_data_handler_config["end_time"],
            "account": 5000000,
            "benchmark": "SH000300",
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.195,
                "deal_price": "open",
                "open_cost": 0.0002,
                "close_cost": 0.0012,
                "min_cost": 5,
            },
        },
    }

    # backtest and analysis
    with R.start(experiment_name="backtest_analysis"):
        R.log_params(**flatten_dict(model_params))

        # prediction
        recorder = R.get_recorder()
        ba_rid = recorder.id
        sr = SignalRecord(model, test_dataset, recorder)
        sr.generate()

        # backtest & analysis
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()

    analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

    return analysis_df.loc[('excess_return_with_cost', 'information_ratio')]["risk"]

def objective(trial):
    global dataset_cache

    days_ahead = 2

    cache_key = f"train_{days_ahead}"
    if cache_key not in dataset_cache:
        dataset_cache[cache_key] = get_dataset(days_ahead)
    dataset = dataset_cache[cache_key]

    enable_sr = trial.suggest_categorical("enable_sr", [True, False])
    alpha1 = 1
    alpha2 = 1
    if enable_sr:
        alpha1 = trial.suggest_uniform("alpha1", 0, 1)
        alpha2 = trial.suggest_uniform("alpha2", 0, 1)

    task = {
        "model": {
            "class": "DEnsembleModel",
            "module_path": "qlib.contrib.model.double_ensemble",
            "kwargs": {
                "num_models": trial.suggest_int("num_models", 3, 8),
                "enable_sr": enable_sr,
                "enable_fs": trial.suggest_categorical("enable_fs", [True, False]),
                "alpha1": alpha1,
                "alpha2": alpha2,
                "decay": 0.5,
                
                'bagging_fraction': 0.7108329708514702, 
                'bagging_freq': 2, 
                'feature_fraction': 0.4977241113173993, 
                'lambda_l1': 1.0274619243959457, 
                'lambda_l2': 80.74475219018196, 
                'learning_rate': 0.015541749212165102, 
                'min_data_in_leaf': 34, 
                'num_leaves': 235,
                'epochs': 356,
                "max_depth": 10
            },
        },
        "days_ahead": days_ahead
    }
    print("Trying params:", task)
    model = init_instance_by_config(task["model"])
    model.fit(dataset)
    return get_model_ir(model, task)

def optimize_model_hyperparam(study_name="model_params_search", storage="sqlite:///params_storage/optuna.db.sqlite3"):
  optuna.create_study(study_name=study_name, direction="maximize", storage=storage, load_if_exists=True, sampler=RandomSampler(seed=int(time.time())))
  study = optuna.Study(study_name=study_name, storage=storage)

  # use default data
  # NOTE: need to download data from remote: python scripts/get_data.py qlib_data_cn --target_dir ~/.qlib/qlib_data/cn_data
  provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
  if not exists_qlib_data(provider_uri):
      print(f"Qlib data is not found in {provider_uri}")
      sys.path.append(str(scripts_dir))
      from get_data import GetData
      GetData().qlib_data(target_dir=provider_uri, region=REG_CN)
  qlib.init(provider_uri=provider_uri, region=REG_CN)

  study.optimize(objective, n_trials=500)

if __name__ == "__main__":
  fire.Fire(optimize_model_hyperparam)