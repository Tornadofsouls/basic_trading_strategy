import qlib
from qlib.workflow import R
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
import pickle
import os
from pathlib import Path
from myalpha158 import MyAlpha158
import fire

def retrain_model_using_with_iteration(model_path = "trained_model", iteration=None):
    print("training model with iteration", iteration)

    market = "csi300"
    benchmark = "SH000300"

    data_handler_config = {
        "start_time": "2000-01-01",
        "end_time": "2025-08-01",
        "fit_start_time": "2000-01-01",
        "fit_end_time": "2014-01-01",
        "instruments": market,
        "benchmark": benchmark,
        "days_ahead": 4
    }

    handler = MyAlpha158(**data_handler_config)

    params = {'bagging_fraction': 0.7108329708514702, 'bagging_freq': 2, 'feature_fraction': 0.4977241113173993, 
              'lambda_l1': 1.0274619243959457, 'lambda_l2': 80.74475219018196, 'learning_rate': 0.015541749212165102, 
              'min_data_in_leaf': 34, 'num_leaves': 235}
    
    task = {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "learning_rate": params["learning_rate"],
                "lambda_l1": params["lambda_l1"],
                "lambda_l2": params["lambda_l2"],
                "max_depth": 10,
                "num_leaves": params["num_leaves"],
                "feature_fraction": params["feature_fraction"],
                "bagging_fraction": params["bagging_fraction"],
                "bagging_freq": params["bagging_freq"],
                "min_data_in_leaf": params["min_data_in_leaf"],
                "num_boost_round": 1000 if iteration is None else iteration
            },
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": handler,
                "segments": {
                    "train": (data_handler_config["start_time"], data_handler_config["end_time"]),
                    "valid": (data_handler_config["end_time"], data_handler_config["end_time"]),
                },
            },
        },
    }

    if iteration is None:
        task["dataset"]["kwargs"]["segments"]["train"] = (data_handler_config["fit_start_time"], data_handler_config["fit_end_time"])
        task["dataset"]["kwargs"]["segments"]["valid"] = (data_handler_config["fit_end_time"], data_handler_config["end_time"])

    recorder_name = "recorder{}".format(iteration)
    with R.start(recorder_name=recorder_name, experiment_name="train_model"):
        # model initiaiton
        model = init_instance_by_config(task["model"])
        dataset = init_instance_by_config(task["dataset"])
        if iteration:
            model.fit(dataset, early_stopping_rounds=iteration)
        else:
            model.fit(dataset)
        R.save_objects(trained_model=model)

    if iteration is None:
        # Train with another round with current iteration but full time range
        curr_iter = model.model.current_iteration()
        del model
        del dataset
        retrain_model_using_with_iteration(model_path = model_path, iteration=curr_iter)
    else:
        with Path(model_path).open("wb") as f:
            print("Saving model to", model_path, "iteration", iteration)
            pickle.dump(model, f, protocol=4)
            print("Complete saving model")

def retrain_model_using_all_data(model_path = "trained_model"):
    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    if not exists_qlib_data(provider_uri):
        print(f"Qlib data is not found in {provider_uri}")
        return
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    retrain_model_using_with_iteration(model_path)

if __name__ == "__main__":
    fire.Fire(retrain_model_using_all_data)