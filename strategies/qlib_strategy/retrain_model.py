import qlib
from qlib.workflow import R
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
import pickle
import os
from pathlib import Path
from myalpha158 import MyAlpha158
from my_model import MyEnsembleModel
import fire

def retrain_model(model_path = "trained_model", fit_end_time="2022-08-01"):
    market = "csi300"
    benchmark = "SH000300"

    data_handler_config = {
        "start_time": "2006-01-01",
        "end_time": fit_end_time,
        "fit_start_time": "2008-01-01",
        "fit_end_time": fit_end_time,
        "instruments": "csi300",
        "benchmark": "SH000300",
        "days_ahead": 3
    }

    dataset_config = {
      "class": "DatasetH",
      "module_path": "qlib.data.dataset",
      "kwargs": {
          "handler": MyAlpha158(**data_handler_config),
          "segments": {
              "train": (data_handler_config["fit_start_time"], data_handler_config["fit_end_time"]),
              "valid": ("2005-08-01", "2007-12-31"),
          },
      },
    }

    handler = MyAlpha158(**data_handler_config)

    recorder_name = "retrain_model"
    with R.start(recorder_name=recorder_name, experiment_name="train_model"):
        # model initiaiton
        model = MyEnsembleModel(
            num_models= 3,
            enable_sr= False,
            enable_fs= True,
            decay= 0.5,
                    
            loss="mse",
            colsample_bytree=0.8879,
            learning_rate=0.0421,
            subsample=0.8789,
            lambda_l1=205.6999,
            lambda_l2=580.9768,
            max_depth=8,
            num_leaves=210,
            num_threads=20,
        )
        dataset = init_instance_by_config(dataset_config)
        model.fit(dataset)
        R.save_objects(trained_model=model)

        with Path(model_path).open("wb") as f:
            print("Saving model to", model_path)
            pickle.dump(model, f, protocol=4)
            print("Complete saving model")

def retrain_model_using_all_data(model_path = "trained_model"):
    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    if not exists_qlib_data(provider_uri):
        print(f"Qlib data is not found in {provider_uri}")
        return
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    retrain_model(model_path)

if __name__ == "__main__":
    fire.Fire(retrain_model_using_all_data)