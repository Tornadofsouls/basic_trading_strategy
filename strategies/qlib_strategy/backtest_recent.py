import qlib
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.contrib.report import analysis_model, analysis_position
import pickle
import os
from pathlib import Path
from myalpha158 import MyAlpha158
import fire
import requests
import json

def send_notification_log(title, text):
    print("Logging to azure", title, text)
    url = "https://di-trading-log.azurewebsites.net/api/log_event?code=gMcbj7J1vKh/VCs8e2MkVaHRp/4NLz8dttpgk03p8SMKcJQHo/8JKQ=="
    data = {
        "title" : title,
        "desp" : text,
    }
    try_cnt = 0
    while try_cnt < 5:
        try:
            response = requests.post(url, data=json.dumps(data))
            print(response.content)
            return
        except Exception as e:
            print(e)
            try_cnt += 1

def run_backtest_using_model(model_path = "trained_model", output_directory = "./", start_date = "2022-07-01", end_date = "2055-07-22"):
  # Init data
  provider_uri = "~/.qlib/qlib_data/cn_data"
  if not exists_qlib_data(provider_uri):
      raise Exception(f"Qlib data is not found in {provider_uri}")
  qlib.init(provider_uri=provider_uri, region=REG_CN)

  # Prepare dataset 
  data_handler_config = {
    "start_time": start_date,
    "end_time": end_date,
    "fit_start_time": start_date,
    "fit_end_time": start_date,
    "instruments": "csi300",
    "benchmark": "SH000300"
  }

  dataset_config =  {
    "class": "DatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {
        "handler": MyAlpha158(**data_handler_config),
        "segments": {
            "test": (start_date, end_date),
        },
    },
  }
  dataset = init_instance_by_config(dataset_config)

  with Path(model_path).open("rb") as f:
    model = pickle.Unpickler(f).load()

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
            "dataset": dataset,
            "topk": 10,
            "n_drop": 10
        },
    },
    "backtest": {
        "start_time": data_handler_config["start_time"],
        "end_time": data_handler_config["end_time"],
        "account": 500000,
        "benchmark": data_handler_config["benchmark"],
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

  with R.start(experiment_name="backtest_analysis"):
    recorder = R.get_recorder()
    ba_rid = recorder.id
    sr = SignalRecord(model, dataset, recorder)
    sr.generate()

    # backtest & analysis
    par = PortAnaRecord(recorder, port_analysis_config, "day")
    par.generate()

    report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    my_report_df = analysis_position.report._calculate_report_data(report_normal_df)[["cum_bench", "cum_return_w_cost", "cum_ex_return_w_cost"]]
    my_report_df["cum_bench"] = my_report_df["cum_bench"] * 100
    my_report_df["cum_return_w_cost"] = my_report_df["cum_return_w_cost"] * 100
    my_report_df["cum_ex_return_w_cost"] = my_report_df["cum_ex_return_w_cost"] * 100

    output_file_name = end_date + ".csv"
    output_file_path = os.path.join(output_directory, output_file_name)
    my_report_df.to_csv(output_file_path)

    title = f"{end_date} return"

    send_notification_log(title, str(my_report_df))

if __name__ == "__main__":
  fire.Fire(run_backtest_using_model)