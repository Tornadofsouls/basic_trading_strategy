from qlib.contrib.data.handler import Alpha158, _DEFAULT_LEARN_PROCESSORS, DataHandlerLP

class MyAlpha158(Alpha158):
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processor=None,
        benchmark = "SH000300",
        days_ahead = 4,
        **kwargs,
    ):
        self.benchmark = benchmark
        self.days_ahead = days_ahead
        super().__init__(instruments, start_time, end_time, freq, infer_processors, learn_processors, fit_start_time, fit_end_time, process_type, filter_pipe, inst_processor, **kwargs)
        

    def get_feature_config(self):
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"],
            },
            "volume": {
                'windows': [0, 1, 2, 3, 4,10,20,30,60]
            },
            "rolling": {},
        }
        normal_fields, normal_names = self.parse_config_to_fields(conf)

        benchmark_diff_fields = ["ChangeInstrument('{}', {}) - ({})".format(self.benchmark, field, field) for field in normal_fields ]
        benchmark_diff_names = ["benchmark_diff_{}_{}".format(self.benchmark, name) for name in normal_names ]
        
        return normal_fields + benchmark_diff_fields , normal_names + benchmark_diff_names 

    def get_label_config(self):
        return (["Ref($open, -{})/Ref($open, -1) - 1".format(self.days_ahead)], ["LABEL0"])