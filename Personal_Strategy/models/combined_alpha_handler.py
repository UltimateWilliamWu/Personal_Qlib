from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.loader import Alpha158DL


class CombinedAlphaHandler(DataHandlerLP):
    def __init__(
            self,
            alpha101_expr_list=None,
            instruments="csi500",
            start_time=None,
            end_time=None,
            freq="day",
            process_type=DataHandlerLP.PTYPE_A,
            infer_processors=None,
            learn_processors=None,
            **kwargs
    ):
        infer_processors = infer_processors or []
        learn_processors = learn_processors or []
        self.alpha158_expr = Alpha158DL.get_feature_config({
            "kbar": {},
            "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW", "VWAP"]},
            "rolling": {}
        })
        self.alpha101_expr = alpha101_expr_list if alpha101_expr_list else self._get_default_alpha101_expr()

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    # "label": label  # ✅ 放这里，而不是直接传入 __init__
                    "label": (["Ref($close, -5)/$close - 1"], ["LABEL0"])
                },
                "freq": freq,
            },
        }

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs
        )

    def get_feature_config(self):
        return list(self.alpha158_expr) + self.alpha101_expr

    def _get_default_alpha101_expr(self):
        return [
            "-1 * correlation(rank(open), rank(volume), 10)",
            "rank(close - open) * volume",
            "-1 * ts_rank(rank(low), 9)",
            "-(close - open) / ((high - low) + 0.001)",
            "-1 * rank((close - open) / ((high - low) + 0.001))"
        ]
