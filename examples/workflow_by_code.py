#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData
from qlib.tests.config import CSI300_BENCH, CSI300_GBDT_TASK


if __name__ == "__main__":
    # =====================
    # 1. 初始化数据和 Qlib
    # =====================
    provider_uri = "~/.qlib/qlib_data/cn_data"  # 数据路径
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    # =====================
    # 2. 初始化模型和数据集
    # =====================
    model = init_instance_by_config(CSI300_GBDT_TASK["model"])
    dataset = init_instance_by_config(CSI300_GBDT_TASK["dataset"])

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
                "signal": (model, dataset),
                "topk": 50,
                "n_drop": 5,
            },
        },
        "backtest": {
            "start_time": "2017-01-01",
            "end_time": "2020-08-01",
            "account": 100000000,
            "benchmark": CSI300_BENCH,
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }

    # =====================
    # 3. 开始实验
    # =====================
    with R.start(experiment_name="workflow"):
        # 记录配置
        R.log_params(**flatten_dict(CSI300_GBDT_TASK))

        # 训练模型
        model.fit(dataset)

        # 🔑 保存训练好的模型
        R.save_objects(model=model)

        # 保存预测信号
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # 保存信号分析
        sar = SigAnaRecord(recorder)
        sar.generate()

        # 保存回测结果
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()
