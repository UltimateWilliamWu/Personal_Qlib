import qlib
import pandas as pd
from qlib.config import REG_CN
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH

from qlib.contrib.strategy.signal_strategy import BaseSignalStrategy
from qlib.backtest.decision import TradeDecisionWO, Order


# ✅ 自定义每日换仓策略：每日买入 topk，自动触发上日清仓
class DailyTopkStrategy(BaseSignalStrategy):
    def __init__(self, model, dataset, topk=5, **kwargs):
        signal = (model, dataset)
        super().__init__(signal=signal, **kwargs)
        self.topk = topk

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)

        if isinstance(score, pd.DataFrame):
            score = score.iloc[:, 0]
        if score is None or score.empty:
            return TradeDecisionWO([], self)

        topk_stocks = score.sort_values(ascending=False).head(self.topk)
        weight = 1.0 / self.topk
        orders = []

        for instrument in topk_stocks.index:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=instrument,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.BUY,
            ):
                continue

            deal_price = self.trade_exchange.get_deal_price(
                instrument, trade_start_time, trade_end_time, Order.BUY
            )
            cash = self.trade_position.get_cash()
            amount = cash * self.get_risk_degree() * weight / deal_price
            factor = self.trade_exchange.get_factor(instrument, trade_start_time, trade_end_time)
            amount = self.trade_exchange.round_amount_by_trade_unit(amount, factor)

            order = Order(
                stock_id=instrument,
                amount=amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.BUY,
            )
            orders.append(order)

        return TradeDecisionWO(orders, self)


# ✅ 初始化 Qlib（使用你本地数据路径）
qlib.init(provider_uri="C:/Users/22363/.qlib/qlib_data/cn_data", region=REG_CN)

# ✅ 数据与模型参数
benchmark = "SH000300"
instruments = "csi800"
experiment_name = "LGB_Backtest_CSI800"
model_experiment = "train_model"

# ✅ 构建数据集
handler = Alpha158(
    instruments=instruments,
    start_time="2017-01-01",
    end_time="2025-05-12",
    infer_processors=[
        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "alpha"}},
        {"class": "Fillna", "kwargs": {"fields_group": "alpha"}},
    ]
)
dataset = DatasetH(
    handler=handler,
    segments={
        "train": ("2017-01-01", "2023-12-31"),
        "valid": ("2024-01-01", "2024-06-30"),
        "test": ("2024-07-01", "2025-05-12"),
    }
)

# ✅ 回测配置（使用 DailyTopkStrategy）
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
        "class": DailyTopkStrategy,
        "kwargs": {
            "model": None,  # 稍后注入
            "dataset": dataset,
            "topk": 5,
        },
    },
    "backtest": {
        "start_time": "2024-07-01",
        "end_time": "2025-05-12",
        "account": 1000000,
        "benchmark": benchmark,
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

# ✅ 回测流程执行
with R.start(experiment_name="backtest_analysis"):
    # 加载已训练模型
    recorder = R.get_recorder(recorder_id=None, experiment_name=model_experiment)
    model = recorder.load_object("trained_model")

    # 注入模型
    port_analysis_config["strategy"]["kwargs"]["model"] = model

    # 预测信号生成
    recorder = R.get_recorder()
    ba_rid = recorder.id
    sr = SignalRecord(model, dataset, recorder)
    sr.generate()

    # 回测与分析
    par = PortAnaRecord(recorder, port_analysis_config, "day")
    par.generate()
