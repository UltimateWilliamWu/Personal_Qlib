import qlib
from qlib.workflow import R
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.backtest import backtest
import matplotlib.pyplot as plt
import pandas as pd

# =====================
# 1. 初始化 Qlib
# =====================
provider_uri = "~/.qlib/qlib_data/cn_data"
qlib.init(provider_uri=provider_uri, region="cn")


# =====================
# 2. 加载最新保存的模型
# =====================
def get_latest_model(exp_name="workflow"):
    recorders = R.list_recorders(experiment_name=exp_name)
    if not recorders:
        raise RuntimeError(f"实验 {exp_name} 下没有找到 recorder！")

    # 找到最后一个 recorder
    latest_rec_id = list(recorders.keys())[-1]
    rec = recorders[latest_rec_id]

    # 直接加载保存的模型
    model = rec.load_object("model")
    print(f"✅ 成功加载模型: recorder_id={rec.id}, experiment={rec.experiment_id}")
    return model, rec


model, rec = get_latest_model("workflow")

# =====================
# 3. 准备 2020–2025 数据集
# =====================
dh = Alpha158(start_time="2020-01-01", end_time="2025-01-01")
ds = DatasetH(handler=dh, segments={"test": ("2020-01-01", "2025-01-01")})

# =====================
# 4. 预测信号
# =====================
pred = model.predict(ds.prepare("test"))
print("预测样例：")
print(pred.head())

# =====================
# 5. 回测
# =====================
strategy = TopkDropoutStrategy(signal=pred, topk=50, n_drop=5)

report, positions = backtest(
    strategy=strategy,
    start_time="2020-01-01",
    end_time="2025-01-01",
    account=1e9,
    benchmark="SH000300",
    trade_exchange="sse",
    freq="day",
    deal_price="close",
    open_cost=0.0005,
    close_cost=0.0015
)

print("\n===== 回测结果 (2020–2025) =====")
print(report)

# =====================
# 6. 保存每日持仓
# =====================
positions_df = pd.DataFrame(positions).T
positions_df.to_csv("positions_2020_2025.csv")
print("📁 每日持仓已保存到 positions_2020_2025.csv")

# =====================
# 7. 绘制净值曲线
# =====================
nav = report["return"]
benchmark = report["bench"]

plt.figure(figsize=(12, 6))
plt.plot(nav.index, nav.values, label="Strategy NAV")
plt.plot(benchmark.index, benchmark.values, label="Benchmark")
plt.title("Strategy vs Benchmark (2020–2025)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.savefig("nav_curve_2020_2025.png")
plt.show()

print("📈 收益曲线已保存为 nav_curve_2020_2025.png")
