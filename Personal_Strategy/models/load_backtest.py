import os
import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# ✅ 配置你的工作区路径
workspace_path = os.path.expanduser("~/.qlib/qlib_data/workspace/record/LGB_Backtest_CSI800")


# ✅ 自动找到最新的 experiment_id 文件夹
def get_latest_exp_dir(workspace_path):
    exp_dirs = [os.path.join(workspace_path, d) for d in os.listdir(workspace_path) if
                os.path.isdir(os.path.join(workspace_path, d))]
    exp_dirs.sort(key=os.path.getmtime, reverse=True)
    return exp_dirs[0] if exp_dirs else None


# ✅ 加载回测分析结果
def load_backtest_results(exp_path):
    port_path = os.path.join(exp_path, "port_analysis")

    # 加载 analysis.json
    with open(os.path.join(port_path, "analysis.json"), "r") as f:
        analysis = json.load(f)

    # 加载累计收益率 DataFrame
    report_df = pickle.load(open(os.path.join(port_path, "report_normal_df.pkl"), "rb"))
    report_df["cumulative_return"] = report_df["excess_return"].cumsum()

    return analysis, report_df


# ✅ 绘图
def plot_cumulative_return(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["cumulative_return"], label="Cumulative Excess Return")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title("📈 Qlib 回测累计收益曲线")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    latest_exp = get_latest_exp_dir(workspace_path)
    if not latest_exp:
        print("❌ 没有找到任何回测结果，请先运行回测脚本！")
        exit()

    print(f"✅ 加载最新实验结果: {latest_exp}")

    analysis, report_df = load_backtest_results(latest_exp)

    # 打印指标
    print("\n📊 回测指标:")
    for key, value in analysis.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    # 绘图
    plot_cumulative_return(report_df)
