import os
import akshare as ak
import pandas as pd
from pathlib import Path
import subprocess
from datetime import datetime

# === 配置 ===
STOCK_LIST_FILE = "nasdaq_symbols.txt"  # 股票代码列表，如 AAPL、MSFT
CSV_OUTPUT_DIR = Path("C:\\Users\\22363\\.qlib\\stock_data\\source\\us_data")
QLIB_OUTPUT_DIR = Path("C:\\Users\\22363\\.qlib\\stock_data\\source\\us_data_norm")
FREQ = "day"

# ✅ 设置日期范围
START_DATE = "20200101"  # 格式：yyyyMMdd
END_DATE = "20250716"


def download_stock_data(stock_code):
    try:
        df = ak.stock_us_daily(symbol=stock_code)

        if df is None or df.empty:
            raise ValueError("抓取结果为空")

        # ✅ 将 date 字段设置为 datetime64 类型，并作为 index
        start_dt = pd.to_datetime(START_DATE)
        end_dt = pd.to_datetime(END_DATE)
        df["date"] = pd.to_datetime(df["date"])  # 确保统一为 Timestamp
        df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

        # 添加 symbol（仍保留为列）
        df["symbol"] = stock_code.upper()

        # ✅ 构造 change 列
        df["change"] = df["close"].pct_change().fillna(0)

        # ✅ 构造 factor（模拟复权因子）
        close_base = df["close"].iloc[0]
        df["factor"] = df["close"] / close_base

        # ✅ 重新排列字段顺序（注意此时 index 是 date，不需要再当列导出）

        df = df[["date", "symbol", "open", "high", "low", "close", "volume", "change", "factor"]]

        return df

    except Exception as e:
        print(f"❌ 下载失败：{stock_code}，原因：{e}")
        return None


def save_all_to_csv():
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(STOCK_LIST_FILE, "r") as f:
        stock_list = [line.strip() for line in f if line.strip()]
    for code in stock_list:
        df = download_stock_data(code)
        if df is not None and not df.empty:
            df.to_csv(CSV_OUTPUT_DIR / f"{code}.csv", index=False)
            print(f"✅ 保存成功：{code}")
        else:
            print(f"⚠️ 无数据或失败：{code}")


def run_dump_bin():
    cmd = [
        "python", "../../dump_bin.py", "dump_all",
        "--csv_path", str(CSV_OUTPUT_DIR),
        "--qlib_dir", str(QLIB_OUTPUT_DIR),
        "--freq", FREQ,
        "--exclude_fields", "date,symbol"
    ]
    subprocess.run(cmd)


def run_normalize():
    cmd = [
        "python", "collector.py", "normalize_data",
        "--qlib_data_1d_dir", str(Path("~/.qlib/qlib_data/us_data").expanduser()),
        "--source_dir", str(CSV_OUTPUT_DIR.resolve()),
        "--normalize_dir", str(QLIB_OUTPUT_DIR.resolve()),
        "--region", "US",
        "--interval", "1d"
    ]
    subprocess.run(cmd)


if __name__ == "__main__":
    print(f"📅 下载日期范围：{START_DATE} 到 {END_DATE}")
    save_all_to_csv()
    print("🧪 开始执行 Qlib 归一化 normalize_data 步骤...")
    run_normalize()
    print("📦 开始执行 dump_bin 转换为 Qlib 二进制格式...")
    run_dump_bin()
    print("✅ 全部完成！Qlib 数据保存在：", QLIB_OUTPUT_DIR.resolve())
