import os
import akshare as ak
import pandas as pd
from pathlib import Path
import subprocess
from datetime import datetime

# === 配置 ===
STOCK_LIST_FILE = "csi800.txt"  # 股票代码列表（如 SH600000）
CSV_OUTPUT_DIR = Path("C:\\Users\\22363\\.qlib\\stock_data\\source\\cn_data_CSI800")  # 输出 csv 文件目录
NORMAL_OUTPUT_DIR = Path("C:\\Users\\22363\\.qlib\\stock_data\\source\\cn_data_CSI800_norm")
QLIB_OUTPUT_DIR = Path("C:\\Users\\22363\\.qlib\\qlib_data\\cn_data_CSI800")  # 输出 bin 格式目录
FREQ = "day"

# ✅ 你可以在这里设置日期范围
START_DATE = "20200101"  # 格式必须是 yyyyMMdd
END_DATE = "20250808"


def download_stock_data(stock_code):
    formatted_code = stock_code
    try:
        df = ak.stock_zh_a_hist(
            symbol=formatted_code,
            period="daily",
            start_date=START_DATE,
            end_date=END_DATE,
            adjust="qfq"  # 前复权，才能构造正确 factor
        )

        if df is None or df.empty:
            raise ValueError("抓取结果为空")

        # 检查字段完整性
        required_cols = {"日期", "开盘", "最高", "最低", "收盘", "成交量"}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"{stock_code} 缺失必要字段，跳过")

        # 重命名
        df.rename(columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume"
        }, inplace=True)

        # 加字段
        df["symbol"] = stock_code
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        # ✅ 构造 change 列：今日收盘 vs 昨日收盘
        df["change"] = df["close"].pct_change()
        df["change"] = df["change"].fillna(0)

        # ✅ 构造 factor：用收盘价与复权价比例替代（akshare 已为前复权）
        # 用原始收盘价 / 当前复权价（可理解为 factor 累计调整因子）
        close_base = df["close"].iloc[0]
        df["factor"] = df["close"] / close_base

        # 保留字段并排序
        df = df[["symbol", "date", "open", "high", "low", "close", "volume", "change", "factor"]]
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
        "--qlib_data_1d_dir", str(QLIB_OUTPUT_DIR.expanduser()),
        "--source_dir", str(CSV_OUTPUT_DIR.resolve()),
        "--normalize_dir", str(NORMAL_OUTPUT_DIR.resolve()),
        "--region", "CN",
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
