from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
# from qlib.contrib.data.handler import Alpha360
from qlib.data.dataset import DatasetH
from qlib.data import D
import pandas as pd
import qlib
import os
import joblib

label = ["Ref($close, -5)/$close - 1"]
qlib_url = "C:\\Users\\22363\\.qlib\\qlib_data\\cn_data"
instruments_para = "csi800"
MODEL_PATH = "results/lgb_model_CSI800.pkl"
start_date = "2020-01-01"
end_date = "2025-09-03"
predict_date = "2025-09-04"
region = "cn"


def get_signals_for_stocks(pred_signal: pd.Series, stock_codes: list[str]) -> pd.DataFrame:
    instruments = pred_signal.index.get_level_values(1)
    signals_df = pd.DataFrame({
        "stock": instruments,
        "signal": pred_signal.values
    }).drop_duplicates("stock")

    result = signals_df[signals_df["stock"].isin(stock_codes)].copy()
    missing = set(stock_codes) - set(result["stock"])
    if missing:
        print(f"\u26a0\ufe0f 以下股票未出现在预测结果中：{missing}")
        missing_df = pd.DataFrame({"stock": list(missing), "signal": [None] * len(missing)})
        result = pd.concat([result, missing_df], ignore_index=True)

    return result.sort_values("stock").reset_index(drop=True)


def check_feature_data_exists(handler, feature_date: str):
    dates_in_data = handler._data.index.get_level_values(0).unique()
    if pd.Timestamp(feature_date) not in dates_in_data:
        raise ValueError(f"\u26a0\ufe0f Alpha360 中没有 {feature_date} 的特征数据！")


def get_prev_trading_date(predict_date: str) -> str:
    calendar = D.calendar(end_time=predict_date, freq="day")
    if len(calendar) < 2:
        raise ValueError(f"❌ 无法获取 {predict_date} 的前一个交易日")
    if pd.Timestamp(predict_date) == calendar[-1]:
        return calendar[-2]
    return calendar[-1]


def train_model(model_path: str):
    handler = Alpha158(
        start_time=start_date,
        end_time=end_date,
        instruments=instruments_para,
        fit_start_time=start_date,
        fit_end_time=end_date,
        label=label
    )
    handler.fetch()
    dataset = DatasetH(handler=handler, segments={"train": (start_date, end_date)})

    model = LGBModel(
        loss="mse",
        max_depth=7,
        num_leaves=127,
        n_estimators=3000,
        subsample=0.8,
        colsample_bytree=0.8,
        learning_rate=0.03,
        lambda_l1=5.0,
        lambda_l2=10.0,
        num_threads=8,
        device="gpu",
        early_stopping_rounds=50,
        verbose=-1,
    )
    model.fit(dataset)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if os.path.exists(model_path):
        os.remove(model_path)  # 安全覆盖旧模型
    joblib.dump(model, model_path)
    print(f"✅ 模型已保存至: {model_path}")
    return model


def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 模型文件不存在：{model_path}")
    model = joblib.load(model_path)
    print(f"✅ 成功加载模型：{model_path}")
    return model


def predict_positions_for_date(predict_date: str, topk: int, model_path: str):
    feature_date = get_prev_trading_date(predict_date)

    infer_handler = Alpha158(
        start_time=feature_date,
        end_time=feature_date,
        instruments=instruments_para,
        fit_start_time=start_date,
        fit_end_time=end_date,
        label=label
    )
    infer_handler.fetch()
    check_feature_data_exists(infer_handler, feature_date)
    infer_dataset = DatasetH(handler=infer_handler, segments={"infer": (feature_date, feature_date)})

    model = load_model(model_path)
    pred_signal = model.predict(infer_dataset, "infer")

    if pred_signal.empty:
        raise ValueError(f"⚠\ufe0f 模型输出为空，无法为 {predict_date} 生成持仓建议")

    top_stocks = pred_signal.sort_values(ascending=False).head(topk)
    weight = 1.0 / len(top_stocks)
    df_result = pd.DataFrame({
        "stock": [idx[-1] for idx in top_stocks.index],
        "weight": weight,
        "signal": top_stocks.values,
        "buy_date": predict_date
    }).reset_index(drop=True)

    return df_result, pred_signal


if __name__ == "__main__":
    qlib.init(provider_uri=qlib_url, region=region)
    # ✅ 训练模型一次（如已训练可跳过）
    # if not os.path.exists(MODEL_PATH):
    #     train_model(model_path=MODEL_PATH)
    train_model(model_path=MODEL_PATH)

    # ✅ 使用保存模型进行预测
    df_0414, pred_signal = predict_positions_for_date(predict_date, topk=20, model_path=MODEL_PATH)
    print(df_0414.head(20))

    # ✅ 查询指定股票的 signal
    stock_list = ["SZ002008", "SZ300418", "SZ002230", "SZ002050", "SZ300866", "SZ300458"]
    signal_df = get_signals_for_stocks(pred_signal, stock_list)
    print(f"\n📊 查询股票的预测 signal:\n{signal_df}")
