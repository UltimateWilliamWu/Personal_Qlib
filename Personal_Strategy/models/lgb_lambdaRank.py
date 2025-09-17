from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data import D
import pandas as pd
import numpy as np
import qlib
import os
import joblib
import lightgbm as lgb
from typing import Tuple
import optuna
from optuna.samplers import TPESampler

label = ["Ref($close, -5)/$close - 1"]
qlib_url = "C:\\Users\\22363\\.qlib\\qlib_data\\cn_data"
instruments_para = "csi800"  # 建议先用 csi300 跑通，再换 csi800
MODEL_PATH = "results/lgb_model_CSI800_ltr.pkl"
start_date = "2020-01-01"
end_date = "2025-09-17"
predict_date = "2025-09-18"
region = "cn"


# -------- 工具函数 --------
def get_prev_trading_date(predict_date: str) -> str:
    calendar = D.calendar(end_time=predict_date, freq="day")
    if len(calendar) < 2:
        raise ValueError(f"❌ 无法获取 {predict_date} 的前一个交易日")
    if pd.Timestamp(predict_date) == calendar[-1]:
        return calendar[-2]
    return calendar[-1]


def _safe_rank_bins(y: pd.Series, bins: int = 5) -> pd.Series:
    """单日横截面收益分桶，转成整数标签"""
    y = y.fillna(0)
    if len(y.unique()) < bins:
        return pd.Series([bins // 2] * len(y), index=y.index, dtype="int32")
    ranked = y.rank(method="first")
    labels = pd.qcut(ranked, bins, labels=False, duplicates="drop")
    return labels.fillna(bins // 2).astype("int32")


def prepare_data(dataset: DatasetH, segment: Tuple[str, str], bins: int = 5):
    """像 Qlib 内部一样，按日期 prepare，避免一次性展开"""
    df = dataset.prepare(segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)

    X_list, y_list, g_list = [], [], []
    for day, df_day in df.groupby(level=0):
        X_day = df_day["feature"].astype("float32").values
        y_raw = df_day["label"]
        if isinstance(y_raw, pd.DataFrame):
            y_raw = y_raw.iloc[:, 0]
        y_day = _safe_rank_bins(y_raw, bins=bins).values.astype("int32")

        if len(y_day) == 0:
            continue
        X_list.append(X_day)
        y_list.append(y_day)
        g_list.append(len(y_day))

    X = np.vstack(X_list).astype("float32")
    y = np.concatenate(y_list).astype("int32")
    groups = np.array(g_list, dtype="int32")
    return X, y, groups


# -------- 训练 / 预测 --------
# def train_model(model_path: str):
#     handler = Alpha158(
#         start_time=start_date,
#         end_time=end_date,
#         instruments=instruments_para,
#         fit_start_time=start_date,
#         fit_end_time=end_date,
#         label=label
#     )
#     handler.fetch()
#
#     dataset = DatasetH(handler=handler, segments={
#         "train": ("2020-01-01", "2024-12-31"),
#         "valid": ("2025-01-01", "2025-06-30"),
#     })
#
#     print("[prepare] 训练集...")
#     X_train, y_train, groups_train = prepare_data(dataset, "train", bins=5)
#     print("[prepare] 验证集...")
#     X_valid, y_valid, groups_valid = prepare_data(dataset, "valid", bins=5)
#
#     print("训练集样本:", X_train.shape, "groups总和:", groups_train.sum())
#     print("验证集样本:", X_valid.shape, "groups总和:", groups_valid.sum())
#
#     model = lgb.LGBMRanker(
#         objective="lambdarank",
#         metric="ndcg",
#         eval_at=[10, 20, 50],
#         max_depth=7,
#         num_leaves=127,
#         n_estimators=2000,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         learning_rate=0.03,
#         reg_alpha=5.0,
#         reg_lambda=10.0,
#         n_jobs=8,
#         device="cpu"  # 先 CPU，跑通后再改 "gpu"
#     )
#
#     model.fit(
#         X_train, y_train,
#         group=groups_train,
#         eval_set=[(X_valid, y_valid)],
#         eval_group=[groups_valid],
#         callbacks=[
#             lgb.early_stopping(50),
#             lgb.log_evaluation(period=100)
#         ]
#     )
#
#     os.makedirs(os.path.dirname(model_path), exist_ok=True)
#     if os.path.exists(model_path):
#         os.remove(model_path)
#     joblib.dump(model, model_path)
#     print(f"✅ 模型已保存至: {model_path}")
#     return model
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

    dataset = DatasetH(handler=handler, segments={
        "train": ("2020-01-01", "2024-12-31"),
        "valid": ("2025-01-01", "2025-06-30"),
    })

    print("[prepare] 训练集...")
    X_train, y_train, groups_train = prepare_data(dataset, "train", bins=5)
    print("[prepare] 验证集...")
    X_valid, y_valid, groups_valid = prepare_data(dataset, "valid", bins=5)

    def objective(trial):
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [10],
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": 2000,
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "n_jobs": 8,
            "device": "cpu",  # 建议先 CPU，稳定后再改 "gpu"
        }

        model = lgb.LGBMRanker(**params)
        model.fit(
            X_train, y_train,
            group=groups_train,
            eval_set=[(X_valid, y_valid)],
            eval_group=[groups_valid],
            callbacks=[
                lgb.early_stopping(50, verbose=False)
            ]
        )
        # Optuna 最大化验证集 NDCG@10
        score = model.best_score_["valid_0"]["ndcg@10"]
        return score

    # Optuna 搜索
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=30)  # 可改为更大，例如 100

    print("✅ 最优参数:", study.best_params)
    print("✅ 最优 NDCG@10:", study.best_value)

    # 用最优参数重新训练完整模型
    best_params = study.best_params
    best_params.update({
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [10, 20, 50],
        "n_estimators": 2000,
        "n_jobs": 8,
        "device": "cpu"
    })

    model = lgb.LGBMRanker(**best_params)
    model.fit(
        X_train, y_train,
        group=groups_train,
        eval_set=[(X_valid, y_valid)],
        eval_group=[groups_valid],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(period=100)
        ]
    )

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if os.path.exists(model_path):
        os.remove(model_path)
    joblib.dump(model, model_path)
    print(f"✅ 模型已保存至: {model_path}")
    return model


def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 模型文件不存在：{model_path}")
    return joblib.load(model_path)


def predict_positions_for_date(predict_date: str, topk: int, model_path: str):
    feature_date = get_prev_trading_date(predict_date)

    handler = Alpha158(
        start_time=feature_date,
        end_time=feature_date,
        instruments=instruments_para,
        fit_start_time=start_date,
        fit_end_time=end_date,
        label=label
    )
    handler.fetch()

    ds = DatasetH(handler=handler, segments={"infer": (feature_date, feature_date)})
    df_infer = ds.prepare("infer", col_set="feature", data_key=DataHandlerLP.DK_I)
    X_infer = df_infer.astype("float32").values

    model = load_model(model_path)
    scores = model.predict(X_infer)

    pred_signal = pd.Series(scores, index=df_infer.index)
    top_stocks = pred_signal.sort_values(ascending=False).head(topk)
    weight = 1.0 / len(top_stocks)
    df_result = pd.DataFrame({
        "stock": [idx[-1] for idx in top_stocks.index],
        "weight": weight,
        "signal": top_stocks.values,
        "buy_date": predict_date
    }).reset_index(drop=True)

    return df_result, pred_signal


# ====== 粘贴到你的脚本中（imports 已有即可） ======

def evaluate_on_valid(dataset: DatasetH, model, topk: int = 20, n_bins: int = 10):
    """
    在 'valid' 分段上评估：RankIC、TopK 命中率、Top-Bottom 价差、均值TopK未来收益
    注意：这里用的是“原始连续标签”（未来5日收益），不是离散分桶后的 y。
    """
    # 取验证集的原始特征与标签
    df_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    X_valid = df_valid["feature"].astype("float32").values

    y_valid = df_valid["label"]
    if isinstance(y_valid, pd.DataFrame):
        y_valid = y_valid.iloc[:, 0]  # 取第一列

    # 预测打分
    pred = model.predict(X_valid)

    # 结果表，MultiIndex: (date, instrument)
    res = pd.DataFrame({"pred": pred, "ret": y_valid.values}, index=df_valid.index)

    # —— 日度 RankIC（Spearman）：corr(rank(pred), rank(ret))，用 pandas 排名 + 皮尔逊
    def _rank_ic_one_day(d: pd.DataFrame) -> float:
        rp = d["pred"].rank(method="first")
        rr = d["ret"].rank(method="first")
        return rp.corr(rr)  # 皮尔逊相关，但输入是秩 → 等价于 Spearman

    ic_daily = res.groupby(level=0).apply(_rank_ic_one_day).dropna()
    rank_ic_mean = float(ic_daily.mean())
    rank_ic_std = float(ic_daily.std(ddof=1)) if len(ic_daily) > 1 else float("nan")
    rank_ic_ir = (rank_ic_mean / rank_ic_std) if pd.notna(rank_ic_std) and rank_ic_std > 0 else float("nan")

    # —— TopK：每天选分数最高的 K 只，统计平均未来收益 & 正收益命中率
    def _topk_stats_one_day(d: pd.DataFrame) -> pd.Series:
        s = d.sort_values("pred", ascending=False).head(topk)["ret"]
        return pd.Series({"avg_ret": s.mean(), "hit_rate": (s > 0).mean()})

    topk_day = res.groupby(level=0).apply(_topk_stats_one_day)
    avg_topk_ret = float(topk_day["avg_ret"].mean())
    avg_topk_hit = float(topk_day["hit_rate"].mean())

    # —— 多空价差：按预测分数分位（n_bins），Top 分位均值 - Bottom 分位均值
    def _decile_spread_one_day(d: pd.DataFrame, q: int = n_bins) -> float:
        # 对同日样本的“预测秩”再做 q 分位
        ranks = d["pred"].rank(method="first")
        try:
            buckets = pd.qcut(ranks, q, labels=False, duplicates="drop")
        except ValueError:
            # 当天样本太少或重复太多，退化为全部放中间，价差记为 0
            return 0.0
        d = d.assign(bucket=buckets)
        top_mean = d.loc[d["bucket"] == d["bucket"].max(), "ret"].mean()
        bot_mean = d.loc[d["bucket"] == d["bucket"].min(), "ret"].mean()
        return float(top_mean - bot_mean)

    spread_daily = res.groupby(level=0).apply(_decile_spread_one_day, q=n_bins)
    long_short_spread = float(spread_daily.mean())

    # 汇总打印
    print("\n==== Validation Metrics ====")
    print(f"RankIC (mean): {rank_ic_mean:.4f} | RankIC (std): {rank_ic_std:.4f} | IR: {rank_ic_ir:.3f}")
    print(f"Top{topk} avg future ret (per-day mean): {avg_topk_ret:.4%}")
    print(f"Top{topk} hit-rate (ret>0): {avg_topk_hit:.2%}")
    print(f"Top-{n_bins} vs Bottom-{n_bins} spread (per-day mean): {long_short_spread:.4%}")
    print("（LightGBM 训练日志里显示的 valid NDCG@K 即为排序质量的另一视角）\n")

    return {
        "rank_ic_mean": rank_ic_mean,
        "rank_ic_std": rank_ic_std,
        "rank_ic_ir": rank_ic_ir,
        "avg_topk_ret": avg_topk_ret,
        "avg_topk_hit": avg_topk_hit,
        "long_short_spread": long_short_spread,
        "ic_daily": ic_daily,
        "topk_daily": topk_day,
        "spread_daily": spread_daily,
    }


def market_hit_rate(dataset: DatasetH, segment: str = "valid") -> float:
    """
    计算验证集期间，市场整体上涨概率（股票未来收益 > 0 的比例）
    """
    df = dataset.prepare(segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)

    y = df["label"]
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]  # 取第一列

    hit_rate = (y > 0).mean()  # 上涨股票占比
    print(f"📊 {segment} 整体市场上涨概率: {hit_rate:.2%}")
    return float(hit_rate)


# -------- 入口 --------
if __name__ == "__main__":
    qlib.init(provider_uri=qlib_url, region=region)
    # ✅ 训练模型一次（如已训练可跳过）加载已训练模型
    # model = load_model(MODEL_PATH)
    # ===== 1. 训练模型 =====
    model = train_model(model_path=MODEL_PATH)

    # ===== 2. 构造和训练时一致的数据集（用来评估）=====
    handler = Alpha158(
        start_time=start_date,
        end_time=end_date,
        instruments=instruments_para,
        fit_start_time=start_date,
        fit_end_time=end_date,
        label=label
    )
    handler.fetch()

    dataset = DatasetH(handler=handler, segments={
        "train": ("2020-01-01", "2025-06-01"),
        "valid": ("2025-01-01", "2025-09-17"),
    })

    # ===== 3. 在验证集上做评估 =====
    metrics = evaluate_on_valid(dataset, model, topk=10, n_bins=10)
    # ===== 4. 对比市场基准 =====
    market_hr = market_hit_rate(dataset, "valid")
    print(f"策略 Top10 命中率: {metrics['avg_topk_hit']:.2%} | 市场基准: {market_hr:.2%}")

    df_pos, pred_signal = predict_positions_for_date(predict_date, topk=10, model_path=MODEL_PATH)
    print(df_pos.head(20))
