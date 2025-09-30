import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import kagglehub


def _infer_price_column(df: pd.DataFrame) -> str:
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not cols:
        raise ValueError("No numeric columns found.")
    price_col = cols[0]
    for c in cols:
        s = c.lower()
        if "price" in s or "close" in s:
            price_col = c
            break
    return price_col


def _add_year_column(df: pd.DataFrame) -> bool:
    cands = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    if not cands:
        return False
    try:
        dt = cands[0]
        df[dt] = pd.to_datetime(df[dt], errors="coerce")
        if df[dt].notna().any():
            df["year"] = df[dt].dt.year
            return True
    except Exception:
        pass
    return False


def compute_yearly_stats(df: pd.DataFrame, price_col: str) -> pd.DataFrame | None:
    if "year" not in df.columns:
        return None
    return df.groupby("year")[price_col].agg(["mean", "count"])


def _prepare_features(df: pd.DataFrame, target_col: str):
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in cols if c != target_col]
    if not feats:
        df = df.copy()
        df["time_index"] = range(len(df))
        feats = ["time_index"]
    X = df[feats].fillna(df[feats].mean(numeric_only=True))
    y = df[target_col].fillna(df[target_col].mean())
    return X, y, feats


def fit_and_score_linear(X_train, y_train, X_test, y_test):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    p = lr.predict(X_test)
    return lr, p, r2_score(y_test, p)


def fit_and_score_random_forest(X_train, y_train, X_test, y_test):
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    p = rf.predict(X_test)
    return rf, p, r2_score(y_test, p)


def load_or_download_data() -> pd.DataFrame | None:
    local_file = "gold_data_2015_25.csv"
    if not os.path.exists(local_file):
        print("Downloading dataset from Kaggle...")
        path = kagglehub.dataset_download("mdanwarhossain200110/gold-price-2015-2025")
        print(f"Downloaded to: {path}")
        csvs = [f for f in os.listdir(path) if f.endswith(".csv")]
        if not csvs:
            print("No CSV files found in Kaggle dataset folder")
            return None
        shutil.copy2(os.path.join(path, csvs[0]), local_file)
        print(f"Copied {csvs[0]} to {local_file}")
    else:
        print("Using existing dataset file")
    df = pd.read_csv(local_file)
    print("Data loaded successfully")
    print(df.head())
    return df


def summarize_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\nDataset Info:")
    print(df.info())
    print("\nBasic Stats:")
    print(df.describe())
    print("\nMissing values:", df.isnull().sum().sum())
    print("Duplicates:", df.duplicated().sum())
    return df


def enrich_with_year_and_flags(df: pd.DataFrame) -> pd.DataFrame:
    try:
        price_col = _infer_price_column(df)
    except ValueError as e:
        print(str(e))
        return df
    print(f"\nUsing {price_col} as main metric")
    thr = df[price_col].quantile(0.75)
    high = df[df[price_col] > thr]
    print(f"High value periods (>{thr:.2f}): {len(high)} records")
    if _add_year_column(df):
        stats = compute_yearly_stats(df, price_col)
        if stats is not None:
            print("\nYearly stats:")
            print(stats)
    else:
        print("No usable date/time column for yearly stats")
    return df


def train_and_evaluate_models(df: pd.DataFrame):
    try:
        target_col = _infer_price_column(df)
    except ValueError:
        print("Not enough numeric data for ML")
        return None, None, None, None, None, None
    X, y, _ = _prepare_features(df, target_col)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    lr, lr_pred, lr_r2 = fit_and_score_linear(X_tr, y_tr, X_te, y_te)
    rf, rf_pred, rf_r2 = fit_and_score_random_forest(X_tr, y_tr, X_te, y_te)
    print("\nML Results:")
    print(f"Linear Regression R²: {lr_r2:.4f}")
    print(f"Random Forest R²: {rf_r2:.4f}")
    return lr, rf, X_te, y_te, lr_pred, rf_pred


def plot_basic_charts(df: pd.DataFrame):
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not cols:
        print("No numeric columns for plotting")
        return
    price_col = _infer_price_column(df)
    s = df[price_col]
    n = len(s)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].hist(s, bins=30, alpha=0.7)
    axes[0, 0].set_title(f"{price_col} Distribution")
    axes[0, 0].set_xlabel(price_col)
    axes[0, 0].set_ylabel("Frequency")

    axes[0, 1].boxplot(s)
    axes[0, 1].set_title(f"{price_col} Box Plot")
    axes[0, 1].set_ylabel(price_col)

    if "year" in df.columns:
        yearly = df.groupby("year")[price_col].mean()
        axes[1, 0].plot(yearly.index, yearly.values, marker="o")
        axes[1, 0].set_title(f"Average {price_col} by Year")
        axes[1, 0].set_xlabel("Year")
        axes[1, 0].set_ylabel(f"Average {price_col}")
    else:
        axes[1, 0].plot(s)
        axes[1, 0].set_title(f"{price_col} Over Time")

    axes[1, 1].scatter(range(n), s, alpha=0.6)
    axes[1, 1].set_title(f"{price_col} Scatter")
    axes[1, 1].set_xlabel("Index")
    axes[1, 1].set_ylabel(price_col)

    plt.tight_layout()
    plt.savefig("gold_analysis.png")
    plt.show()


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    d = df.rename(columns={c: c.replace("/", "") for c in df.columns}).copy()
    assets = [c for c in ["SPX", "GLD", "USO", "SLV", "EURUSD"] if c in d.columns]
    for c in assets:
        d[f"{c}_ret"] = d[c].pct_change()
    for c in [f"{a}_ret" for a in assets] + assets:
        if c in d.columns:
            d[f"{c}_lag1"] = d[c].shift(1)
    for c in [f"{a}_ret" for a in assets if f"{a}_ret" in d.columns]:
        d[f"{c}_vol20"] = d[c].rolling(20).std()
    d["y_next"] = d["SPX_ret"].shift(-1)
    d = d.dropna().reset_index(drop=True)
    return d


def _time_series_split(df: pd.DataFrame, test_ratio: float = 0.2):
    n = len(df)
    cut = int(n * (1 - test_ratio))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def plot_nextday_summary(
    data: pd.DataFrame,
    raw_df: pd.DataFrame,
    y_test,
    y_pred,
    rf_model: RandomForestRegressor,
    feature_cols: list[str],
    fname: str = "nextday_summary.png",
):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    num = data.select_dtypes(include=[np.number])
    if not num.empty:
        sns.heatmap(num.corr(numeric_only=True), cmap="viridis", ax=axes[0, 0])
        axes[0, 0].set_title("Correlation")

    if {"SPX", "GLD"}.issubset(raw_df.columns):
        spx_ret = raw_df["SPX"].pct_change()
        gld_ret = raw_df["GLD"].pct_change()
        rc = spx_ret.rolling(120).corr(gld_ret)
        axes[0, 1].plot(rc.index, rc.values)
        axes[0, 1].set_title("GLD vs SPX 120D rolling corr")
        axes[0, 1].set_xlabel("Time")
        axes[0, 1].set_ylabel("Corr")

    if hasattr(rf_model, "feature_importances_"):
        imp = rf_model.feature_importances_
        order = np.argsort(imp)[::-1][:15]
        axes[0, 2].bar(range(len(order)), imp[order])
        axes[0, 2].set_xticks(range(len(order)))
        axes[0, 2].set_xticklabels(
            [feature_cols[i] for i in order], rotation=60, ha="right"
        )
        axes[0, 2].set_title("RF importance (top 15)")

    axes[1, 0].scatter(y_test, y_pred, alpha=0.6)
    axes[1, 0].set_title("y_true vs y_pred")
    axes[1, 0].set_xlabel("True")
    axes[1, 0].set_ylabel("Pred")

    resid = np.array(y_test) - np.array(y_pred)
    axes[1, 1].plot(range(len(resid)), resid, alpha=0.8)
    axes[1, 1].axhline(0, linestyle="--")
    axes[1, 1].set_title("Residuals")
    axes[1, 1].set_xlabel("Index")

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)
    axes[1, 2].axis("off")
    axes[1, 2].text(0, 0.8, "RF Test Metrics", fontsize=12, fontweight="bold")
    axes[1, 2].text(0, 0.6, f"R² = {r2:.4f}")
    axes[1, 2].text(0, 0.45, f"RMSE = {rmse:.6f}")
    axes[1, 2].text(0, 0.25, "Split: 80/20")
    axes[1, 2].text(0, 0.1, "Target: next-day SPX return")

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def run_nextday_model(df: pd.DataFrame):
    data = prepare_data(df)
    feats = [c for c in data.columns if c not in ["y_next", "Date", "year", "SPX_ret"]]
    tr, te = _time_series_split(data, test_ratio=0.2)
    X_tr, y_tr = tr[feats], tr["y_next"]
    X_te, y_te = te[feats], te["y_next"]

    naive_pred = te["SPX_ret"]
    lr = LinearRegression().fit(X_tr, y_tr)
    lr_pred = lr.predict(X_te)
    rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_tr, y_tr)
    rf_pred = rf.predict(X_te)

    def _m(y, yhat):
        rmse = float(np.sqrt(mean_squared_error(y, yhat)))
        return {"r2": r2_score(y, yhat), "rmse": rmse}

    m_naive = _m(y_te, naive_pred)
    m_lr = _m(y_te, lr_pred)
    m_rf = _m(y_te, rf_pred)

    print("\n[Next-Day Prediction] Test metrics")
    print(f"Naive:  R²={m_naive['r2']:.4f}, RMSE={m_naive['rmse']:.6f}")
    print(f"Linear: R²={m_lr['r2']:.4f}, RMSE={m_lr['rmse']:.6f}")
    print(f"RF:     R²={m_rf['r2']:.4f}, RMSE={m_rf['rmse']:.6f}")

    plot_nextday_summary(
        data, df, y_te, rf_pred, rf, feats, fname="nextday_summary.png"
    )

    return {
        "naive": m_naive,
        "lr": m_lr,
        "rf": m_rf,
        "y_test": y_te,
        "rf_pred": rf_pred,
    }


def main():
    print("Starting Gold Price Analysis...")
    df = load_or_download_data()
    if df is None:
        print("Failed to load data")
        return
    df = summarize_data(df)
    df = enrich_with_year_and_flags(df)
    _ = train_and_evaluate_models(df)
    plot_basic_charts(df)
    _ = run_nextday_model(df)
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
