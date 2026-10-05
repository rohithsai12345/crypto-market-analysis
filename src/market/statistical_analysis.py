import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/processed/market_sentiment_analysis.csv"
)

OUTPUT_FILE = Path(
    "data/processed/statistical_correlation_results.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("STATISTICAL SENTIMENT–MARKET ANALYSIS")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

print("\nRecords:", len(df))


# ============================================================
# CORRELATION FUNCTION
# ============================================================

def calculate_correlation(x_column, y_column, name):

    temp = df[
        [x_column, y_column]
    ].dropna()

    x = temp[x_column]
    y = temp[y_column]

    r, p = pearsonr(x, y)

    n = len(temp)

    # Fisher transformation for approximate 95% CI
    if abs(r) < 1:

        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)

        z_low = z - 1.96 * se
        z_high = z + 1.96 * se

        ci_low = np.tanh(z_low)
        ci_high = np.tanh(z_high)

    else:

        ci_low = r
        ci_high = r

    return {
        "relationship": name,
        "x_variable": x_column,
        "y_variable": y_column,
        "n": n,
        "pearson_r": r,
        "p_value": p,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high
    }


# ============================================================
# RELATIONSHIPS
# ============================================================

results = []


# Same-day relationships

results.append(
    calculate_correlation(
        "avg_sentiment",
        "btc_return",
        "Daily sentiment vs BTC return"
    )
)

results.append(
    calculate_correlation(
        "avg_sentiment",
        "eth_return",
        "Daily sentiment vs ETH return"
    )
)

results.append(
    calculate_correlation(
        "avg_sentiment",
        "btc_volume",
        "Daily sentiment vs BTC volume"
    )
)

results.append(
    calculate_correlation(
        "avg_sentiment",
        "eth_volume",
        "Daily sentiment vs ETH volume"
    )
)


# ============================================================
# LAGGED RELATIONSHIPS
# ============================================================

results.append(
    calculate_correlation(
        "sentiment_lag_1",
        "btc_return_next_day",
        "Previous-day sentiment vs next-day BTC return"
    )
)

results.append(
    calculate_correlation(
        "sentiment_lag_1",
        "eth_return_next_day",
        "Previous-day sentiment vs next-day ETH return"
    )
)

results.append(
    calculate_correlation(
        "sentiment_lag_2",
        "btc_return_next_day",
        "2-day lag sentiment vs next-day BTC return"
    )
)

results.append(
    calculate_correlation(
        "sentiment_lag_3",
        "btc_return_next_day",
        "3-day lag sentiment vs next-day BTC return"
    )
)


# ============================================================
# NEWS ACTIVITY
# ============================================================

results.append(
    calculate_correlation(
        "news_count",
        "btc_abs_return",
        "News volume vs BTC absolute return"
    )
)

results.append(
    calculate_correlation(
        "news_count",
        "eth_abs_return",
        "News volume vs ETH absolute return"
    )
)


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# SIGNIFICANCE LABEL
# ============================================================

results_df["significance_0_05"] = (
    results_df["p_value"] < 0.05
)

results_df["significance_0_01"] = (
    results_df["p_value"] < 0.01
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("CORRELATION RESULTS")
print("=" * 60)

display_columns = [
    "relationship",
    "n",
    "pearson_r",
    "p_value",
    "ci_95_low",
    "ci_95_high",
    "significance_0_05"
]

print(
    results_df[display_columns]
    .round(6)
    .to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)

print("\n" + "=" * 60)
print("STATISTICAL ANALYSIS COMPLETE")
print("=" * 60)
