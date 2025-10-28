import numpy as np
import pandas as pd

# =========================================================
# z-Score diagnostics and quick spread analysis
# =========================================================

def z_diagnostics(W: pd.DataFrame) -> dict:
    """
    Compute summary statistics for the z-score series.

    Parameters
    ----------
    W : pd.DataFrame
        Must contain a column 'z'.

    Returns
    -------
    dict
        Mean, standard deviation, tail probabilities, and count.
    """
    z = W["z"].dropna()
    return {
        "mean": float(z.mean()),
        "std": float(z.std(ddof=1)),
        "p(|z|>2)": float((z.abs() > 2).mean()),
        "p(|z|>3)": float((z.abs() > 3).mean()),
        "n": int(z.size),
    }


def quick_spread_analysis(W: pd.DataFrame) -> dict:
    """
    Quick analysis of spread behavior based on z-score dynamics.

    Parameters
    ----------
    W : pd.DataFrame
        Must contain a column 'z'.

    Returns
    -------
    dict
        Dictionary summarizing key metrics: mean, std, outlier frequency, and lag-1 autocorrelation.
    """
    z = W["z"].dropna()

    print("=== QUICK SPREAD ANALYSIS ===")
    print(f"Period: {z.index[0].date()} → {z.index[-1].date()}")
    print(f"Mean: {z.mean():.4f}")
    print(f"Standard deviation: {z.std():.4f}")
    print(f"Maximum: {z.max():.4f}")
    print(f"Minimum: {z.min():.4f}")

    # Percent of time outside thresholds
    outside_2std = (abs(z) > 2).mean() * 100
    outside_1std = (abs(z) > 1).mean() * 100

    print(f"\nTime outside ±2σ: {outside_2std:.2f}%")
    print(f"Time outside ±1σ: {outside_1std:.2f}%")

    # Autocorrelation (lag 1)
    autocorr_lag1 = z.autocorr(lag=1)
    print(f"Lag-1 autocorrelation: {autocorr_lag1:.4f}")

    if autocorr_lag1 > 0.5:
        print("⚠️  WARNING: High autocorrelation — spread may not be mean-reverting.")
    elif autocorr_lag1 < 0.2:
        print("✅ Good autocorrelation — spread appears mean-reverting.")

    return {
        "mean": z.mean(),
        "std": z.std(),
        "outside_2std_pct": outside_2std,
        "autocorr_lag1": autocorr_lag1,
    }
