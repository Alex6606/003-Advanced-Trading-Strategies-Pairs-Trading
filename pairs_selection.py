import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.stattools as tsa
from typing import Dict

# =========================================================
# Rolling correlation and cointegration testing
# =========================================================

def _rolling_correlation(df: pd.DataFrame, window: int) -> dict:
    """
    Compute the average rolling correlation between all pairs of columns.
    """
    df = df.sort_index().astype(float)
    cols = list(df.columns)
    out = {}
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            rc = df[c1].rolling(window=window).corr(df[c2]).dropna()
            out[(c1, c2)] = float(rc.mean()) if len(rc) else np.nan
    return out


def _test_cointegration(s1: pd.Series, s2: pd.Series) -> dict:
    """
    Run OLS regression and test for cointegration using ADF on residuals.
    Model: S1 = α + β·S2 + e; ADF test on residuals.
    """
    s1, s2 = s1.astype(float), s2.astype(float)
    s1, s2 = s1.align(s2, join="inner")
    s1, s2 = s1.dropna(), s2.dropna()
    if s1.empty or s2.empty:
        return {"alpha": np.nan, "beta": np.nan, "adf_resid_p": np.nan, "r2": np.nan, "n": 0}

    X = sm.add_constant(s2.values, has_constant="add")
    model = sm.OLS(s1.values, X, missing="drop").fit()
    alpha, beta = model.params
    resid = s1.values - (alpha + beta * s2.values)

    def _adf_p(x: np.ndarray) -> float:
        x = pd.Series(x).dropna()
        return np.nan if len(x) < 20 else tsa.adfuller(x, autolag="AIC")[1]

    adf_resid_p = _adf_p(resid)

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "adf_resid_p": float(adf_resid_p) if not np.isnan(adf_resid_p) else np.nan,
        "r2": float(model.rsquared),
        "n": int(len(s1))
    }


def pairs_dict(
    sets: Dict[str, object],
    corr_threshold: float = 0.7,
    adf_threshold: float = 0.05,
    window: int = 90
) -> pd.DataFrame:
    """
    Identify cointegrated pairs across multiple asset classes.
    Accepts either:
      - Nested dict: {'commodities': {'full': df, 'train': ...}}
      - Flat dict:   {'commodities': df, 'equities': df, ...}
    """
    all_rows = []

    for asset_class, data in sets.items():
        # Handle nested structure automatically
        df = data["full"] if isinstance(data, dict) and "full" in data else data

        if not isinstance(df, pd.DataFrame) or df.empty or df.shape[1] < 2:
            continue

        corr_map = _rolling_correlation(df, window)
        for (a, b), corr_mean in corr_map.items():
            if np.isnan(corr_mean) or abs(corr_mean) < corr_threshold:
                continue

            stats = _test_cointegration(df[a], df[b])
            is_coint = (
                not np.isnan(stats["adf_resid_p"]) and stats["adf_resid_p"] < adf_threshold
            )

            all_rows.append({
                "Class": asset_class,
                "Asset_1": a,
                "Asset_2": b,
                "Rolling_Corr": corr_mean,
                "ADF_resid_p": stats["adf_resid_p"],
                "Alpha": stats["alpha"],
                "Beta": stats["beta"],
                "R2_OLS": stats["r2"],
                "N_obs": stats["n"],
                "Cointegrated": is_coint
            })

    res = pd.DataFrame(all_rows)
    if res.empty:
        return res

    return res.sort_values(
        by=["Cointegrated", "ADF_resid_p", "Rolling_Corr"],
        ascending=[False, True, False]
    ).reset_index(drop=True)
