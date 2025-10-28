# ==========================================
# data_loader.py
# ==========================================
from typing import Dict, List
import yfinance as yf
import pandas as pd


def define_universe() -> Dict[str, List[str]]:
    """
    Defines the asset universe by class.

    Categories:
    -----------
    - Equities: Mega/Large caps diversified by sector and country.
    - ETFs: Broad market, sector, factor, fixed income, and commodity ETFs.
    - Commodities: Metals, energy, agriculture, livestock, and rates.
    - Forex: Mix of major, minor, and exotic pairs (no inverse pairs included).

    Returns
    -------
    dict
        Dictionary with asset classes as keys and lists of tickers as values.
    """
    return {
        "equities": [
            "AAPL","MSFT","AMZN","GOOGL","META","TSLA","NVDA","ORCL","CRM","ADBE",
            "AVGO","AMD","INTC","CSCO","NOW","NFLX","DIS","UBER","ABNB","BKNG",
            "LLY","UNH","JNJ","PFE","ABBV","MRK","NVO",
            "WMT","COST","TGT","NKE","HD","LOW","MCD","KO","PEP",
            "JPM","BAC","GS","MS","V","MA","BRK-B","C",
            "XOM","CVX","SHEL","COP","BP",
            "CAT","DE","BA","LMT","GE","HON","UPS","FDX",
            "TSM","ASML","SAP","TM","SONY","RIO","BHP","NEM","GOLD",
            "HSBC","UBS","BABA","BIDU","PDD"
        ],
        "etfs": [
            "SPY","VTI","QQQ","IWM","MDY",
            "IWF","IWD","MTUM","QUAL","USMV","VYM","SCHD",
            "XLK","XLF","XLE","XLV","XLY","XLP","XLI","XLB","XLU","XLC","XBI",
            "ITB","IYT",
            "EFA","EEM","EWJ","EWG","EWZ","EWW","VT",
            "TLT","IEF","SHY","LQD","HYG","TIP",
            "VNQ","IAU","SLV","DBA","DBC","ICLN","TAN"
        ],
        "commodities": [
            "GC=F","SI=F","PL=F","PA=F","HG=F",
            "CL=F","BZ=F","NG=F","RB=F","HO=F",
            "ZC=F","ZS=F","ZM=F","ZL=F","ZW=F","KE=F","ZO=F","ZR=F",
            "CC=F","KC=F","SB=F","CT=F","OJ=F",
            "LE=F","GF=F","HE=F","LBS=F",
            "ZN=F","ZB=F","ZF=F","ZT=F","DX=F",
            "MGC=F","MCL=F","ALI=F"
        ],
        "forex": [
            "EURUSD=X","GBPUSD=X","AUDUSD=X","NZDUSD=X",
            "JPY=X","USDCHF=X","USDCAD=X","USDMXN=X",
            "USDNOK=X","USDSEK=X","USDZAR=X","USDTRY=X",
            "USDCNH=X","USDINR=X","USDHKD=X","USDKRW=X","USDIDR=X",
            "USDPLN=X","USDCZK=X","USDHUF=X","USDILS=X","USDTHB=X","USDMYR=X",
            "USDCLP=X","USDCOP=X","USDPHP=X","USDTWD=X","USDARS=X","USDNGN=X",
            "EURJPY=X","EURGBP=X","GBPJPY=X","AUDJPY=X",
            "EURCHF=X","EURCAD=X","GBPAUD=X","AUDNZD=X"
        ],
    }

def download_assets(
    years: int = 15,
    drop_thresh: float = 0.05,
    var_thresh: float = 1e-5,
    train_ratio: float = 0.6,
    test_ratio: float = 0.2,
    val_ratio: float = 0.2,
    universe: Dict[str, List[str]] | None = None
) -> dict:
    """
    Downloads and cleans historical price data for multiple assets (non-crypto),
    grouped by class (equities, ETFs, commodities, forex).
    Uses a business-day calendar, interpolates internal gaps, and performs
    temporal train/test/validation splits.

    Parameters
    ----------
    years : int
        Number of years of historical data to fetch.
    drop_thresh : float
        Maximum fraction of missing values allowed per series.
    var_thresh : float
        Minimum variance required to keep a series (filters out flat assets).
    train_ratio, test_ratio, val_ratio : float
        Temporal split ratios. Must sum to 1.0.
    universe : dict or None
        Custom asset universe. If None, uses `define_universe()`.

    Returns
    -------
    dict
        A nested dictionary structured as:
        {class: {"full": df, "train": df_train, "test": df_test, "val": df_val}}
    """

    assert abs(train_ratio + test_ratio + val_ratio - 1.0) < 1e-8, "Ratios must sum to 1.0"

    if universe is None:
        universe = define_universe()

    END = pd.Timestamp.today().normalize()
    START = END - pd.DateOffset(years=years)

    print(f"📡 Downloading data from {START.date()} → {END.date()}")

    # ---------- Helper functions ----------
    def _time_splits(df: pd.DataFrame, tr, te, va):
        n = len(df)
        i1, i2 = int(n * tr), int(n * (tr + te))
        return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()

    def _summary_split(name, df):
        """Prints summary of each temporal split."""
        if df.empty:
            print(f"{name:<6}: (empty)")
        else:
            print(f"{name:<6}: {df.index[0].date()} → {df.index[-1].date()}  n={len(df)}  cols={df.shape[1]}")

    def _download_close(tickers: List[str]) -> pd.DataFrame:
        """Helper: downloads close prices from Yahoo Finance."""
        if not tickers:
            return pd.DataFrame()
        df = yf.download(tickers, start=START, end=END, progress=False, auto_adjust=False)["Close"]
        if isinstance(df, pd.Series):
            df = df.to_frame()
        return df

    # ---------- Download per asset class ----------
    raw = {cls: _download_close(tickers) for cls, tickers in universe.items()}
    out = {}

    for cls, df in raw.items():
        if df.empty:
            print(f"⚠️ {cls}: empty download")
            out[cls] = {"full": df, "train": df, "test": df, "val": df}
            continue

        # 1) Filter out assets with excessive missing data
        frac_nans = df.isna().mean()
        keep_cols = frac_nans[frac_nans < drop_thresh].index
        df = df[keep_cols]

        # 2) Reindex to business days
        df = df.asfreq("B")

        # 3) Interpolate internal gaps, forward-fill edges
        df = df.interpolate(method="time", limit_area="inside").ffill()

        # 4) Remove nearly constant series
        var = df.var()
        df = df[var[var > var_thresh].index]

        # 5) Drop rows with any remaining NaN
        df = df.dropna(axis=0, how="any")

        print(f"\n✅ {cls.upper():<12} clean: {df.shape[1]} assets × {df.shape[0]} rows (B-days)")

        # 6) Temporal split
        df_train, df_test, df_val = _time_splits(df, train_ratio, test_ratio, val_ratio)

        print(f"📅 Time split ({cls}):")
        _summary_split("Train", df_train)
        _summary_split("Test", df_test)
        _summary_split("Val", df_val)

        if not df.empty:
            print("HEAD:"); print(df.head(2))
            print("TAIL:"); print(df.tail(2))

        out[cls] = {"full": df, "train": df_train, "test": df_test, "val": df_val}

    return out
