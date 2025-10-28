# visualization.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pairs_selection import _test_cointegration
import seaborn as sns


# =========================================================
# Normalization utilities
# =========================================================
def _normalize_series(s: pd.Series, how: str = "rebased"):
    """
    Normalize a time series in different ways.

    Parameters
    ----------
    s : pd.Series
        Input series to normalize.
    how : str
        Method of normalization:
        - 'rebased': rebase to first value (100)
        - 'zscore' : (x - μ) / σ
        - 'minmax' : (x - min) / (max - min)
    """
    s = s.dropna().astype(float)
    if s.empty:
        return s

    if how == "rebased":
        return 100 * s / s.iloc[0]
    elif how == "zscore":
        mu, sd = s.mean(), s.std(ddof=0)
        return (s - mu) / (sd if sd != 0 else 1.0)
    elif how == "minmax":
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn) if mx != mn else s * 0.0
    else:
        raise ValueError("Parameter 'how' must be 'rebased', 'zscore', or 'minmax'.")


# =========================================================
# Main pair visualization function
# =========================================================
def plot_pair_compare(
    df_prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    window: int = 120,
    norm: str = "rebased",
    show_scatter: bool = True,
):
    """
    Plot normalized price comparison and rolling correlation for a pair of assets.
    """
    # 1️⃣ Align and clean data
    s1, s2 = df_prices[ticker_a].astype(float), df_prices[ticker_b].astype(float)
    s1, s2 = s1.align(s2, join="inner")
    s1, s2 = s1.dropna(), s2.dropna()
    if len(s1) < max(20, window):
        raise ValueError("Insufficient observations after alignment.")

    # 2️⃣ Normalize
    n1 = _normalize_series(s1, norm)
    n2 = _normalize_series(s2, norm)

    # 3️⃣ Compute metrics
    pearson_corr = float(np.corrcoef(s1, s2)[0, 1])
    rolling_corr = s1.rolling(window).corr(s2)
    rolling_mean = float(rolling_corr.dropna().mean()) if rolling_corr.notna().sum() else np.nan
    coint_stats = _test_cointegration(s1, s2)

    # 4️⃣ Plot normalized series + rolling correlation
    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.25)

    ax0 = fig.add_subplot(gs[0])
    ax0.plot(n1.index, n1.values, label=f"{ticker_a} ({norm})")
    ax0.plot(n2.index, n2.values, label=f"{ticker_b} ({norm})")
    ax0.set_title(f"Normalized Comparison: {ticker_a} vs {ticker_b}")
    ax0.set_ylabel("Normalized Level")
    ax0.legend(loc="best")
    ax0.grid(True, alpha=0.25)

    ax1 = fig.add_subplot(gs[1])
    ax1.plot(rolling_corr.index, rolling_corr.values, label=f"Rolling Corr ({window}d)")
    ax1.axhline(0.0, linestyle="--", linewidth=1)
    ax1.set_title(f"Rolling Correlation ({window}-day window)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Correlation")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.25)

    plt.show()

    # 5️⃣ Optional scatter plot
    if show_scatter:
        x, y = s2.values, s1.values
        beta, alpha = np.polyfit(x, y, deg=1)
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = alpha + beta * x_line

        plt.figure(figsize=(6.5, 5))
        plt.scatter(x, y, s=12, alpha=0.6, label="Observations")
        plt.plot(x_line, y_line, linewidth=2, label=f"OLS: y = {alpha:.4f} + {beta:.4f}·x")
        plt.title(f"Scatter Plot: {ticker_a} vs {ticker_b}")
        plt.xlabel(ticker_b)
        plt.ylabel(ticker_a)
        plt.legend()
        plt.grid(True, alpha=0.25)
        plt.show()

    # 6️⃣ Console summary
    print("==== Relationship Summary ====")
    print(f"Common Period: {s1.index.min().date()} → {s1.index.max().date()}  (n={len(s1)})")
    print(f"Pearson Correlation:          {pearson_corr:.4f}")
    print(f"Mean Rolling Correlation:     {rolling_mean:.4f} (window={window})")
    print(f"OLS alpha: {coint_stats['alpha']:.6f} | beta: {coint_stats['beta']:.6f} | R²: {coint_stats['r2']:.4f}")
    print(f"ADF Residuals p-value:        {coint_stats['adf_resid_p']:.4g}  "
          f"{'(cointegrated)' if (not np.isnan(coint_stats['adf_resid_p']) and coint_stats['adf_resid_p'] < 0.05) else ''}")


# =========================================================
# Plot Kalman Filter diagnostics and trading signals
# =========================================================

def plot_alpha_beta(W: pd.DataFrame, roll: int | None = None, title: str | None = None):
    """
    Plot alpha and beta dynamics estimated by the Kalman filter.

    Parameters
    ----------
    W : pd.DataFrame
        Must contain ['alpha', 'beta'] and have a DatetimeIndex.
    roll : int | None
        Optional rolling window for smoothing.
    title : str | None
        Custom title for the plot.
    """
    if W is None or W.empty:
        raise ValueError("Input DataFrame 'W' is empty. Run kalman_run_W first.")

    data = W[['alpha', 'beta']].copy()
    if roll and roll > 1:
        data = data.rolling(roll, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    data.plot(ax=ax)
    ax.set_title(title or "Evolution of α and β (Kalman Filter)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


def plot_z_with_thresholds(
    W: pd.DataFrame,
    z_entry: float = 2.0,
    z_exit: float | None = None,
    roll: int | None = None,
    title: str = "Normalized Spread (z-score)"
):
    """
    Plot the z-score evolution with entry/exit thresholds.

    Parameters
    ----------
    W : pd.DataFrame
        Must contain column 'z'.
    z_entry : float
        Entry threshold (draws ±z_entry lines).
    z_exit : float | None
        Optional exit threshold (draws ±z_exit lines if provided).
    roll : int | None
        Rolling window for smoothing.
    title : str
        Plot title.
    """
    if W is None or W.empty or "z" not in W.columns:
        raise ValueError("DataFrame 'W' is empty or missing column 'z'.")

    z = W["z"].copy()
    if roll and roll > 1:
        z = z.rolling(roll, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    z.plot(ax=ax, label="z", linewidth=1.6)
    ax.axhline(z_entry, linestyle="--", linewidth=2, color="orange", label="Entry band (±)")
    ax.axhline(-z_entry, linestyle="--", linewidth=2, color="orange")

    if z_exit is not None:
        ax.axhline(z_exit, linestyle="--", linewidth=1, color="grey", alpha=0.8, label="Exit band (±)")
        ax.axhline(-z_exit, linestyle="--", linewidth=1, color="grey", alpha=0.8)

    # Shade regions where |z| ≥ z_entry
    ax.fill_between(z.index, z, z_entry, where=(z >= z_entry), alpha=0.12)
    ax.fill_between(z.index, z, -z_entry, where=(z <= -z_entry), alpha=0.12)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("z-score")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


def plot_spread_with_signals(
    W: pd.DataFrame,
    spread_col: str = "z",
    pos_col: str = "position",
    entry: float | None = None,
    exit_: float | None = None,
    roll: int | None = None,
    shade_states: bool = True,
    title: str = "Spread & Trading Signals"
):
    """
    Plot the spread (e.g., z-score) and mark trading events.

    Parameters
    ----------
    W : pd.DataFrame
        Must contain ['z', 'position'].
    spread_col : str
        Column name for spread.
    pos_col : str
        Column name for position.
    entry, exit_ : float | None
        Entry/exit thresholds.
    roll : int | None
        Rolling window for smoothing.
    shade_states : bool
        Whether to color background by position regime.
    """
    if W is None or W.empty:
        raise ValueError("W is empty.")
    for c in [spread_col, pos_col]:
        if c not in W.columns:
            raise ValueError(f"Missing column '{c}' in W.")
    if not isinstance(W.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        raise ValueError("W must be indexed by dates.")

    df = W[[spread_col, pos_col]].copy().sort_index()
    s = df[spread_col].astype(float)
    if roll and roll > 1:
        s = s.rolling(roll, min_periods=1).mean()

    state = df[pos_col].astype(int)
    prev = state.shift(1).fillna(0)

    # Identify trading state transitions
    enter_long = (prev == 0) & (state == 1)
    enter_short = (prev == 0) & (state == -1)
    exit_flat = (prev != 0) & (state == 0)
    flip_long = (prev == -1) & (state == 1)
    flip_short = (prev == 1) & (state == -1)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(s.index, s.values, linewidth=1.6, label=spread_col)

    # Threshold bands
    if entry is not None:
        ax.axhline(+entry, linestyle="--", linewidth=1.2, color="orange", label="±entry")
        ax.axhline(-entry, linestyle="--", linewidth=1.2, color="orange")
    if exit_ is not None:
        ax.axhline(+exit_, linestyle="--", linewidth=1.0, color="grey", alpha=0.9, label="±exit")
        ax.axhline(-exit_, linestyle="--", linewidth=1.0, color="grey", alpha=0.9)

    # Optional regime shading
    if shade_states:
        idx = s.index
        st = state.reindex(idx).astype(int).values
        change = np.r_[True, st[1:] != st[:-1], True]
        bounds = np.flatnonzero(change)
        colors = {1: "tab:green", 0: "lightgrey", -1: "tab:red"}
        for i in range(len(bounds) - 1):
            a = idx[bounds[i]]
            b = idx[bounds[i + 1] - 1]
            ax.axvspan(a, b, color=colors.get(st[bounds[i]], "lightgrey"), alpha=0.10, linewidth=0)

    def _scatter(mask, marker, color, label, size=64):
        if mask.any():
            x = s.index[mask]
            y = s.loc[x]
            ax.scatter(x, y, marker=marker, s=size, color=color, label=label, zorder=3)

    _scatter(enter_long, "^", "tab:green", "Enter Long")
    _scatter(enter_short, "v", "tab:red", "Enter Short")
    _scatter(exit_flat, "o", "grey", "Exit to Flat", size=48)
    _scatter(flip_long, "s", "tab:blue", "Flip → Long")
    _scatter(flip_short, "D", "tab:orange", "Flip → Short")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Spread")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncol=2)
    plt.tight_layout()
    plt.show()

def plot_pair_compare_with_signals(
    df_prices: pd.DataFrame,
    W: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    pos_col: str = "position",
    window: int = 120,
    norm: str = "rebased",
    show_scatter: bool = True,
):
    """
    Plot two normalized price series for a pair (ticker_a vs ticker_b) and overlay regime shading
    and event markers based on a discrete position state in W (e.g., {-1, 0, +1}).

    This version draws a single chart (no lower rolling-correlation subplot). It expects the
    position/time state to be provided in the DataFrame `W` under the column given by `pos_col`.
    The function will align df_prices and W by their datetime indices and only keep overlapping dates.

    Parameters
    ----------
    df_prices : pd.DataFrame
        Wide price DataFrame with datetime index and columns including `ticker_a` and `ticker_b`.
        Values must be numeric (float-compatible).
    W : pd.DataFrame
        DataFrame indexed by datetime with at least one column containing the discrete position
        state (e.g., -1, 0, +1). The column name is provided through `pos_col`.
    ticker_a, ticker_b : str
        Column names in `df_prices` for the two assets to compare.
    pos_col : str, default "position"
        Column in `W` storing the regime/position state per timestamp.
    window : int, default 120
        Currently unused here; kept for API compatibility if rolling stats are added elsewhere.
    norm : {"rebased", "zscore", "minmax"}, default "rebased"
        Normalization method passed to `_normalize_series`:
        - "rebased": rebase to 100 at the first in-window value
        - "zscore" : standard score
        - "minmax" : scale to [0, 100]
    show_scatter : bool, default True
        Reserved for compatibility; this version does not draw an additional scatter/OLS panel.

    Raises
    ------
    ValueError
        If tickers are missing in df_prices, `pos_col` is missing in W,
        there are no overlapping dates, or there are too few observations.

    Notes
    -----
    - The function relies on an existing helper `_normalize_series(series, how=norm)`.
    - Regime shading is applied per contiguous block of equal state in `W[pos_col]`.
      Colors: +1 (green), 0 (light grey), -1 (red), with light transparency (alpha=0.10).
    - Event markers highlight transitions: enter long/short, exit to flat, and flips.
    """

    # ---------- Input validation ----------
    if ticker_a not in df_prices.columns or ticker_b not in df_prices.columns:
        raise ValueError("Tickers not found in df_prices.")
    if pos_col not in W.columns:
        raise ValueError(f"Column '{pos_col}' not found in W.")

    # Ensure numeric series; non-numeric data will raise downstream errors during normalization
    s1 = df_prices[ticker_a].astype(float)
    s2 = df_prices[ticker_b].astype(float)

    # ---------- Index alignment ----------
    # Keep only dates common to both price series and the W state DataFrame
    idx = s1.index.intersection(s2.index).intersection(W.index)
    if idx.empty:
        raise ValueError("No overlapping dates between prices and W.")

    s1, s2 = s1.loc[idx], s2.loc[idx]
    state = W.loc[idx, pos_col].astype(int)  # ensure discrete integer states
    if len(s1) < 20:
        raise ValueError("Too few observations after alignment/cleaning (need >= 20).")

    # ---------- Normalization ----------
    # Delegate scaling to your existing helper; falls back to sensible y-axis labels
    n1 = _normalize_series(s1, how=norm)
    n2 = _normalize_series(s2, how=norm)
    y_labels = {"rebased": "Index (base=100)", "zscore": "Z-score", "minmax": "Scale 0–100"}
    ylab = y_labels.get(norm.lower(), "Index (base=100)")

    # ---------- Figure (single axes) ----------
    fig = plt.figure(figsize=(11, 5), constrained_layout=True)
    ax0 = fig.add_subplot(1, 1, 1)

    # Plot normalized series
    ax0.plot(n1.index, n1.values, label=ticker_a, linewidth=1.6)
    ax0.plot(n2.index, n2.values, label=ticker_b, linewidth=1.6, alpha=0.9)
    ax0.set_title(f"Normalized comparison: {ticker_a} vs {ticker_b}")
    ax0.set_xlabel("Date")
    ax0.set_ylabel(ylab)
    ax0.grid(True, alpha=0.25)

    # ---------- Regime shading ----------
    # Identify boundaries where the discrete state changes (including start/end sentinels)
    st = state.values
    change = np.r_[True, st[1:] != st[:-1], True]      # True at state changes and at both ends
    bounds = np.flatnonzero(change)                    # indices of change/start/end
    colors = {1: "tab:green", 0: "lightgrey", -1: "tab:red"}

    # Shade each contiguous block [bounds[i], bounds[i+1]) with the color of its state
    for i in range(len(bounds) - 1):
        block_state = st[bounds[i]]
        a = state.index[bounds[i]]
        b = state.index[bounds[i+1] - 1]              # inclusive end for visual continuity
        ax0.axvspan(a, b, color=colors.get(block_state, "lightgrey"), alpha=0.10, linewidth=0)

    # ---------- Event markers ----------
    # Define transitions relative to the previous state
    prev = state.shift(1).fillna(0)
    enter_long     = (prev == 0)  & (state ==  1)
    enter_short    = (prev == 0)  & (state == -1)
    exit_flat      = (prev != 0)  & (state ==  0)
    flip_to_long   = (prev == -1) & (state ==  1)
    flip_to_short  = (prev ==  1) & (state == -1)

    def _scatter(mask: pd.Series, yseries: pd.Series, marker: str, color: str,
                 label: str | None = None, size: int = 64) -> None:
        """
        Helper to place scatter markers for boolean event masks on the main axis.
        Uses the y-values from `yseries` at the event timestamps for visual anchoring.
        """
        if mask.any():
            x = yseries.index[mask]
            y = yseries.loc[x]
            ax0.scatter(x, y, marker=marker, s=size, color=color, label=label, zorder=3)

    # Use ASCII markers to avoid font issues with special glyphs
    _scatter(enter_long,    n1, "^", "tab:green",  "Enter Long")
    _scatter(enter_short,   n1, "v", "tab:red",    "Enter Short")
    _scatter(exit_flat,     n1, "o", "grey",       "Exit to Flat", size=48)
    _scatter(flip_to_long,  n1, "s", "tab:blue",   "Flip → Long")
    _scatter(flip_to_short, n1, "D", "tab:orange", "Flip → Short")

    # ---------- Legend & render ----------
    ax0.legend(loc="best")
    plt.show()

# =========================================================
# PLOTTING FUNCTIONS
# =========================================================

def plot_equity_curves(bt_train, bt_test, bt_val):
    """Plot the equity curves for Train, Test, and Validation periods."""
    plt.figure(figsize=(10, 5))
    plt.plot(bt_train["equity"], label="Train", lw=2)
    plt.plot(bt_test["equity"], label="Test", lw=2)
    plt.plot(bt_val["equity"], label="Validation", lw=2)
    plt.title("Equity Curves (Train / Test / Validation)")
    plt.xlabel("Date")
    plt.ylabel("Normalized Equity")
    plt.legend()
    plt.show()

def plot_drawdown(bt):
    """Plot drawdown percentage over time."""
    equity = bt["equity"]
    peak = equity.cummax()
    drawdown = (equity / peak - 1) * 100
    plt.figure(figsize=(10, 3))
    plt.fill_between(bt.index, drawdown, 0, color="red", alpha=0.4)
    plt.title("Drawdown (%)")
    plt.ylabel("Drawdown (%)")
    plt.xlabel("Date")
    plt.show()

def plot_return_distribution(bt):
    """Plot histogram of daily returns."""
    daily_ret = bt["equity"].pct_change().dropna()
    plt.figure(figsize=(7, 4))
    sns.histplot(daily_ret, bins=40, kde=True, color="steelblue")
    plt.title("Distribution of Daily Returns")
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.show()

def plot_trades(bt):
    """Visualize active positions and trade points."""
    plt.figure(figsize=(10, 3))
    plt.plot(bt.index, bt["position"], color="black", lw=1.2)
    plt.title("System Positions (1 = Long Spread, -1 = Short Spread, 0 = Flat)")
    plt.xlabel("Date")
    plt.ylabel("Position")
    plt.yticks([-1, 0, 1])
    plt.show()

# =========================================================
# TABLES AND REPORTING
# =========================================================

def display_summary_table(train_stats, test_stats, val_stats):
    """Display a comparative summary table of metrics across periods."""
    df = pd.DataFrame(
        [train_stats, test_stats, val_stats],
        index=["Train", "Test", "Validation"]
    )
    print("\n===== PERFORMANCE SUMMARY =====")
    print(df.round(4))
    return df

def plot_equity_with_regions(bt_train, bt_test, bt_val):
    """Plot all equity curves together with shaded regions per period."""
    plt.figure(figsize=(10, 5))
    plt.plot(bt_train["equity"], color="green", label="Train", lw=2)
    plt.plot(bt_test["equity"], color="orange", label="Test", lw=2)
    plt.plot(bt_val["equity"], color="blue", label="Validation", lw=2)
    plt.axvspan(bt_train.index[0], bt_train.index[-1], color="green", alpha=0.1)
    plt.axvspan(bt_test.index[0], bt_test.index[-1], color="orange", alpha=0.1)
    plt.axvspan(bt_val.index[0], bt_val.index[-1], color="blue", alpha=0.1)
    plt.legend()
    plt.title("Equity Curves with Period Highlights")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.show()

def show_results(resultados):
    """Automatically generate all plots and summary table."""
    bt_train = resultados["BT"]["train"]
    bt_test = resultados["BT"]["test"]
    bt_val = resultados["BT"]["val"]
    stats = resultados["stats"]

    # === Plots ===
    plot_equity_curves(bt_train, bt_test, bt_val)
    plot_drawdown(bt_train)
    plot_return_distribution(bt_train)
    plot_trades(bt_train)
    plot_equity_with_regions(bt_train, bt_test, bt_val)

    # === Summary Table ===
    display_summary_table(stats["train"], stats["test"], stats["val"])
