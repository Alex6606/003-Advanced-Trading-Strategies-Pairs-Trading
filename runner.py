from data_loader import download_assets
from kalman_filter import kalman_run_W
from backtest import backtest_kalman_pair, analyze_backtest


# =========================================================
# 1️⃣ Automatically download and prepare data
# =========================================================
def prepare_and_run(
    asset_class="commodities",
    assets=["BZ=F", "CL=F"],
    years=15,
    window=90,
    q_alpha=1e-3,
    q_beta=1e-4,
    R_user=None,
    inflate=2
):
    """
    Automatically download, split, apply Kalman Filter, and run sequential backtests.

    Parameters
    ----------
    asset_class : str
        Asset class to download (e.g., "commodities", "equities", "etfs", "forex").
    assets : list[str]
        Pair of tickers to test.
    years : int
        Lookback period for data download.
    window : int
        Initial window for Kalman filter estimation.
    q_alpha, q_beta : float
        Process noise variances.
    R_user : float | None
        Optional measurement noise override.
    inflate : float
        Inflation factor for initial covariance.

    Returns
    -------
    dict
        Contains all Kalman outputs, backtests, and summary statistics.
    """

    # Download only the selected assets
    custom_universe = {asset_class: assets}
    sets_2 = download_assets(years=years, universe=custom_universe)
    data = sets_2[asset_class]

    # Split into train / test / validation
    df_train = data["train"][assets].dropna()
    df_test  = data["test"][assets].dropna()
    df_val   = data["val"][assets].dropna()

    y_tkr, x_tkr = assets

    # =========================================================
    # 2️⃣ Kalman Filter + Backtest with capital continuity
    # =========================================================

    # Training phase
    W_train, state_train = kalman_run_W(df_train, y_tkr, x_tkr, window, q_alpha, q_beta, R_user, inflate)
    bt_train = backtest_kalman_pair(df_train, W_train, y_tkr, x_tkr)

    # Testing phase (continues from TRAIN)
    W_test, state_test = kalman_run_W(df_test, y_tkr, x_tkr, window, q_alpha, q_beta, R_user, inflate, init_state=state_train)
    bt_test = backtest_kalman_pair(df_test, W_test, y_tkr, x_tkr, initial_cash=bt_train["equity"].iloc[-1])

    # Validation phase (continues from TEST)
    W_val, state_val = kalman_run_W(df_val, y_tkr, x_tkr, window, q_alpha, q_beta, R_user, inflate, init_state=state_test)
    bt_val = backtest_kalman_pair(df_val, W_val, y_tkr, x_tkr, initial_cash=bt_test["equity"].iloc[-1])

    # =========================================================
    # 3️⃣ Comparative summary
    # =========================================================
    print(f"\n===== {asset_class.upper()} | {assets[0]} vs {assets[1]} =====")
    print("\n===== TRAIN =====")
    train_stats = analyze_backtest(bt_train); print(train_stats)

    print("\n===== TEST =====")
    test_stats = analyze_backtest(bt_test); print(test_stats)

    print("\n===== VALIDATION =====")
    val_stats = analyze_backtest(bt_val); print(val_stats)

    return {
        "sets": data,
        "W": {"train": W_train, "test": W_test, "val": W_val},
        "BT": {"train": bt_train, "test": bt_test, "val": bt_val},
        "stats": {"train": train_stats, "test": test_stats, "val": val_stats}
    }
