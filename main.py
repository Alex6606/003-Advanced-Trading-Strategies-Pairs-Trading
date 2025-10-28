# main.py
from data_loader import download_assets
from pairs_selection import pairs_dict
from visualization import plot_pair_compare_with_signals, plot_pair_compare, plot_alpha_beta, plot_z_with_thresholds, plot_spread_with_signals, \
    show_results
from kalman_filter import kalman_run_W, compute_spread  # 👈 assuming you put that module as kalman_filter.py
from analysis import z_diagnostics, quick_spread_analysis
from runner import prepare_and_run

import pandas as pd

if __name__ == "__main__":
    # 1️⃣ Download all asset classes
    sets = download_assets(years=15)

    # 2️⃣ Run the cointegration analysis
    pairs_all = pairs_dict(sets)
    print("\n===== Top 10 Cointegrated Pairs =====")
    print(pairs_all.head(10))

    # 3️⃣ Focus on a specific asset class
    assets = sets["commodities"]["full"]
    ticker_a = "CL=F"
    ticker_b = "RB=F"

    # 4️⃣ Prepare aligned data for the pair
    df_pair = (
        assets.loc[:, [ticker_a, ticker_b]]
        .asfreq("B")
        .interpolate("time")
        .dropna()
    )

    # 5️⃣ Run the Kalman filter
    W, final_state = kalman_run_W(
        df_pair,
        y_tkr=ticker_a,
        x_tkr=ticker_b,
        window=120,
        q_alpha=1e-8,
        q_beta=1e-6,
        R_user=None,
        inflate=2,
    )

    # 6️⃣ Compute spread
    sp = compute_spread(df_pair, ticker_a, ticker_b, W)

    # 7️⃣ Display diagnostics
    print("\n===== Kalman Filter Output =====")
    print(W.head())
    print("\n===== Computed Spread =====")
    print(sp.head())

    # 8️⃣ Plot diagnostics
    plot_pair_compare(assets, ticker_a, ticker_b, norm="rebased")
    plot_alpha_beta(W)
    plot_z_with_thresholds(W, z_entry=2.0, z_exit=0.5)
    plot_spread_with_signals(W, spread_col="z", pos_col="position")
    plot_pair_compare_with_signals(
    df_prices=assets,
    W=W,
    ticker_a=ticker_a,
    ticker_b=ticker_b,
    window=120,
    norm="rebased",
    show_scatter=True
    )
    
    # 9️⃣ Run diagnostics
    print("\n===== z-Score Diagnostics =====")
    print(z_diagnostics(W))

    print("\n===== Spread Behavior Analysis =====")
    spread_stats = quick_spread_analysis(W)

    #  Run full pipeline (download → Kalman → backtest → summary)
    results = prepare_and_run(
        asset_class="commodities",
        assets=["CL=F", "RB=F"],
        years=15,
        window=120,
        q_alpha=1e-8,
        q_beta=1e-6,
        inflate=4
    )

    show_results(results)



