# =========================================================
# Helpers
# =========================================================
import numpy as np
import pandas as pd


def _target_dollar_holdings(equity, invest_frac, price_y, price_x, beta, position):
    """Compute target dollar weights and notionals for each leg of the pair."""
    if position == 0 or not np.isfinite(beta):
        return 0.0, 0.0, 0.0, 0.0

    gross = invest_frac * equity
    denom = 1.0 + abs(beta)

    if position > 0:  # long spread: +y, -βx
        w_y = +1.0 / denom
        w_x = -beta / denom
    else:  # short spread: -y, +βx
        w_y = -1.0 / denom
        w_x = +beta / denom

    h_y = gross * w_y
    h_x = gross * w_x

    u_y = 0.0 if price_y == 0 or not np.isfinite(price_y) else h_y / price_y
    u_x = 0.0 if price_x == 0 or not np.isfinite(price_x) else h_x / price_x
    return h_y, h_x, u_y, u_x


# =========================================================
# Main backtest
# =========================================================

def backtest_kalman_pair(
    df_pair: pd.DataFrame,
    W_df: pd.DataFrame,
    y_tkr: str,
    x_tkr: str,
    commission_rate: float = 0.00125,    # 0.125%
    borrow_rate_annual: float = 0.0025,  # 0.25% annual
    invest_frac: float = 0.8,
    initial_cash: float = 1.0
) -> pd.DataFrame:
    """Run a daily backtest with dynamic hedge ratio rebalancing."""
    px = df_pair[[y_tkr, x_tkr]].copy()
    px = px.reindex(W_df.index).dropna()
    W = W_df.reindex(px.index).copy()

    ry = px[y_tkr].pct_change().fillna(0.0)
    rx = px[x_tkr].pct_change().fillna(0.0)

    out = []
    equity = initial_cash
    hy_prev = hx_prev = uy_prev = ux_prev = 0.0
    daily_borrow_rate = borrow_rate_annual / 252.0
    dates = px.index.to_list()

    for i in range(1, len(dates)):
        d_prev, d = dates[i - 1], dates[i]
        price_y_prev, price_x_prev = float(px.loc[d_prev, y_tkr]), float(px.loc[d_prev, x_tkr])
        r_y, r_x = float(ry.loc[d]), float(rx.loc[d])

        pnl_gross = hy_prev * r_y + hx_prev * r_x
        r_pair = pnl_gross / (abs(hy_prev) + abs(hx_prev)) if (abs(hy_prev) + abs(hx_prev)) > 0 else 0.0

        short_notional = abs(hy_prev) * (hy_prev < 0) + abs(hx_prev) * (hx_prev < 0)
        borrow_cost = daily_borrow_rate * short_notional

        equity_mid = equity + pnl_gross - borrow_cost

        beta = float(W.loc[d, "beta"])
        position = int(W.loc[d, "position"]) if "position" in W.columns else 0
        price_y, price_x = float(px.loc[d, y_tkr]), float(px.loc[d, x_tkr])
        hy_tgt, hx_tgt, uy_tgt, ux_tgt = _target_dollar_holdings(
            equity_mid, invest_frac, price_y, price_x, beta, position
        )

        turnover = abs(hy_tgt - hy_prev) + abs(hx_tgt - hx_prev)
        commission = commission_rate * turnover

        equity_end = equity_mid - commission

        out.append({
            "date": d,
            "equity": equity_end,
            "pnl_gross": pnl_gross,
            "pnl_net": pnl_gross - commission - borrow_cost,
            "r_pair": r_pair,
            "commission": commission,
            "borrow": borrow_cost,
            "beta": beta,
            "position": position,
            "turnover": turnover,
            "hy": hy_tgt,
            "hx": hx_tgt,
            "uy": uy_tgt,
            "ux": ux_tgt
        })

        hy_prev, hx_prev = hy_tgt, hx_tgt
        uy_prev, ux_prev = uy_tgt, ux_tgt
        equity = equity_end

    return pd.DataFrame(out).set_index("date")


# =========================================================
# Performance & trade statistics
# =========================================================

def performance_metrics(bt: pd.DataFrame) -> dict:
    """Compute global performance and risk metrics."""
    daily_returns = bt["equity"].pct_change().dropna()
    avg = daily_returns.mean()
    std = daily_returns.std()
    downside = daily_returns[daily_returns < 0].std()

    sharpe = np.sqrt(252) * avg / std if std > 0 else np.nan
    sortino = np.sqrt(252) * avg / downside if downside > 0 else np.nan

    equity = bt["equity"]
    peak = equity.cummax()
    dd = (equity / peak - 1)
    mdd = dd.min()
    calmar = (avg * 252) / abs(mdd) if mdd < 0 else np.nan

    return {
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Max_Drawdown": mdd,
        "Total_Return": equity.iloc[-1] / equity.iloc[0] - 1,
        "Annualized_Return": (equity.iloc[-1] / equity.iloc[0]) ** (252 / len(bt)) - 1
    }


def trade_statistics(bt: pd.DataFrame) -> dict:
    """Compute trade count, win rate, and profit factor."""
    positions = bt["position"]
    trades = (positions != positions.shift()).sum()
    if trades == 0:
        return {"Trades": 0, "Win_Rate": np.nan, "Avg_Win": np.nan,
                "Avg_Loss": np.nan, "Profit_Factor": np.nan}

    pnl = bt["pnl_net"].copy()
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    avg_win = wins.mean() if not wins.empty else 0
    avg_loss = losses.mean() if not losses.empty else 0
    win_rate = len(wins) / (len(wins) + len(losses)) if (len(wins) + len(losses)) > 0 else np.nan
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.nan

    return {
        "Trades": int(trades),
        "Win_Rate": win_rate,
        "Avg_Win": avg_win,
        "Avg_Loss": avg_loss,
        "Profit_Factor": profit_factor
    }


def cost_analysis(bt: pd.DataFrame) -> dict:
    """Compute total trading and borrowing costs."""
    return {
        "Total_Commissions": bt["commission"].sum(),
        "Total_Borrow": bt["borrow"].sum(),
        "Total_Costs": bt["commission"].sum() + bt["borrow"].sum()
    }


def analyze_backtest(bt: pd.DataFrame) -> dict:
    """Aggregate performance, trade, and cost metrics."""
    metrics = performance_metrics(bt)
    trades = trade_statistics(bt)
    costs = cost_analysis(bt)
    return {**metrics, **trades, **costs}
