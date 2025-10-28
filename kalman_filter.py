# =========================================================
# 0) Model:
#    - State:       w_t = [alpha_t, beta_t]^T
#    - Transition:  w_t = F w_{t-1} + ε_t, with F = I (random walk)
#    - Observation: y_t = C_t w_t + ν_t, with C_t = [1, x_t]
#    - Process noise:     ε_t ~ N(0, Q)
#    - Measurement noise: ν_t ~ N(0, R)
# =========================================================
import numpy as np
import pandas as pd


def kalman_init(
    df,
    y_tkr,
    x_tkr,
    window=120,
    q_alpha=1e-8,
    q_beta=1e-6,
    R_user=None,
    inflate=4
):
    """
    Initialize the Kalman filter using OLS over an initial 'window'.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing series with columns y_tkr and x_tkr.
    y_tkr : str
        Column name for y_t (dependent series).
    x_tkr : str
        Column name for x_t (independent series).
    window : int
        Initial OLS window length.
    q_alpha : float
        Process variance for alpha_t (diagonal entry in Q).
    q_beta : float
        Process variance for beta_t (diagonal entry in Q).
    R_user : float or None
        If provided, sets R; if None, R is estimated from the OLS residual variance.
    inflate : float
        Inflation factor for initial state covariance P0 to avoid overconfidence.

    Returns
    -------
    dict with:
        w : np.ndarray  --> initial state [alpha0, beta0] (shape (2,))
        P : np.ndarray  --> initial state covariance (2x2)
        Q : np.ndarray  --> process noise (2x2)
        R : float       --> measurement noise (scalar)
        policy : dict   --> {"Q":Q,"R":R,"z_entry":..., "z_exit":...}
        x, y : np.ndarray
        idx : list(pd.Timestamp)
        t0 : int        --> index of the last point of the OLS window
        n  : int        --> series length
    """
    # 1) Extract numeric series and align by common index
    y = pd.to_numeric(df[y_tkr], errors="coerce").dropna()
    x = pd.to_numeric(df[x_tkr], errors="coerce").dropna()
    y, x = y.align(x, join="inner")
    y, x = y.dropna(), x.dropna()

    # 2) Validate minimum length
    if len(y) < window + 5:
        raise ValueError("Insufficient data for the initialization window.")

    # 3) To numpy for efficiency
    idx = y.index.to_list()
    yv = y.values.astype(float)
    xv = x.values.astype(float)

    # t0 = last index of the initial window
    t0 = window - 1

    # 4) Design matrix and target on the window
    Xw = np.column_stack([np.ones(window), xv[t0 - window + 1 : t0 + 1]])  # (window x 2)
    yw = yv[t0 - window + 1 : t0 + 1]                                      # (window,)

    # 5) OLS: w0 = [alpha0, beta0]
    w0 = np.linalg.lstsq(Xw, yw, rcond=None)[0]  # shape (2,)

    # 6) Residual variance to estimate R if not provided
    resid = yw - Xw @ w0
    sigma2 = float(resid.var(ddof=2))  # ddof=2 for 2 parameters

    # 7) Measurement noise R
    R = float(sigma2) if (R_user is None) else float(R_user)

    # 8) Initial state covariance P0 ≈ sigma^2 * (X'X)^{-1}, inflated
    XtX_inv = np.linalg.inv(Xw.T @ Xw)
    P0 = sigma2 * XtX_inv * float(inflate)  # (2x2)

    # 9) Process noise Q (controls alpha/beta mobility)
    Q = np.diag([float(q_alpha), float(q_beta)])  # (2x2)

    # 10) Explicit policy (Q, R and z-score thresholds)
    policy = {"Q": Q, "R": R, "z_entry": 2.0, "z_exit": 0.5}

    return {
        "w": w0,
        "P": P0,
        "Q": Q,
        "R": R,
        "policy": policy,
        "x": xv, "y": yv, "idx": idx,
        "t0": t0, "n": len(yv)
    }


def kalman_predict(w, P, Q):
    """
    Time update (prediction step):
    w_pred = w            (F = I)
    P_pred = P + Q
    """
    w_pred = w
    P_pred = P + Q
    return w_pred, P_pred


def kalman_update(w_pred, P_pred, y_t, x_t, R, eps=1e-12):
    """
    Measurement update:

    c_t = [1, x_t] (as 1-D vector, shape (2,))
    S_t = c P_pred c^T + R             (scalar, innovation variance)
    k_t = (P_pred c^T) / S_t           (shape (2,))
    innov = y_t - c w_pred
    w_new = w_pred + k_t * innov
    P_new = (I - k_t c^T) P_pred
    z     = innov / sqrt(S_t)          (normalized innovation)
    yhat_pred = c w_pred               (prior prediction)
    """
    # Observation vector as 1-D to ensure scalar outputs without deprecated casts
    c = np.array([1.0, float(x_t)], dtype=float)  # shape (2,)

    # Innovation variance (scalar)
    S = (c @ P_pred @ c) + float(R)
    if S < eps:
        S = eps  # numerical safety

    # Kalman gain as 1-D vector (shape (2,))
    k = (P_pred @ c) / S

    # Prior prediction and innovation (both scalars)
    yhat_pred = c @ w_pred
    innov = float(y_t) - yhat_pred

    # State and covariance updates
    w_new = w_pred + k * innov
    I = np.eye(P_pred.shape[0])
    P_new = (I - np.outer(k, c)) @ P_pred

    # Normalized innovation (z-score)
    z = innov / np.sqrt(S)

    return w_new, P_new, innov, S, z, yhat_pred


def decide_from_z(z, z_entry=2.0, z_exit=0.5, current_pos=0):
    """
    Position rule:
      - z >=  z_entry  -> open/keep SHORT spread (sell y, buy x)   => -1
      - z <= -z_entry  -> open/keep LONG  spread (buy y, sell x)   => +1
      - |z| <= z_exit  -> close (go flat)                          =>  0
      - otherwise      -> keep current position
    """
    if z >= z_entry:
        return -1
    elif z <= -z_entry:
        return +1
    elif abs(z) <= z_exit:
        return 0
    else:
        return current_pos


def kalman_stream_with_decisions(
    df, y_tkr, x_tkr,
    window=120, q_alpha=1e-7, q_beta=1e-6, R_user=None, inflate=5.0,
    z_entry=2.0, z_exit=0.5
):
    """
    Generator that streams the filter step-by-step and emits decisions.
    Yields tuples:
      (date, alpha, beta, z, innov, S, yhat_pred, position)
    """
    st = kalman_init(df, y_tkr, x_tkr, window, q_alpha, q_beta, R_user, inflate)
    w, P, Q, R = st["w"], st["P"], st["Q"], st["R"]
    x, y, idx, t0, n = st["x"], st["y"], st["idx"], st["t0"], st["n"]

    pos = 0
    for t in range(t0 + 1, n):
        w_pred, P_pred = kalman_predict(w, P, Q)
        w, P, innov, S, z, yhat_pred = kalman_update(w_pred, P_pred, y[t], x[t], R)
        pos = decide_from_z(z, z_entry, z_exit, current_pos=pos)
        yield (
            idx[t],
            float(w[0]), float(w[1]),
            float(z), float(innov), float(S), float(yhat_pred),
            int(pos)
        )


def kalman_run_W(
    df,
    y_tkr,
    x_tkr,
    window=120,
    q_alpha=1e-8,
    q_beta=1e-6,
    R_user=None,
    inflate=4,
    z_entry=2.0,
    z_exit=0.5,
    init_state=None  # keep walk-forward option
):
    """
    Run the Kalman filter (with decisions) over a data block.

    If 'init_state' is passed, continue from that state (strict walk-forward).
    Returns:
      - DataFrame with results (α, β, z, innovation, etc.)
      - Final filter state (to continue on the next block)
    """
    # 1) Continue from previous state if provided
    if init_state is not None:
        w = init_state["w"]
        P = init_state["P"]
        Q = init_state["Q"]
        R = init_state["R"]

        y = pd.to_numeric(df[y_tkr], errors="coerce").dropna().values.astype(float)
        x = pd.to_numeric(df[x_tkr], errors="coerce").dropna().values.astype(float)
        idx = df.index.to_list()
        t0 = 0
        n = len(y)
    else:
        # Fresh initialization (first block)
        st = kalman_init(df, y_tkr, x_tkr, window, q_alpha, q_beta, R_user, inflate)
        w, P, Q, R = st["w"], st["P"], st["Q"], st["R"]
        y, x, idx, t0, n = st["y"], st["x"], st["idx"], st["t0"], st["n"]

    pos = 0
    rows = []

    # 2) Sequential loop
    for t in range(t0 + 1, n):
        w_pred, P_pred = kalman_predict(w, P, Q)
        w, P, innov, S, z, yhat_pred = kalman_update(w_pred, P_pred, y[t], x[t], R)
        pos = decide_from_z(z, z_entry, z_exit, current_pos=pos)

        rows.append({
            "date": idx[t],
            "alpha": float(w[0]),
            "beta": float(w[1]),
            "z": float(z),
            "innov": float(innov),
            "S": float(S),
            "yhat_pred": float(yhat_pred),
            "position": int(pos)
        })

    # 3) Result
    W_df = pd.DataFrame(rows).set_index("date")
    final_state = {"w": w, "P": P, "Q": Q, "R": R}
    return W_df, final_state


def compute_spread(df, y_tkr, x_tkr, W_df):
    """
    Build y, x, yhat and spread (y - yhat) aligned to W_df index.
    """
    # Reindex y and x to W_df dates
    y = pd.to_numeric(df[y_tkr], errors="coerce").reindex(W_df.index)
    x = pd.to_numeric(df[x_tkr], errors="coerce").reindex(W_df.index)

    # yhat_t = alpha_t + beta_t * x_t
    yhat = W_df["alpha"] + W_df["beta"] * x

    # Observed spread
    spread = y - yhat

    out = pd.DataFrame({"y": y, "x": x, "yhat": yhat, "spread": spread}, index=W_df.index)
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out