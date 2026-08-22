"""Backtest de la señal y métricas de rendimiento.

El retorno de cada operación ya viene medido desde el precio del ancla, así que
aquí solo aplicamos posición, costes de transacción y, si procede, el ruido del
proxy de ancla (para no reportar un Sharpe que solo existe si puedes ejecutar al
cierre exacto).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .config import Config, TRADING_DAYS_YEAR


@dataclass
class BacktestResult:
    equity: pd.Series
    net_returns: pd.Series
    gross_returns: pd.Series
    position: pd.Series
    metrics: dict
    trades: pd.DataFrame


def position_from_prob(p: pd.Series, cfg: Config,
                       expected_ret: pd.Series | None = None,
                       sigma: pd.Series | None = None) -> pd.Series:
    """Traduce probabilidad calibrada en tamaño de posición.

    'binary'     : ±1 si supera el umbral
    'confidence' : escala lineal con el exceso de probabilidad sobre el umbral
    'kelly'      : Kelly fraccional con la magnitud esperada y la vol ex-ante
    """
    bt = cfg.backtest
    thr = bt.prob_threshold
    long_sig = (p >= thr).astype(float)
    short_sig = (p <= 1 - thr).astype(float) if bt.allow_short else 0.0

    if bt.sizing == "binary":
        pos = long_sig - short_sig
    elif bt.sizing == "confidence":
        scale = max(1e-6, 1.0 - thr)
        pos = (long_sig * (p - thr) - short_sig * ((1 - thr) - p)) / scale
    elif bt.sizing == "kelly":
        if expected_ret is None or sigma is None:
            raise ValueError("sizing='kelly' requiere expected_ret y sigma")
        edge = expected_ret.astype(float)
        var = (sigma.astype(float) ** 2).clip(lower=1e-8)
        k = (edge / var) * bt.kelly_fraction
        gate = ((p >= thr) | (p <= 1 - thr)).astype(float)
        pos = k * gate
    else:
        raise ValueError(f"sizing no soportado: {bt.sizing}")

    return pd.Series(pos, index=p.index).clip(-bt.max_leverage, bt.max_leverage).fillna(0.0)


def run_backtest(position: pd.Series, y_ret: pd.Series, cfg: Config,
                 anchor_noise_std: float = 0.0, seed: int = 0,
                 n_mc: int = 0) -> BacktestResult:
    """Aplica la posición al retorno y descuenta costes.

    anchor_noise_std > 0 activa una simulación Monte Carlo que perturba el
    precio de entrada con el residuo medido entre Close y ancla real. Sirve para
    responder: "¿sobrevive el edge si mi ejecución a las 15:30 se desvía lo que
    históricamente se desvía?"
    """
    idx = position.index.intersection(y_ret.index)
    pos = position.reindex(idx).fillna(0.0)
    r = y_ret.reindex(idx).astype(float)

    gross = pos * r
    turnover = pos.diff().abs().fillna(pos.abs())
    cost_rate = (cfg.backtest.commission_bps + cfg.backtest.spread_bps
                 + cfg.backtest.slippage_bps) / 10_000.0
    costs = turnover * cost_rate
    net = (gross - costs).fillna(0.0)

    equity = (1.0 + net).cumprod()
    m = compute_metrics(net, pos, r, cfg)
    m["turnover_mean"] = float(turnover.mean())
    m["cost_drag_annual"] = float(costs.mean() * TRADING_DAYS_YEAR)

    if anchor_noise_std > 1e-9 and n_mc > 0:
        m.update(_anchor_stress(pos, r, cost_rate, anchor_noise_std, n_mc, seed, cfg))

    trades = pd.DataFrame({
        "position": pos, "gross_ret": gross, "cost": costs, "net_ret": net,
        "target_ret": r,
    })
    return BacktestResult(equity=equity, net_returns=net, gross_returns=gross,
                          position=pos, metrics=m, trades=trades)


def _anchor_stress(pos: pd.Series, r: pd.Series, cost_rate: float,
                   noise_std: float, n_mc: int, seed: int, cfg: Config) -> dict:
    rng = np.random.default_rng(seed)
    base_r = r.to_numpy()
    p = pos.to_numpy()
    turn = np.abs(np.diff(p, prepend=0.0))
    sharpes = np.empty(n_mc)
    for i in range(n_mc):
        eps = rng.normal(0.0, noise_std, size=len(base_r))
        # entrar a un precio peor por eps reduce el retorno en ~eps
        rr = (1 + base_r) / (1 + eps) - 1
        nn = p * rr - turn * cost_rate
        sd = nn.std(ddof=1)
        sharpes[i] = (nn.mean() / sd * np.sqrt(TRADING_DAYS_YEAR)) if sd > 1e-12 else 0.0
    return {
        "sharpe_anchor_stress_mean": float(np.mean(sharpes)),
        "sharpe_anchor_stress_p05": float(np.percentile(sharpes, 5)),
        "sharpe_anchor_stress_p95": float(np.percentile(sharpes, 95)),
        "anchor_noise_std_bps": float(1e4 * noise_std),
    }


def continuous_pnl(position: pd.Series, anchor: pd.Series, cfg: Config) -> dict:
    """PnL real de una posición que se MANTIENE entre anclas consecutivas.

    El backtest principal suma pos(t)·y(t), donde y(t) va del ancla de t al
    cierre de t+1. Esos tramos se pisan: si mantienes la posición varios días,
    la suma cuenta dos veces la última hora de cada sesión y exagera el
    resultado (medido: ~6% de más en 3 años sobre una posición larga fija).

    Aquí se calcula lo que de verdad ganarías: retornos ancla→ancla, que no se
    solapan y encadenan sin huecos. Si ambos números divergen mucho, la
    estrategia depende de un solapamiento que no puedes ejecutar.
    """
    idx = position.index.intersection(anchor.index)
    pos = position.reindex(idx).fillna(0.0)
    a = anchor.reindex(idx).astype(float)
    r = (a.shift(-1) / a.where(a.abs() > 1e-12) - 1.0)

    ok = r.notna()
    pos, r = pos[ok], r[ok]
    if len(r) < 20:
        return {"n": int(len(r))}

    turnover = pos.diff().abs().fillna(pos.abs())
    cost = (cfg.backtest.commission_bps + cfg.backtest.spread_bps
            + cfg.backtest.slippage_bps) / 10_000.0
    net = (pos * r - turnover * cost).fillna(0.0)

    x = net.to_numpy()
    sd = float(x.std(ddof=1))
    eq = np.cumprod(1 + x)
    peak = np.maximum.accumulate(eq)
    act = pos.abs() > 1e-9
    return {
        "n": int(len(x)),
        "sharpe": float(x.mean() / sd * np.sqrt(TRADING_DAYS_YEAR)) if sd > 1e-12 else np.nan,
        "total_return": float(eq[-1] - 1),
        "max_dd": float((eq / peak - 1).min()),
        "hit_rate": float((net[act] > 0).mean()) if act.sum() else np.nan,
        "autocorr_1": float(pd.Series(x).autocorr(1)) if len(x) > 10 else np.nan,
        "note": "Retornos ancla->ancla: no se solapan, es el PnL ejecutable.",
    }


def compute_metrics(net: pd.Series, pos: pd.Series, r: pd.Series,
                    cfg: Config) -> dict:
    x = net.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 5:
        return {"n": n, "sharpe": np.nan, "cagr": np.nan, "max_dd": np.nan}

    ann = TRADING_DAYS_YEAR
    mu, sd = float(x.mean()), float(x.std(ddof=1))
    sharpe = mu / sd * np.sqrt(ann) if sd > 1e-12 else np.nan

    downside = x[x < 0]
    dsd = float(downside.std(ddof=1)) if len(downside) > 2 else np.nan
    sortino = mu / dsd * np.sqrt(ann) if dsd and dsd > 1e-12 else np.nan

    eq = np.cumprod(1 + x)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    max_dd = float(dd.min())
    years = n / ann
    cagr = float(eq[-1] ** (1 / years) - 1) if years > 0 and eq[-1] > 0 else np.nan
    calmar = cagr / abs(max_dd) if max_dd < -1e-9 and np.isfinite(cagr) else np.nan

    active = pos.abs() > 1e-9
    n_active = int(active.sum())
    wins = x[x > 0]
    losses = x[x < 0]
    hit = float((net[active] > 0).mean()) if n_active else np.nan
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() < 0 else np.nan

    # exactitud direccional pura (sin costes ni sizing)
    da = np.nan
    if n_active:
        sgn = np.sign(pos[active].to_numpy()) * np.sign(r[active].to_numpy())
        sgn = sgn[np.isfinite(sgn)]
        da = float((sgn > 0).mean()) if len(sgn) else np.nan

    # --- solapamiento de operaciones ----------------------------------------
    # y(t) va del ancla del día t al cierre de t+1; y(t+1) arranca en el ancla
    # de t+1. Se solapan durante el tramo final de t+1, así que mantener
    # posición dos días seguidos implica exposición doble en esa franja y deja
    # los retornos autocorrelacionados. El Sharpe ingenuo queda algo inflado:
    # aquí se corrige con Newey-West y se reporta el factor aplicado.
    ac1 = float(pd.Series(x).autocorr(1)) if n > 10 else np.nan
    sharpe_adj = sharpe
    nw_factor = 1.0
    if np.isfinite(ac1) and abs(ac1) > 1e-6:
        nw_factor = float(np.sqrt(max(1e-6, 1 + 2 * ac1 * (1 - 1 / max(n, 2)))))
        sharpe_adj = sharpe / nw_factor if np.isfinite(sharpe) else np.nan
    overlap = float(((pos.shift(1).abs() > 1e-9) & (pos.abs() > 1e-9)).mean())

    return {
        "n": n, "n_active": n_active,
        "autocorr_1": ac1,
        "sharpe_overlap_adj": float(sharpe_adj) if np.isfinite(sharpe_adj) else np.nan,
        "newey_west_factor": nw_factor,
        "overlap_frac": overlap,
        "exposure": float(active.mean()),
        "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "sortino": float(sortino) if np.isfinite(sortino) else np.nan,
        "cagr": cagr, "max_dd": max_dd, "calmar": calmar,
        "vol_annual": float(sd * np.sqrt(ann)),
        "hit_rate": hit, "directional_accuracy": da, "profit_factor": pf,
        "avg_win": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss": float(losses.mean()) if len(losses) else np.nan,
        "skew": float(stats.skew(x)) if n > 8 else np.nan,
        "kurtosis": float(stats.kurtosis(x, fisher=False)) if n > 8 else np.nan,
        "total_return": float(eq[-1] - 1),
        "worst_day": float(x.min()), "best_day": float(x.max()),
        "t_stat": float(mu / (sd / np.sqrt(n))) if sd > 1e-12 else np.nan,
    }


def buy_and_hold(daily_close: pd.Series, index: pd.Index, cfg: Config) -> dict:
    r = daily_close.pct_change().reindex(index).fillna(0.0)
    pos = pd.Series(1.0, index=index)
    return compute_metrics(r, pos, r, cfg)
