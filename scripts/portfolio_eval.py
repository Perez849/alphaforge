#!/usr/bin/env python3
"""Evalúa la CARTERA combinada, no los valores por separado.

Por qué hace falta: cada modelo, suelto, tiene una exposición baja (medido:
~24% del tiempo) y un Sharpe pequeño. Juzgarlos de uno en uno se pierde dos
cosas que solo existen en el conjunto:

  * Diversificación. Ocho señales con edge pequeño y correlación imperfecta
    dan un Sharpe agregado mayor que la media de los ocho.
  * Uso del capital. Si cada modelo está fuera el 76% del tiempo, la cartera
    combinada sí tiene algo puesto casi todos los días.

Lee los CSV de predicciones fuera de muestra que deja `train_all.py` y calcula
el resultado de combinarlas. También reporta la correlación entre señales, que
es lo que decide si la diversificación es real o una ilusión.

    python scripts/portfolio_eval.py --reports models/reports
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from alphaforge.config import TRADING_DAYS_YEAR, Config
from alphaforge.utils import fmt_num, fmt_pct, get_logger


def load_oos(reports_dir: str) -> dict[str, pd.DataFrame]:
    out = {}
    for path in sorted(glob.glob(os.path.join(reports_dir, "*_oos.csv"))):
        ticker = os.path.basename(path).split("_")[0]
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if {"prob_up", "y_ret"} <= set(df.columns):
                out[ticker] = df
        except Exception as e:                           # noqa: BLE001
            print(f"  {path}: ilegible ({e})", file=sys.stderr)
    return out


def sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return np.nan
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(TRADING_DAYS_YEAR)) if sd > 1e-12 else np.nan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="models/reports")
    ap.add_argument("--out", default=None, help="guardar el resumen en JSON")
    ap.add_argument("--gain", type=float, default=None, help="sobrescribe continuous_gain")
    ap.add_argument("--cap", type=float, default=None, help="sobrescribe max_position")
    args = ap.parse_args()

    log = get_logger(1)
    data = load_oos(args.reports)
    if not data:
        log.error(f"no hay *_oos.csv en {args.reports}")
        return 1

    cfg = Config()
    gain = args.gain if args.gain is not None else cfg.backtest.continuous_gain
    cap = args.cap if args.cap is not None else cfg.backtest.max_position
    cost = (cfg.backtest.commission_bps + cfg.backtest.spread_bps
            + cfg.backtest.slippage_bps) / 10_000.0

    # --- posición continua y PnL por valor, sobre el calendario común -------
    pos, ret = {}, {}
    for t, df in data.items():
        e = (df["prob_up"].astype(float) - 0.5) / 0.5
        p = (e * gain).clip(-cap, cap)
        p = p.where(p.abs() >= cfg.backtest.min_position, 0.0)
        pos[t], ret[t] = p, df["y_ret"].astype(float)

    P = pd.DataFrame(pos).sort_index()
    R = pd.DataFrame(ret).reindex(P.index)
    net = (P * R).sub(P.diff().abs().fillna(P.abs()) * cost)

    print("=" * 74)
    print("CARTERA COMBINADA — señales continuas, sin umbral")
    print("=" * 74)
    print(f"{'valor':<8}{'Sharpe':>9}{'expos.':>9}{'|pos| med':>11}{'rotación':>10}"
          f"{'acierto':>9}")
    print("-" * 74)
    for t in P.columns:
        n = net[t].dropna()
        act = P[t].abs() > 1e-9
        hit = float((np.sign(P[t][act]) == np.sign(R[t][act])).mean()) if act.any() else np.nan
        print(f"{t:<8}{fmt_num(sharpe(n), 2):>9}{fmt_pct(float(act.mean()), 0):>9}"
              f"{fmt_num(float(P[t].abs().mean()), 3):>11}"
              f"{fmt_num(float(P[t].diff().abs().mean()), 3):>10}"
              f"{fmt_pct(hit, 1):>9}")

    # --- ¿es real la diversificación? ---------------------------------------
    common = net.dropna(how="all")
    corr = common.corr()
    off = corr.to_numpy()[np.triu_indices(len(corr), 1)]
    off = off[np.isfinite(off)]
    rho = float(np.mean(off)) if len(off) else np.nan

    port = common.mean(axis=1, skipna=True).dropna()
    ind = np.array([sharpe(net[t].dropna()) for t in P.columns], dtype=float)
    ind = ind[np.isfinite(ind)]
    k = len(ind)
    teo_indep = float(np.mean(ind) * np.sqrt(k)) if k else np.nan
    teo_corr = (float(np.mean(ind) * np.sqrt(k / (1 + (k - 1) * max(rho, 0))))
                if k and np.isfinite(rho) else np.nan)

    print("-" * 74)
    print(f"{'CARTERA':<8}{fmt_num(sharpe(port), 2):>9}"
          f"{fmt_pct(float((P.abs() > 1e-9).any(axis=1).mean()), 0):>9}"
          f"{fmt_num(float(P.abs().sum(axis=1).mean()), 3):>11}")
    print()
    print(f"  Sharpe medio de los valores sueltos : {fmt_num(float(np.mean(ind)), 2)}")
    print(f"  correlación media entre señales     : {fmt_num(rho, 3)}")
    print(f"  Sharpe teórico si fueran indep.     : {fmt_num(teo_indep, 2)}")
    print(f"  Sharpe teórico con esa correlación  : {fmt_num(teo_corr, 2)}")
    print(f"  Sharpe REAL de la cartera           : {fmt_num(sharpe(port), 2)}")
    print()
    if np.isfinite(rho) and rho > 0.35:
        print("  La correlación entre señales es alta: son ocho apuestas parecidas,")
        print("  no ocho apuestas distintas. La diversificación aporta poco.")
    elif np.isfinite(rho):
        print("  Correlación baja: la diversificación es real y es donde está el")
        print("  grueso del valor, no en ningún valor por separado.")

    x = port.to_numpy()
    eq = np.cumprod(1 + x)
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    yrs = len(x) / TRADING_DAYS_YEAR
    print()
    print(f"  periodo   {port.index.min().date()} -> {port.index.max().date()} "
          f"({yrs:.1f} años)")
    print(f"  retorno   {fmt_pct(float(eq[-1] - 1), 1)}   "
          f"CAGR {fmt_pct(float(eq[-1] ** (1 / yrs) - 1), 2) if yrs > 0 else 'n/a'}")
    print(f"  peor caída {fmt_pct(dd, 1)}   días en verde {fmt_pct(float((x > 0).mean()), 1)}")
    t_stat = float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if x.std(ddof=1) > 0 else np.nan
    print(f"  t-stat    {fmt_num(t_stat, 2)}  (>2 para tomárselo en serio)")

    # --- año a año: ¿sigue vivo o se apagó? ---------------------------------
    print("\n  año a año:")
    for y, g in port.groupby(port.index.year):
        if len(g) < 30:
            continue
        bar = ("+" if g.sum() > 0 else "-") * min(int(abs(100 * g.sum())), 40)
        print(f"    {y}  {fmt_pct(float((1 + g).prod() - 1), 2):>9}  "
              f"Sharpe {fmt_num(sharpe(g.to_numpy()), 2):>6}  {bar}")

    print("=" * 74)

    if args.out:
        res = {"sharpe": sharpe(port), "corr_media": rho,
               "sharpe_medio_individual": float(np.mean(ind)),
               "sharpe_teorico_indep": teo_indep, "n_valores": int(k),
               "retorno_total": float(eq[-1] - 1), "max_dd": dd,
               "t_stat": t_stat, "n_dias": int(len(x))}
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        log.info(f"resumen guardado en {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
