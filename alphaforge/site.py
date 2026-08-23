"""Resumen del walk-forward listo para publicar en la web.

La idea: no hay que esperar dos meses a acumular señales en vivo para saber si
un modelo vale. El walk-forward YA produjo miles de predicciones fuera de
muestra — cada día del backtest fue predicho por un modelo entrenado solo con
datos anteriores. Eso es fiabilidad medible desde el primer entrenamiento.

Lo que sí hay que hacer es no mentir sobre su origen. Por eso todo lo que sale
de aquí va etiquetado como `walk-forward`, separado del marcador en vivo. Las
dos diferencias que importan:

  * los hiperparámetros se eligieron una vez mirando el bloque inicial
  * en la parte antigua del histórico el ancla es un proxy del precio a T−60′

Ambas están cuantificadas en el informe y ninguna se esconde en el tablón.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .report import _jsonable
from .validation import calibration_report


def oos_summary(res, max_recent: int = 120) -> dict:
    """Empaqueta el resultado del walk-forward para el tablón."""
    oos = res.oos.copy()
    m = res.metrics
    cfg = res.cfg

    # --- curva de resultados, submuestreada para que el JSON no engorde
    eq = oos["equity"].dropna()
    bh = (1 + oos["y_ret"].fillna(0)).cumprod()
    step = max(1, len(eq) // 260)
    curve = [{"d": str(d.date()), "e": round(float(v), 4), "b": round(float(bh.loc[d]), 4)}
             for d, v in eq.iloc[::step].items()]
    if len(eq) and curve and curve[-1]["d"] != str(eq.index[-1].date()):
        curve.append({"d": str(eq.index[-1].date()), "e": round(float(eq.iloc[-1]), 4),
                      "b": round(float(bh.iloc[-1]), 4)})

    # --- la prueba de fiabilidad más directa que existe:
    #     cuando el modelo dice 60%, ¿sube el 60% de las veces?
    cal = calibration_report(oos["y_cls"].to_numpy(), oos["prob_up"].to_numpy(), bins=10)

    # --- rendimiento por año natural: revela si el edge se apagó hace tiempo
    by_year = []
    for y, g in oos.groupby(oos.index.year):
        net = g["net_ret"].dropna()
        if len(net) < 20:
            continue
        sd = float(net.std(ddof=1))
        act = g["position"].abs() > 1e-9
        by_year.append({
            "year": int(y), "n": int(len(net)), "n_trades": int(act.sum()),
            "return": round(float((1 + net).prod() - 1), 4),
            "sharpe": round(float(net.mean() / sd * np.sqrt(252)), 2) if sd > 1e-12 else None,
            "hit": round(float((net[act] > 0).mean()), 3) if act.sum() else None,
        })

    # --- últimas señales OOS con su resultado real: el "papel trading" ya hecho
    recent = []
    for d, r in oos.tail(max_recent).iterrows():
        recent.append({
            "date": str(d.date()),
            "prob_up": round(float(r["prob_up"]), 4),
            "position": round(float(r.get("position", 0.0)), 3),
            "realized": round(float(r["y_ret"]), 5) if np.isfinite(r["y_ret"]) else None,
            "pnl": round(float(r["net_ret"]), 5) if np.isfinite(r.get("net_ret", np.nan)) else None,
            "p10": _q(r, "ret_q10"), "p50": _q(r, "ret_q50"), "p90": _q(r, "ret_q90"),
            "correct": bool((r["y_ret"] > 0) == (r["prob_up"] > 0.5))
            if np.isfinite(r["y_ret"]) else None,
        })

    # cobertura real del intervalo P10-P90: debería rondar el 80%
    cov = None
    if {"ret_q10", "ret_q90"} <= set(oos.columns):
        inside = ((oos["y_ret"] >= oos["ret_q10"]) & (oos["y_ret"] <= oos["ret_q90"]))
        cov = round(float(inside.mean()), 3)

    act = oos["position"].abs() > 1e-9
    return _jsonable({
        "ticker": cfg.data.ticker,
        "verdict": res.verdict["decision"],
        "checks": res.verdict["checks"],
        "period": {"start": str(oos.index.min().date()), "end": str(oos.index.max().date()),
                   "days": int(len(oos)), "years": round(len(oos) / 252, 1)},
        "headline": {
            "sharpe": _r(m.get("sharpe_overlap_adj", m.get("sharpe")), 2),
            "sharpe_raw": _r(m.get("sharpe"), 2),
            "overlap_frac": _r(m.get("overlap_frac"), 3),
            "autocorr_1": _r(m.get("autocorr_1"), 3),
            "auc": _r(m.get("auc_oos"), 4),
            "cagr": _r(m.get("cagr"), 4), "max_dd": _r(m.get("max_dd"), 4),
            "hit_rate": _r(m.get("hit_rate"), 4),
            "directional_accuracy": _r(m.get("directional_accuracy"), 4),
            # El listón real: qué fracción de días sube el valor. Sin esto, un
            # 56% de acierto parece señal cuando la tasa base ya es del 56%.
            "base_rate_up": _r(m.get("base_rate_up"), 4),
            "edge_over_base": _r(m.get("edge_over_base"), 4),
            "profit_factor": _r(m.get("profit_factor"), 2),
            "n_trades": int(act.sum()), "exposure": _r(m.get("exposure"), 3),
            "total_return": _r(m.get("total_return"), 4),
            "bh_sharpe": _r(m.get("buy_hold", {}).get("sharpe"), 2),
            "bh_total_return": _r(m.get("buy_hold", {}).get("total_return"), 4),
        },
        "executable": {
            "sharpe": _r((m.get("continuous") or {}).get("sharpe"), 2),
            "total_return": _r((m.get("continuous") or {}).get("total_return"), 4),
            "max_dd": _r((m.get("continuous") or {}).get("max_dd"), 4),
            "note": "Retornos ancla->ancla, sin tramos solapados.",
        },
        "trust": {
            "pbo": _r(res.pbo.pbo, 3) if res.pbo else None,
            "deflated_sharpe": _r(m.get("deflated_sharpe"), 3),
            "p_permutation": _r(res.permutation.get("p_sharpe"), 4),
            "folds_positive": _r(m.get("stability", {}).get("frac_positive"), 3),
            "n_folds": m.get("stability", {}).get("n_folds"),
            "brier_skill": _r(cal.get("brier_skill"), 4),
            "ece": _r(cal.get("ece"), 4),
            "interval_coverage": cov,
            "anchor_source": m.get("anchor_source"),
            "anchor_coverage": _r(m.get("anchor_coverage"), 3),
            "anchor_stress_p05": _r(m.get("sharpe_anchor_stress_p05"), 2),
            "n_configs_tried": res.search.get("n_trials"),
        },
        "calibration": cal.get("curve", []),
        "curve": curve,
        "by_year": by_year,
        "by_fold": [{"fold": f["fold"], "start": f["test_start"], "end": f["test_end"],
                     "sharpe": _r(f.get("sharpe"), 2), "auc": _r(f.get("auc"), 3),
                     "return": _r(f.get("total_return"), 4)}
                    for f in res.fold_metrics],
        "recent": recent,
    })


def _q(row, col):
    v = row.get(col, np.nan)
    return round(float(v), 5) if v is not None and np.isfinite(v) else None


def _r(x, nd=4):
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else round(x, nd)
