#!/usr/bin/env python3
"""Comprueba las predicciones pasadas contra lo que ocurrió de verdad.

Se ejecuta cada mañana. Rellena el resultado real de cada señal y calcula el
acierto EN VIVO, que es distinto del backtest: aquí no hay reentrenamientos
convenientes ni parámetros elegidos a posteriori. Es el marcador honesto.

    python scripts/reconcile.py --out docs/data
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from alphaforge.config import Config, DataConfig
from alphaforge.data import download
from alphaforge.utils import get_logger, suppress_noisy_warnings

def _roundtrip_cost(universe: str = "universe.json") -> float:
    """Mismo coste que usa el backtest: si difieren, el marcador en vivo y el
    backtest dejan de ser comparables y no hay forma de saber cuál mirar."""
    try:
        with open(universe) as f:
            cfg = Config.from_dict(json.load(f).get("config", {}))
        return cfg.roundtrip_cost
    except Exception:                                # noqa: BLE001
        return Config().roundtrip_cost


COST_ROUNDTRIP = _roundtrip_cost()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/data")
    ap.add_argument("--lookback", type=int, default=180)
    ap.add_argument("--universe", default="universe.json")
    args = ap.parse_args()

    global COST_ROUNDTRIP
    COST_ROUNDTRIP = _roundtrip_cost(args.universe)

    suppress_noisy_warnings()
    log = get_logger(1)
    path = os.path.join(args.out, "history.json")
    if not os.path.exists(path):
        log.info("todavía no hay histórico que resolver")
        return 0

    with open(path) as f:
        hist = json.load(f)
    entries = hist.get("entries", [])[-args.lookback:]
    pending = [e for e in entries if not e.get("resolved")]
    if not pending:
        log.info("nada pendiente de resolver")

    # La sesión de salida tiene que estar CERRADA. Este script corre antes que
    # predict_daily, es decir con el mercado abierto: la última fila diaria de
    # yfinance es parcial y su "Close" es el precio del momento. Resolver con
    # ella falsearía el marcador en vivo, que es justo el número que no debe
    # mentir. Se descarta cualquier fila cuya sesión no haya terminado.
    now_et = pd.Timestamp.now(tz="America/New_York")
    session_over = now_et.hour >= 16 or now_et.dayofweek >= 5
    today_et = pd.Timestamp(now_et.date())
    last_closed = today_et if session_over else today_et - pd.Timedelta(days=1)
    log.info(f"resolviendo solo con sesiones cerradas (hasta {last_closed.date()} "
             f"inclusive; ahora son las {now_et:%H:%M} ET)")

    dc = DataConfig()
    closes: dict[str, pd.Series] = {}
    tickers = sorted({s["ticker"] for e in pending for s in e["signals"]})
    for t in tickers:
        try:
            d = download(t, "1d", dc, use_cache=False, verbose=0)
            d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
            c = d["Close"]
            n_before = len(c)
            c = c[c.index <= last_closed]          # fuera la fila parcial de hoy
            if len(c) < n_before:
                log.debug(f"{t}: descartadas {n_before - len(c)} filas sin cerrar")
            closes[t] = c
        except Exception as e:                       # noqa: BLE001
            log.warning(f"{t}: sin datos para resolver ({e})")

    n_res = 0
    for e in pending:
        day = pd.Timestamp(e["date"])
        done = True
        for s in e["signals"]:
            c = closes.get(s["ticker"])
            if c is None:
                done = False
                continue
            after = c[c.index > day]
            if after.empty:                          # el día siguiente aún no cerró
                done = False
                continue
            exit_day = after.index[0]
            gap = (exit_day - day).days
            if gap > 5:            # hueco de cotización: la comparación no es válida
                s["skipped"] = f"hueco de {gap} días hasta el precio de salida"
                continue
            exit_px = float(after.iloc[0])
            entry = float(s["anchor_price"])
            real = exit_px / entry - 1 if entry > 0 else None
            if real is None:
                done = False
                continue
            s["exit_date"] = str(exit_day.date())
            s["realized_return"] = round(real, 5)
            s["realized_up"] = bool(real > 0)
            s["predicted_up"] = bool(s["prob_up"] > 0.5)
            s["correct"] = bool(s["realized_up"] == s["predicted_up"])
            pos = float(s.get("position") or 0.0)
            s["pnl"] = round(pos * real - abs(pos) * COST_ROUNDTRIP, 5)
            s["in_interval"] = (
                None if s.get("ret_p10") is None
                else bool(s["ret_p10"] <= real <= s["ret_p90"]))
            n_res += 1
        e["resolved"] = done

    stats = _scoreboard(entries)
    hist["entries"] = entries
    hist["scoreboard"] = stats
    hist["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(path, "w") as f:
        json.dump(hist, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.out, "scoreboard.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    log.info(f"resueltas {n_res} señales | acierto global "
             f"{100 * (stats['overall']['hit_rate'] or 0):.1f}% "
             f"sobre {stats['overall']['n']} señales")
    return 0


def _scoreboard(entries: list) -> dict:
    """Marcador en vivo: global, solo operables y por valor."""
    rows = [dict(s, date=e["date"]) for e in entries for s in e["signals"]
            if "correct" in s]

    def agg(sel: list) -> dict:
        if not sel:
            return {"n": 0, "hit_rate": None, "avg_pnl": None, "cum_pnl": None,
                    "interval_coverage": None, "avg_prob": None}
        pnl = np.array([s.get("pnl") or 0.0 for s in sel], dtype=float)
        cov = [s["in_interval"] for s in sel if s.get("in_interval") is not None]
        return {
            "n": len(sel),
            "hit_rate": round(float(np.mean([s["correct"] for s in sel])), 4),
            "avg_pnl": round(float(pnl.mean()), 5),
            "cum_pnl": round(float(np.prod(1 + pnl) - 1), 5),
            "interval_coverage": round(float(np.mean(cov)), 3) if cov else None,
            "avg_prob": round(float(np.mean([s["prob_up"] for s in sel])), 3),
        }

    by_ticker = {}
    for t in sorted({s["ticker"] for s in rows}):
        by_ticker[t] = agg([s for s in rows if s["ticker"] == t])

    # curva acumulada solo con las señales operables
    curve, eq = [], 1.0
    for e in entries:
        day = [s for s in e["signals"] if s.get("operable") and "pnl" in s]
        if not day:
            continue
        eq *= (1 + float(np.mean([s["pnl"] for s in day])))
        curve.append({"date": e["date"], "equity": round(eq, 5)})

    return {
        "overall": agg(rows),
        "operable_only": agg([s for s in rows if s.get("operable")]),
        "by_ticker": by_ticker,
        "equity_curve": curve,
        "note": ("Acierto medido en vivo sobre señales publicadas antes de "
                 "conocerse el resultado. Sin reentrenamientos retroactivos."),
    }


if __name__ == "__main__":
    sys.exit(main())
