#!/usr/bin/env python3
"""Ensayo del camino de producción sobre una sesión YA CERRADA.

Para qué sirve: comprobar que todo funciona —descarga, ancla intradía real,
features, modelo, señal— sin esperar a que abra el mercado. Reconstruye el
precio del ancla de una sesión pasada a partir de las barras intradía reales
(yfinance guarda 60 días de 30m y ~730 de 1h) y genera la señal que se habría
publicado ese día. Si la sesión siguiente ya cerró, además dice si acertó.

Para qué NO sirve: para operar, ni para alimentar el marcador. La señal se
imprime y se descarta. NO se escribe en history.json ni en latest.json. El
marcador en vivo solo puede contener señales publicadas antes de conocerse el
resultado; si aceptara reconstrucciones dejaría de significar nada.

    python scripts/rehearse.py --universe universe.json --models models --back 1
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from alphaforge.backtest import position_from_prob
from alphaforge.config import Config
from alphaforge.data import load_market_data
from alphaforge.features import build_features
from alphaforge.labeling import build_labels
from alphaforge.pipeline import _coherence_gate, _ensemble_prob, _apply_calibrator, _proba
from alphaforge.utils import fmt_num, fmt_pct, get_logger, suppress_noisy_warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict_daily import load_model                       # noqa: E402


def rehearse_one(ticker: str, blob: dict, back: int, verbose: int = 0) -> dict:
    cfg = Config.from_dict(blob["cfg"])
    cfg.data.ticker = ticker
    cfg.verbose = verbose

    md = load_market_data(cfg, verbose=verbose)
    X = build_features(md, cfg)
    labels = build_labels(md, cfg)
    art = blob["final"]["artifacts"]

    if back >= len(X):
        raise ValueError(f"solo hay {len(X)} sesiones disponibles")
    pos_i = len(X) - 1 - back
    day = X.index[pos_i]
    row = X.iloc[[pos_i]]

    missing = [f for f in art.features if f not in row.columns]
    if missing:
        raise RuntimeError(f"faltan {len(missing)} features; reentrena el modelo")

    Z = art.imputer.transform(row[art.features])
    P = np.column_stack([_apply_calibrator(art.calibrators[f], _proba(art.models[f], Z))
                         for f in art.families])
    p_up = float(_ensemble_prob(art, P)[0])

    sigma = float(labels.sigma_ex_ante.iloc[pos_i])
    q = {}
    if art.quantiles is not None:
        qq = art.quantiles.predict(pd.DataFrame(Z, columns=art.features, index=row.index))
        q = {k: float(v.iloc[0]) * sigma for k, v in qq.items()}

    s = position_from_prob(pd.Series([p_up], index=row.index), cfg,
                           expected_ret=pd.Series([q.get("q50", 0.0)], index=row.index),
                           sigma=pd.Series([sigma], index=row.index))
    s, _ = _coherence_gate(s, pd.DataFrame({"ret_q50": [q.get("q50", np.nan)]},
                                           index=row.index), cfg)
    position = float(s.iloc[0])

    real = labels.y_ret.iloc[pos_i]
    real = float(real) if np.isfinite(real) else None

    return {
        "ticker": ticker, "date": str(day.date()),
        "anchor_price": float(md.anchor.price.iloc[pos_i]),
        "anchor_source": md.anchor.source,
        "anchor_real": md.anchor.coverage > 0
        and md.anchor.source != "close_proxy",
        "prob_up": p_up, "expected_return": q.get("q50"),
        "p10": q.get("q10"), "p90": q.get("q90"),
        "position": position,
        "verdict": blob["verdict"]["decision"],
        "realized": real,
        "correct": None if real is None else bool((real > 0) == (p_up > 0.5)),
        "pnl": None if real is None else position * real,
        "in_range": None if (real is None or q.get("q10") is None)
        else bool(q["q10"] <= real <= q["q90"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="universe.json")
    ap.add_argument("--models", default="models")
    ap.add_argument("--back", type=int, default=1,
                    help="cuántas sesiones atrás (1 = la última cerrada)")
    args = ap.parse_args()

    suppress_noisy_warnings()
    log = get_logger(1)
    import json
    with open(args.universe) as f:
        uni = json.load(f)

    print("=" * 78)
    print(f"ENSAYO — señal reconstruida de hace {args.back} sesión(es)")
    print("Prueba del camino de producción. NO es operable y NO toca el marcador.")
    print("=" * 78)
    print(f"{'valor':<9}{'fecha':<12}{'P(sube)':>9}{'E[ret]':>9}{'pos':>7}"
          f"{'real':>9}{'ok':>5}{'rango':>7}")
    print("-" * 78)

    rows, n_ok, n_tot = [], 0, 0
    for t in uni["tickers"]:
        try:
            blob = load_model(args.models, t)
            if blob is None:
                print(f"{t:<9}sin modelo entrenado")
                continue
            r = rehearse_one(t, blob, args.back)
            rows.append(r)
            if r["correct"] is not None:
                n_tot += 1
                n_ok += int(r["correct"])
            print(f"{r['ticker']:<9}{r['date']:<12}{fmt_pct(r['prob_up'], 1):>9}"
                  f"{fmt_pct(r['expected_return'], 2):>9}{fmt_num(r['position'], 2):>7}"
                  f"{fmt_pct(r['realized'], 2) if r['realized'] is not None else '—':>9}"
                  f"{('sí' if r['correct'] else 'no') if r['correct'] is not None else '—':>5}"
                  f"{('sí' if r['in_range'] else 'no') if r['in_range'] is not None else '—':>7}")
        except Exception as e:                       # noqa: BLE001
            print(f"{t:<9}ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()

    print("-" * 78)
    if n_tot:
        print(f"Acierto en esta sesión: {n_ok}/{n_tot}")
        print("Una sola sesión no dice NADA sobre si el sistema funciona: con 4")
        print("valores, acertar 3 pasa por azar una de cada tres veces. Sirve para")
        print("comprobar que la maquinaria va, no para juzgar el modelo.")
    else:
        print("La sesión siguiente aún no ha cerrado: no hay resultado que comparar.")

    if rows and not rows[0]["anchor_real"]:
        print("\nAVISO: el ancla es un proxy del cierre, no un precio intradía real.")
        print("La reconstrucción no refleja lo que habrías visto a las 15:00 ET.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
