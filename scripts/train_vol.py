#!/usr/bin/env python3
"""Entrena modelos de VOLATILIDAD y los enfrenta a HAR-RV.

    python scripts/train_vol.py --tickers AAPL,MSFT,SPY --horizon 1
    python scripts/train_vol.py --universe universe.json --horizon 5

El listón no es acertar: es batir a HAR-RV, que con tres regresores lleva
quince años siendo dificilísimo de superar. Un R2 de 0.45 no significa nada si
HAR da 0.47 con la mitad de esfuerzo y ningún hiperparámetro.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from alphaforge.config import Config
from alphaforge.data import load_market_data
from alphaforge.utils import fmt_num, get_logger, suppress_noisy_warnings
from alphaforge.volatility import run_vol_experiment


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", help="lista separada por comas")
    ap.add_argument("--universe", default="universe.json")
    ap.add_argument("--horizon", type=int, default=1,
                    help="días de volatilidad a predecir (1 = mañana)")
    ap.add_argument("--estimator", default="yang_zhang",
                    choices=["yang_zhang", "garman_klass", "parkinson",
                             "rogers_satchell"])
    ap.add_argument("--out", default="docs/data/volatility.json")
    ap.add_argument("--start", default=None)
    args = ap.parse_args()

    suppress_noisy_warnings()
    log = get_logger(1)
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        base = {}
    else:
        with open(args.universe) as f:
            u = json.load(f)
        tickers, base = u["tickers"], u.get("config", {})

    out = {}
    for i, t in enumerate(tickers, 1):
        log.info(f"[{i}/{len(tickers)}] {t}")
        try:
            cfg = Config.from_dict(base) if base else Config()
            cfg.data.ticker = t
            if args.start:
                cfg.data.start = args.start
            md = load_market_data(cfg, verbose=0)
            r = run_vol_experiment(md, cfg, args.horizon, args.estimator)
            m = r.metrics
            out[t] = {"verdict": r.verdict["decision"], "metrics": _clean(m),
                      "checks": r.verdict["checks"], "folds": r.fold_metrics,
                      "horizon": args.horizon, "estimator": args.estimator}
        except Exception as e:                       # noqa: BLE001
            log.error(f"  {t}: {type(e).__name__}: {e}")
            traceback.print_exc()

    if not out:
        return 1

    print("\n" + "=" * 78)
    print(f"VOLATILIDAD A {args.horizon} DÍA(S) — estimador {args.estimator}")
    print("=" * 78)
    print(f"{'valor':<8}{'R2 modelo':>11}{'R2 HAR':>9}{'mejora':>9}{'QLIKE':>9}"
          f"{'DM p':>8}{'peso':>7}{'veredicto':>11}")
    print("-" * 78)
    for t, v in sorted(out.items(), key=lambda x: -(x[1]["metrics"]["r2_vs_har"] or -9)):
        m = v["metrics"]
        print(f"{t:<8}{fmt_num(m['r2_vs_media']):>11}{fmt_num(m['har_r2_vs_media']):>9}"
              f"{fmt_num(m['r2_vs_har']):>9}"
              f"{fmt_num(m['qlike_mejora'], 3):>9}{fmt_num(m['dm_pvalue'], 4):>8}"
              f"{fmt_num(m.get('shrink_medio'), 2):>7}{v['verdict']:>11}")
    print("-" * 78)
    mej = [v["metrics"]["r2_vs_har"] for v in out.values()
           if v["metrics"]["r2_vs_har"] is not None]
    ngo = sum(1 for v in out.values() if v["verdict"] == "GO")
    print(f"mejora media sobre HAR: {np.mean(mej):+.4f} | "
          f"{sum(1 for x in mej if x > 0)}/{len(mej)} por encima | {ngo} con GO")
    print("La columna que decide es 'mejora': R2 frente a HAR, no frente a la media.")
    print("=" * 78)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"horizon": args.horizon, "estimator": args.estimator,
                   "models": out}, f, indent=2, ensure_ascii=False)
    log.info(f"resultados en {args.out}")
    return 0


def _clean(d):
    o = {}
    for k, v in d.items():
        if isinstance(v, dict):
            o[k] = _clean(v)
        elif isinstance(v, (int, str, bool)) or v is None:
            o[k] = v
        else:
            f = float(v)
            o[k] = None if not np.isfinite(f) else round(f, 6)
    return o


if __name__ == "__main__":
    sys.exit(main())
