#!/usr/bin/env python3
"""Reúne los resultados de volatilidad de todos los jobs y publica el resumen.

    python scripts/merge_vol.py --parts artifacts --out docs/data/volatility.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from alphaforge.utils import fmt_num, get_logger


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="artifacts")
    ap.add_argument("--out", default="docs/data/volatility.json")
    args = ap.parse_args()

    log = get_logger(1)
    # se conserva lo publicado: un valor que hoy no se mide no desaparece
    runs: dict = {}
    if os.path.exists(args.out):
        try:
            with open(args.out) as f:
                runs = json.load(f).get("runs", {})
        except Exception as e:                             # noqa: BLE001
            log.warning(f"volatility.json previo ilegible ({e})")

    n_new = 0
    for path in sorted(glob.glob(os.path.join(args.parts, "**", "*.json"),
                                 recursive=True)):
        try:
            with open(path) as f:
                blob = json.load(f)
        except Exception as e:                             # noqa: BLE001
            log.warning(f"trozo ilegible {path}: {e}")
            continue
        h = blob.get("horizon", 1)
        for t, v in (blob.get("models") or {}).items():
            runs[f"{t}@h{h}"] = {**v, "ticker": t, "horizon": h}
            n_new += 1

    if not runs:
        log.error("no hay resultados de volatilidad que publicar")
        return 1

    filas = sorted(runs.values(),
                   key=lambda v: -(v["metrics"].get("r2_vs_har") or -9))
    print("\n" + "=" * 82)
    print("VOLATILIDAD — el listón es batir a HAR-RV, no acertar")
    print("=" * 82)
    print(f"{'valor':<8}{'h':>3}{'R2 modelo':>11}{'R2 HAR':>9}{'MEJORA':>9}"
          f"{'QLIKE':>9}{'DM p':>8}{'peso':>7}{'veredicto':>11}")
    print("-" * 82)
    for v in filas:
        m = v["metrics"]
        print(f"{v['ticker']:<8}{v['horizon']:>3}{fmt_num(m.get('r2_vs_media')):>11}"
              f"{fmt_num(m.get('har_r2_vs_media')):>9}{fmt_num(m.get('r2_vs_har')):>9}"
              f"{fmt_num(m.get('qlike_mejora'), 3):>9}"
              f"{fmt_num(m.get('dm_pvalue'), 4):>8}"
              f"{fmt_num(m.get('shrink_medio'), 2):>7}{v.get('verdict', '?'):>11}")
    print("-" * 82)

    resumen = {}
    for h in sorted({v["horizon"] for v in runs.values()}):
        sub = [v for v in runs.values() if v["horizon"] == h]
        mej = [v["metrics"].get("r2_vs_har") for v in sub
               if v["metrics"].get("r2_vs_har") is not None]
        sig = [v for v in sub
               if (v["metrics"].get("dm_pvalue") or 1) < 0.05
               and (v["metrics"].get("r2_vs_har") or 0) > 0]
        go = [v for v in sub if v.get("verdict") == "GO"]
        resumen[f"h{h}"] = {
            "n": len(sub),
            "mejora_media_sobre_har": round(float(np.mean(mej)), 5) if mej else None,
            "n_por_encima": int(sum(1 for x in mej if x > 0)),
            "n_significativos": len(sig), "n_go": len(go),
            "tickers_go": [v["ticker"] for v in go],
        }
        print(f"  horizonte {h}d: mejora media {fmt_num(resumen[f'h{h}']['mejora_media_sobre_har'], 4)} | "
              f"{resumen[f'h{h}']['n_por_encima']}/{len(mej)} por encima de HAR | "
              f"{len(sig)} significativos | {len(go)} con GO")

    print("\n  R2 modelo alto con MEJORA cero significa que HAR ya lo explicaba todo.")
    print("  Lo único que cuenta es la columna MEJORA con su DM p por debajo de 0.05.")
    print("=" * 82)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "runs": runs, "resumen": resumen}, f, indent=2, ensure_ascii=False)
    log.info(f"publicados {len(runs)} resultados ({n_new} nuevos) en {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
