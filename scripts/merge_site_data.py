#!/usr/bin/env python3
"""Junta los resultados de entrenamientos ejecutados en paralelo.

Cuando cada valor se entrena en su propio job, cada uno deja su trozo suelto.
Este script los reúne en el `backtest.json` y el `index.json` que consume la web,
conservando lo que ya hubiera publicado de valores que hoy no se reentrenaron.

    python scripts/merge_site_data.py --parts artifacts --out docs/data \
                                      --models models
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alphaforge.utils import get_logger                    # noqa: E402
from train_all import _portfolio                           # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="artifacts",
                    help="carpeta con los trozos descargados de cada job")
    ap.add_argument("--out", default="docs/data")
    ap.add_argument("--models", default="models")
    args = ap.parse_args()

    log = get_logger(1)
    os.makedirs(args.out, exist_ok=True)

    # lo ya publicado se conserva: un valor que hoy no se entrena no desaparece
    site = os.path.join(args.out, "backtest.json")
    models: dict = {}
    if os.path.exists(site):
        try:
            with open(site) as f:
                models = json.load(f).get("models", {})
            log.info(f"partiendo de {len(models)} modelos ya publicados")
        except Exception as e:                             # noqa: BLE001
            log.warning(f"backtest.json previo ilegible ({e})")

    entries, failures, n_new = [], [], 0
    for root, _, files in os.walk(args.parts):
        for fn in sorted(files):
            path = os.path.join(root, fn)
            if fn == "summary.json":
                try:
                    with open(path) as f:
                        blob = json.load(f)
                    for tk, summary in blob.get("models", {}).items():
                        models[tk] = summary
                        n_new += 1
                    entries.extend(blob.get("index", []))
                    failures.extend(blob.get("failures", []))
                except Exception as e:                     # noqa: BLE001
                    log.warning(f"trozo ilegible {path}: {e}")

    if not models:
        log.error("no hay ningún modelo que publicar")
        return 1

    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(site, "w") as f:
        json.dump({"generated": generated, "models": models,
                   "portfolio": _portfolio(models)}, f, indent=2, ensure_ascii=False)

    os.makedirs(args.models, exist_ok=True)
    with open(os.path.join(args.models, "index.json"), "w") as f:
        json.dump({"generated": generated, "models": entries,
                   "failures": failures,
                   "n_go": sum(1 for e in entries if e.get("verdict") == "GO")},
                  f, indent=2, ensure_ascii=False)

    go = [t for t, m in models.items() if m.get("verdict") == "GO"]
    rev = [t for t, m in models.items() if m.get("verdict") == "REVISAR"]
    log.info(f"publicados {len(models)} modelos ({n_new} nuevos hoy), "
             f"{len(failures)} fallos")
    log.info(f"  GO:      {', '.join(go) if go else '(ninguno)'}")
    log.info(f"  REVISAR: {', '.join(rev) if rev else '(ninguno)'}")
    if failures:
        for fl in failures:
            log.warning(f"  falló {fl.get('ticker')}: {fl.get('error')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
