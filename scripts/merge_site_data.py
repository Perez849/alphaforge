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

    _compare_variants(models, log)

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


RANK = {"GO": 2, "REVISAR": 1, "NO-GO": 0}


def _compare_variants(models: dict, log) -> None:
    """Enfrenta las variantes del mismo valor y nombra una ganadora.

    Comparar hipótesis entrenadas con el mismo código, los mismos datos y el
    mismo día es la única forma de que la comparación signifique algo. El
    criterio no es el Sharpe: es la ventaja sobre la TASA BASE, porque un
    Sharpe alto puede venir sin más de estar largo en un mercado que sube.
    """
    grupos: dict[str, list] = {}
    for key, m in models.items():
        base = m.get("base_ticker") or key.split("::")[0]
        grupos.setdefault(base, []).append((key, m))
    if not any(len(v) > 1 for v in grupos.values()):
        return

    log.info("comparación de variantes (ventaja sobre la tasa base):")
    header = f"  {'valor':<8}{'variante':<9}{'acierto':>9}{'tasa base':>11}" \
             f"{'ventaja':>10}{'Sharpe':>9}{'vs B&H':>9}{'veredicto':>10}"
    log.info(header)
    for base, items in sorted(grupos.items()):
        if len(items) < 2:
            continue
        best, best_edge = None, -9e9
        for key, m in sorted(items, key=lambda x: x[0]):
            h = m.get("headline", {})
            da = h.get("directional_accuracy")
            br = h.get("base_rate_up")
            edge = (da - br) if (da is not None and br is not None) else None
            sh, bh = h.get("sharpe"), h.get("bh_sharpe")
            ratio = (sh / bh) if (sh is not None and bh) else None
            f = lambda x, n=2, suf="": "—" if x is None else f"{x:.{n}f}{suf}"
            log.info(f"  {base:<8}{m.get('variant', '?'):<9}"
                     f"{f(100 * da if da is not None else None, 2, '%'):>9}"
                     f"{f(100 * br if br is not None else None, 2, '%'):>11}"
                     f"{f(100 * edge if edge is not None else None, 2, 'pp'):>10}"
                     f"{f(sh):>9}{f(ratio):>9}{m.get('verdict', '?'):>10}")
            score = (RANK.get(m.get("verdict"), 0), edge if edge is not None else -9e9)
            if score > (RANK.get(models[best].get("verdict"), 0) if best else -1,
                        best_edge):
                best, best_edge = key, (edge if edge is not None else -9e9)
        for key, m in items:
            m["variant_winner"] = (key == best)
        log.info(f"  {'':8}-> gana '{models[best].get('variant')}' para {base}")


if __name__ == "__main__":
    sys.exit(main())
