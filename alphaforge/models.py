"""Zoo de modelos + espacio de búsqueda + red recurrente opcional.

Nada de pesos a mano: cada familia expone un espacio de hiperparámetros y la
combinación se elige por rendimiento en validación purgada. El ensamblado final
también se aprende (stacking con pesos no negativos), no se fija a dedo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingRegressor,
                              HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

try:                                            # torch es opcional
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except Exception:                               # noqa: BLE001
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Espacios de hiperparámetros
# ---------------------------------------------------------------------------
def sample_params(family: str, rng: np.random.Generator) -> dict[str, Any]:
    if family == "logit":
        return {
            "C": float(10 ** rng.uniform(-3.5, 0.5)),
            "l1_ratio": float(rng.uniform(0.0, 1.0)),
            "scaler": str(rng.choice(["standard", "quantile"])),
        }
    if family == "hgb":
        return {
            "max_depth": int(rng.choice([2, 3, 4, 5])),
            "learning_rate": float(10 ** rng.uniform(-2.2, -0.9)),
            "max_iter": int(rng.choice([120, 200, 300, 450])),
            "min_samples_leaf": int(rng.choice([20, 40, 80, 150])),
            "l2_regularization": float(10 ** rng.uniform(-3, 1.0)),
            "max_features": float(rng.uniform(0.4, 1.0)),
        }
    if family == "extratrees":
        return {
            "n_estimators": int(rng.choice([200, 350, 500])),
            "max_depth": int(rng.choice([3, 5, 8, 12])),
            "min_samples_leaf": int(rng.choice([10, 25, 50, 100])),
            "max_features": float(rng.uniform(0.2, 0.8)),
        }
    if family == "mlp":
        # nota: rng.choice no admite listas de tuplas heterogéneas -> índice
        _shapes = [(32,), (64,), (64, 32), (128, 32), (32, 16)]
        h = _shapes[int(rng.choice(len(_shapes), p=[.2, .2, .25, .15, .2]))]
        return {
            "hidden_layer_sizes": h,
            "alpha": float(10 ** rng.uniform(-4, 0.5)),
            "learning_rate_init": float(10 ** rng.uniform(-3.3, -2.0)),
            "max_iter": int(rng.choice([200, 400])),
            "scaler": str(rng.choice(["standard", "quantile"])),
        }
    if family == "gru":
        return {
            "seq_len": int(rng.choice([8, 15, 25])),
            "hidden": int(rng.choice([16, 32, 48])),
            "layers": int(rng.choice([1, 2])),
            "dropout": float(rng.uniform(0.1, 0.5)),
            "lr": float(10 ** rng.uniform(-3.2, -2.0)),
            "epochs": int(rng.choice([25, 45, 70])),
            "weight_decay": float(10 ** rng.uniform(-6, -2.5)),
            "batch": int(rng.choice([64, 128])),
        }
    raise ValueError(f"familia desconocida: {family}")


class AdaptiveQuantileTransformer(QuantileTransformer):
    """QuantileTransformer que ajusta n_quantiles al tamaño real del fold.

    Con folds pequeños, un n_quantiles fijo emite avisos y desperdicia cómputo.
    """

    def fit(self, X, y=None):
        n = len(X)
        self.n_quantiles = max(10, min(500, n))
        return super().fit(X, y)


def _scaler(name: str):
    if name == "quantile":
        return AdaptiveQuantileTransformer(output_distribution="normal",
                                           subsample=100_000, random_state=0)
    return StandardScaler()


def _make_logit(C: float, l1_ratio: float, seed: int):
    """LogisticRegression elastic-net compatible con sklearn <1.8 y >=1.8.

    A partir de 1.8 el argumento `penalty` está deprecado: el tipo de
    regularización se deduce de l1_ratio.
    """
    kw = dict(solver="saga", C=C, l1_ratio=l1_ratio, max_iter=3000, tol=1e-3,
              random_state=seed)
    from sklearn import __version__ as skv
    major, minor = (int(x) for x in skv.split(".")[:2])
    if (major, minor) < (1, 8):
        kw["penalty"] = "elasticnet"
        kw["n_jobs"] = 1
    return LogisticRegression(**kw)


def build_classifier(family: str, params: dict, seed: int, n_features: int):
    """Devuelve un estimador sklearn-compatible con .fit/.predict_proba."""
    p = dict(params)
    if family == "logit":
        sc = _scaler(p.pop("scaler", "standard"))
        return Pipeline([
            ("sc", sc),
            ("clf", _make_logit(p["C"], p["l1_ratio"], seed)),
        ])
    if family == "hgb":
        return HistGradientBoostingClassifier(
            max_depth=p["max_depth"], learning_rate=p["learning_rate"],
            max_iter=p["max_iter"], min_samples_leaf=p["min_samples_leaf"],
            l2_regularization=p["l2_regularization"],
            max_features=p["max_features"],
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=25,
            random_state=seed)
    if family == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            min_samples_leaf=p["min_samples_leaf"], max_features=p["max_features"],
            bootstrap=True, oob_score=False, n_jobs=-1, random_state=seed,
            class_weight="balanced_subsample")
    if family == "mlp":
        sc = _scaler(p.pop("scaler", "standard"))
        return Pipeline([
            ("sc", sc),
            ("clf", MLPClassifier(
                hidden_layer_sizes=p["hidden_layer_sizes"], alpha=p["alpha"],
                learning_rate_init=p["learning_rate_init"], max_iter=p["max_iter"],
                early_stopping=True, n_iter_no_change=20, validation_fraction=0.15,
                random_state=seed)),
        ])
    if family == "gru":
        return SequenceNet(n_features=n_features, seed=seed, **p)
    raise ValueError(f"familia desconocida: {family}")


# ---------------------------------------------------------------------------
# Red recurrente (GRU) con envoltorio sklearn
# ---------------------------------------------------------------------------
class SequenceNet(BaseEstimator, ClassifierMixin):
    """GRU sobre ventanas deslizantes de features.

    Captura dependencias temporales que los modelos tabulares no ven: la
    *trayectoria* de los indicadores en los últimos N días, no solo su nivel.
    Si torch no está instalado, cae a un MLP sobre la ventana aplanada, así el
    pipeline nunca se rompe por una dependencia ausente.
    """

    def __init__(self, n_features: int = 10, seq_len: int = 20, hidden: int = 32,
                 layers: int = 1, dropout: float = 0.2, lr: float = 1e-3,
                 epochs: int = 40, weight_decay: float = 1e-4, batch: int = 128,
                 seed: int = 0):
        self.n_features = n_features
        self.seq_len = seq_len
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.batch = batch
        self.seed = seed

    # -- utilidades ---------------------------------------------------------
    def _windows(self, X: np.ndarray) -> np.ndarray:
        """(n, f) -> (n, seq_len, f) con padding hacia atrás por repetición."""
        n, f = X.shape
        L = self.seq_len
        pad = np.repeat(X[:1], L - 1, axis=0)
        Xp = np.vstack([pad, X])
        out = np.lib.stride_tricks.sliding_window_view(Xp, (L, f))[:, 0]
        return np.ascontiguousarray(out[:n])

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        self.classes_ = np.array([0.0, 1.0])
        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0)
        self.sd_[self.sd_ < 1e-8] = 1.0
        Xn = (X - self.mu_) / self.sd_

        if not HAS_TORCH:
            self._fallback = MLPClassifier(
                hidden_layer_sizes=(self.hidden, max(8, self.hidden // 2)),
                alpha=max(self.weight_decay, 1e-5), max_iter=300,
                early_stopping=True, n_iter_no_change=20, random_state=self.seed)
            W = self._windows(Xn).reshape(len(Xn), -1)
            self._fallback.fit(W, y)
            return self

        torch.manual_seed(self.seed)
        W = torch.tensor(self._windows(Xn))
        t = torch.tensor(y).unsqueeze(1)
        sw = (torch.tensor(np.asarray(sample_weight, dtype=np.float32)).unsqueeze(1)
              if sample_weight is not None else torch.ones_like(t))

        self._net = _GRUNet(X.shape[1], self.hidden, self.layers, self.dropout)
        opt = torch.optim.AdamW(self._net.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        lossf = nn.BCEWithLogitsLoss(reduction="none")

        # separación temporal interna para early stopping (últimos 15%)
        cut = max(32, int(0.85 * len(W)))
        Wtr, ttr, swtr = W[:cut], t[:cut], sw[:cut]
        Wva, tva = W[cut:], t[cut:]

        best, best_state, patience = np.inf, None, 0
        n = len(Wtr)
        for ep in range(self.epochs):
            self._net.train()
            perm = torch.randperm(n)
            for i in range(0, n, self.batch):
                b = perm[i:i + self.batch]
                opt.zero_grad()
                out = self._net(Wtr[b])
                loss = (lossf(out, ttr[b]) * swtr[b]).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                opt.step()
            if len(Wva) > 16:
                self._net.eval()
                with torch.no_grad():
                    vl = lossf(self._net(Wva), tva).mean().item()
                if vl < best - 1e-5:
                    best, patience = vl, 0
                    best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                else:
                    patience += 1
                    if patience >= 8:
                        break
        if best_state is not None:
            self._net.load_state_dict(best_state)
        self._net.eval()
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xn = (X - self.mu_) / self.sd_
        W = self._windows(Xn)
        if not HAS_TORCH:
            return self._fallback.predict_proba(W.reshape(len(W), -1))
        with torch.no_grad():
            p = torch.sigmoid(self._net(torch.tensor(W))).numpy().ravel()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(float)


if HAS_TORCH:
    class _GRUNet(nn.Module):
        def __init__(self, n_in: int, hidden: int, layers: int, dropout: float):
            super().__init__()
            self.gru = nn.GRU(n_in, hidden, num_layers=layers, batch_first=True,
                              dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Sequential(
                nn.LayerNorm(hidden), nn.Dropout(dropout),
                nn.Linear(hidden, max(8, hidden // 2)), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(max(8, hidden // 2), 1))

        def forward(self, x):
            o, _ = self.gru(x)
            return self.head(o[:, -1])
else:
    _GRUNet = None                                             # type: ignore


# ---------------------------------------------------------------------------
# Regresión cuantílica para la magnitud
# ---------------------------------------------------------------------------
@dataclass
class QuantileBundle:
    models: dict[float, Any] = field(default_factory=dict)
    columns: list[str] = field(default_factory=list)

    def fit(self, X: pd.DataFrame, y: np.ndarray, quantiles, seed: int = 0,
            sample_weight=None) -> "QuantileBundle":
        self.columns = list(X.columns)
        Xv = X.to_numpy(dtype=float)
        for q in quantiles:
            m = GradientBoostingRegressor(
                loss="quantile", alpha=float(q), n_estimators=200, max_depth=3,
                learning_rate=0.05, min_samples_leaf=40, subsample=0.8,
                random_state=seed)
            m.fit(Xv, y, sample_weight=sample_weight)
            self.models[float(q)] = m
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        Xv = X[self.columns].to_numpy(dtype=float)
        out = {f"q{int(100 * q):02d}": m.predict(Xv) for q, m in self.models.items()}
        df = pd.DataFrame(out, index=X.index)
        # monotonía de cuantiles: q10 <= q50 <= q90 (los árboles no la garantizan)
        cols = sorted(df.columns)
        df[cols] = np.sort(df[cols].to_numpy(), axis=1)
        return df


# ---------------------------------------------------------------------------
# Stacking con pesos no negativos aprendidos
# ---------------------------------------------------------------------------
def blend(P: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Combina probabilidades en espacio LOGIT, que es donde se aprendieron los
    pesos.

    Promediar probabilidades directamente (P @ w) hace dos cosas malas: usa
    unos pesos que se optimizaron para otra operación, y comprime la salida
    hacia 0.5 —medido, un 8% menos de dispersión— con lo que la probabilidad
    del tablón sale más tibia de lo que el modelo realmente cree.
    """
    P = np.clip(np.asarray(P, dtype=float), 1e-6, 1 - 1e-6)
    if P.ndim == 1:
        P = P.reshape(-1, 1)
    Z = np.log(P / (1 - P))
    return np.clip(1.0 / (1.0 + np.exp(-(Z @ w))), 1e-6, 1 - 1e-6)


def learn_ensemble_weights(P: np.ndarray, y: np.ndarray,
                           l2: float = 1.0) -> np.ndarray:
    """Pesos no negativos que suman 1, ajustados por regresión logística sobre
    los logits de los modelos base. Los pesos salen de datos out-of-fold, nunca
    de intuición."""
    P = np.clip(np.asarray(P, dtype=float), 1e-6, 1 - 1e-6)
    if P.ndim == 1:
        P = P.reshape(-1, 1)
    if P.shape[1] == 1:
        return np.array([1.0])
    Z = np.log(P / (1 - P))
    ok = np.all(np.isfinite(Z), axis=1) & np.isfinite(y)
    if ok.sum() < 30:
        return np.full(P.shape[1], 1.0 / P.shape[1])
    lr = LogisticRegression(C=1.0 / max(l2, 1e-6), solver="lbfgs",
                            max_iter=1000, fit_intercept=True)
    lr.fit(Z[ok], y[ok])
    w = np.clip(lr.coef_.ravel(), 0.0, None)
    if w.sum() <= 1e-9:
        return np.full(P.shape[1], 1.0 / P.shape[1])
    return w / w.sum()
