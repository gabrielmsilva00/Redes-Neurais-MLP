"""mlp.py – treinamento, validação e inspeção visual de Multilayer-Perceptron
interativo a partir de arquivos CSV.

Principais melhorias
────────────────────
• Navegação por folds via lista de botões – o lineplot e as setas foram removidos.  
• Ao carregar um novo CSV no visualizador, cada fold é avaliado apenas sobre
  as amostras correspondentes ao respectivo subconjunto de validação usado
  no treinamento, evitando que todas as folds processem o dataset completo.
"""

from __future__ import annotations

import sys, copy, logging, platform, shutil, warnings
from enum          import Enum
from pathlib       import Path
from argparse      import ArgumentParser, Namespace
from dataclasses   import asdict, dataclass
from time          import perf_counter, strftime
from typing        import Optional, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
from matplotlib.widgets  import Button
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute       import IterativeImputer, SimpleImputer
from sklearn.metrics      import (
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network  import MLPClassifier
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler

# ──────────────────────────────────────────────────────────────────────────────
# ENUMS & CONFIGURAÇÕES
# ──────────────────────────────────────────────────────────────────────────────
class BestBy(str, Enum):
    VAL_LOSS        = "min"
    FALSE_NEGATIVE  = "min"
    FALSE_POSITIVE  = "min"
    TRUE_POSITIVE   = "max"
    TRUE_NEGATIVE   = "max"
    DIFF_POSITIVE   = "max"
    DIFF_NEGATIVE   = "max"
    F1_SCORE        = "max"
    ACCURACY        = "max"
    PRECISION       = "max"
    RECALL          = "max"

class FillStrategy(Enum):
    MEAN    = 1
    MEDIAN  = 2
    MODE    = 3
    ITER    = 4
    KEEP    = 5
    ZERO    = 6

class Activation(str, Enum):
    RELU    = "relu"
    TANH    = "tanh"
    LINEAR  = "identity"
    SIGMOID = "logistic"

    @classmethod
    def from_str(cls, name: str) -> "Activation":
        try:    return cls[name.upper()]
        except KeyError as exc:
            raise ValueError(f"Invalid activation: {name}") from exc

@dataclass(frozen=True, slots=True)
class HyperParams:
    k_folds       : int     = 5
    epochs        : int     = 2048
    patience      : int     = 64
    min_delta     : float   = 1e-4
    watch_for     : BestBy  = BestBy.F1_SCORE
    batch_size    : int     = 64
    learning_rate : float   = 1e-3
    layers        : tuple[tuple[Activation, int], ...] = (
        (Activation.TANH, 7),
        (Activation.TANH, 7),
        (Activation.SIGMOID, 1),
    )

@dataclass(frozen=True, slots=True)
class Config:
    csv_sep       : str             = ";"
    target_cols   : tuple[int, ...] = (-1,)
    scramble_rows : bool            = True
    normalize     : bool            = True
    null_value    : Optional[float] = 0
    fill_strategy : FillStrategy    = FillStrategy.KEEP
    exclude_cols  : tuple[int, ...] = (0,)
    csv_dir       : Path            = Path(".data")
    img_dir       : Path            = Path(".img")
    mdl_dir       : Path            = Path(".models")
    log_dir       : Path            = Path(".log")

cfg, hp = Config(), HyperParams()
for d in (cfg.csv_dir, cfg.img_dir, cfg.mdl_dir, cfg.log_dir): d.mkdir(exist_ok=True)

_log = cfg.log_dir / f"{strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(filename=_log, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    encoding="utf-8")

# Cabeçalho da sessão
for k, v in [
    ("===== SESSION START =====", ""),
    ("OS", platform.platform()),
    ("Python", sys.version.replace("\n", " ")),
    ("Arch", platform.machine()),
    ("Config", asdict(cfg)),
    ("HyperParams", asdict(hp)),
]:  logging.info("%s: %s", k, v)

plt.rcParams.update({"font.family": "monospace", "font.size": 8.5})
warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────────────────────
# UTILIDADES DE ARQUIVOS
# ──────────────────────────────────────────────────────────────────────────────
def _collect_csv_files() -> list[Path]:
    local = list(cfg.csv_dir.glob("*.csv"))
    if local:
        return local
    root = list(Path().glob("*.csv"))
    if not root:
        sys.exit("Nenhum CSV encontrado.")
    for csv in root:
        shutil.move(csv, cfg.csv_dir / csv.name)
    sys.exit("CSV movidos para .data; execute novamente.")

def _validate_indices(idxs: Sequence[int], total: int, label: str) -> None:
    if not all(-total <= i < total for i in idxs):
        sys.exit(f"{label} fora do intervalo.")

# ──────────────────────────────────────────────────────────────────────────────
# CARGA E PREPARO DE DADOS
# ──────────────────────────────────────────────────────────────────────────────
def load_data(csv_paths: Optional[Sequence[Path]] = None) -> tuple[pd.DataFrame, np.ndarray]:
    try:
        files = list(csv_paths) if csv_paths else _collect_csv_files()
        frames = [pd.read_csv(f, sep=cfg.csv_sep, dtype=str) for f in files]
        data = pd.concat(frames, ignore_index=True)
    except Exception as exc:
        sys.exit(f"Falha ao carregar CSV: {exc}")

    if cfg.scramble_rows:
        data = data.sample(frac=1, ignore_index=True)

    data = data.apply(pd.to_numeric, errors="coerce")
    total_cols = data.shape[1]
    _validate_indices(cfg.target_cols, total_cols, "Target")
    _validate_indices(cfg.exclude_cols, total_cols, "exclude_cols")

    y = data.iloc[:, list(cfg.target_cols)].values.ravel()
    predictors = data.copy()
    drop = set(cfg.exclude_cols) | set(cfg.target_cols)
    pred_cols = predictors.columns.difference(predictors.columns[list(drop)])

    if cfg.null_value is not None:
        mask = predictors[pred_cols] == cfg.null_value
        predictors.loc[mask.index, pred_cols] = predictors[pred_cols].where(~mask, np.nan)

    X = predictors[pred_cols]
    if X.empty:
        sys.exit("Sem colunas preditoras.")
    return X, y

def make_imputer() -> IterativeImputer | SimpleImputer | str:
    return {
        FillStrategy.ITER   : IterativeImputer(),
        FillStrategy.MEDIAN : SimpleImputer(strategy="median"),
        FillStrategy.MODE   : SimpleImputer(strategy="most_frequent"),
        FillStrategy.KEEP   : SimpleImputer(strategy="constant", fill_value=cfg.null_value),
        FillStrategy.ZERO   : SimpleImputer(strategy="constant", fill_value=0),
    }.get(cfg.fill_strategy, SimpleImputer(strategy="mean"))

def make_classifier() -> MLPClassifier:
    hidden = tuple(u for _, u in hp.layers if u > 1)
    return MLPClassifier(
        hidden_layer_sizes=hidden,
        activation=hp.layers[0][0].value,
        solver="sgd",
        learning_rate_init=hp.learning_rate,
        learning_rate="adaptive",
        warm_start=True,
        batch_size=hp.batch_size,
        max_iter=1,  # iteramos “manual” pelas épocas
    )

# ──────────────────────────────────────────────────────────────────────────────
# MÉTRICAS
# ──────────────────────────────────────────────────────────────────────────────
def _metric_value(name: BestBy, y_true, y_pred, prob=None) -> float:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sum_matrix = tn + fp + fn + tp
    match name:
        case BestBy.F1_SCORE:         return f1_score(y_true, y_pred)
        case BestBy.VAL_LOSS if prob is not None:  return log_loss(y_true, prob)
        case BestBy.RECALL:           return recall_score(y_true, y_pred)
        case BestBy.PRECISION:        return precision_score(y_true, y_pred)
        case BestBy.TRUE_NEGATIVE:    return tn / sum_matrix if sum_matrix else 0.0
        case BestBy.TRUE_POSITIVE:    return tp / sum_matrix if sum_matrix else 0.0
        case BestBy.FALSE_NEGATIVE:   return fn / sum_matrix if sum_matrix else 0.0
        case BestBy.FALSE_POSITIVE:   return fp / sum_matrix if sum_matrix else 0.0
        case BestBy.DIFF_POSITIVE:    return (tp - fp) / sum_matrix if sum_matrix else 0.0
        case BestBy.DIFF_NEGATIVE:    return (tn - fn) / sum_matrix if sum_matrix else 0.0
        case BestBy.ACCURACY:         return (tp + tn) / sum_matrix if sum_matrix else 0.0
        case _:                       return 0.0

# ──────────────────────────────────────────────────────────────────────────────
# PLOTAGEM
# ──────────────────────────────────────────────────────────────────────────────
def _plot_confusion_matrix(cm: np.ndarray, ax: plt.Axes) -> None:
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=9)
    ax.set_xticks([0, 1]), ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["Real 0", "Real 1"])
    ax.set_xlabel(""), ax.set_ylabel("")
    ax.set_title("Matriz de Confusão")

def _architecture_string(n_features: int) -> str:
    layers_desc = "\n\t" + "\n\t".join(f"{u} {a.value}" for a, u in hp.layers)
    info = (
        f"Input:\t\t\t{n_features} features\n"
        f"Architecture:{layers_desc}\n"
        f"Max Epochs:\t\t{hp.epochs}\n"
        f"Batch:\t\t\t{hp.batch_size}\n"
        f"Learning Rate:\t{hp.learning_rate}"
    )
    return info.expandtabs(4)

# ──────────────────────────────────────────────────────────────────────────────
# VISUALIZAÇÃO INTERATIVA
# ──────────────────────────────────────────────────────────────────────────────
class FoldViewer:
    """Interface gráfica para inspeção de folds (modelo completo + k folds)."""

    def __init__(
        self,
        cms:   list[np.ndarray],
        hist:  dict[str, list[float]],
        metric: str,
        direction: str,
        arch: str,
        dataset: str,
        models: list[Pipeline],
        X: pd.DataFrame,
        y: np.ndarray,
        init_idx: int,
        val_indices: list[np.ndarray],   # índices de validação por fold
    ):
        self.cms_current = cms
        self.hist = hist
        self.metric = metric
        self.dir = direction
        self.arch = arch
        self.dataset_name = dataset
        self.models = models      # [0]=modelo completo, 1..k = folds
        self.val_indices = val_indices
        self.X_base, self.y_base = X, y

        self.X, self.y = X, y
        self.n = len(cms)         # inclui Fold 0
        self.idx = init_idx
        self.saved = False

        self._make_figure()
        self._draw()
        self.fig.canvas.mpl_connect("close_event", self._on_close)
        plt.show()

    # ── construção da figura ────────────────────────────────────────────────
    def _make_figure(self):
        gs = plt.GridSpec(1, 2, width_ratios=[1.1, 1])
        self.fig = plt.figure(figsize=(10, 4))
        self.ax_cm  = self.fig.add_subplot(gs[0])
        self.ax_txt = self.fig.add_subplot(gs[1])
        self.ax_txt.set_yticklabels([]), self.ax_txt.set_xticklabels([])
        self.ax_txt.title.set_text("Métricas")
        self.fig.subplots_adjust(bottom=0.23)

        # ── botões de controle principais ───────────────────────────────────
        main_btns = [("Nova Base", self.load_new_csv),
                     ("Restaurar CSV", self.restore_csv),
                     ("Salvar", self.save_current)]
        w, h, pad = 0.18, 0.08, 0.01
        y0 = 0.02
        for i, (label, cb) in enumerate(main_btns):
            ax = plt.axes([0.05 + i * (w + pad), y0, w, h])
            btn = Button(ax, label)
            btn.on_clicked(cb)
            setattr(self, f"btn_{label.replace(' ', '_')}", btn)

        # ── botões para seleção de folds ─────────────────────────────────────
        self.fold_btns: list[Button] = []
        fold_w = max(0.06, 0.8 / self.n)
        y_folds = 0.12
        for i in range(self.n):
            label = "All" if i == 0 else f"{i}"
            ax = plt.axes([0.05 + i * (fold_w + 0.005), y_folds, fold_w, 0.07])
            btn = Button(ax, label)
            btn.on_clicked(self._make_select_cb(i))
            self.fold_btns.append(btn)

    def _make_select_cb(self, idx: int):
        def _cb(_):
            self.idx = idx
            self._draw()
        return _cb

    # ── recomputar métricas ao trocar CSV ────────────────────────────────────
    def _recompute_metrics(self):
        try:
            for i, model in enumerate(self.models):
                if i == 0:
                    X_eval, y_eval = self.X, self.y
                else:
                    idxs = self.val_indices[i - 1]
                    idxs = [j for j in idxs if j < len(self.X)]
                    X_eval, y_eval = (self.X.iloc[idxs], self.y[idxs]) if idxs else (self.X, self.y)

                pred = model.predict(X_eval)
                prob = model.predict_proba(X_eval)
                cm   = confusion_matrix(y_eval, pred)
                self.cms_current[i] = cm

                self.hist["F1_SCORE"][i]   = f1_score(y_eval, pred)
                self.hist["PRECISION"][i]  = precision_score(y_eval, pred)
                self.hist["RECALL"][i]     = recall_score(y_eval, pred)
                self.hist["VAL_LOSS"][i]   = log_loss(y_eval, prob)

                tn, fp, fn, tp = cm.ravel()
                s = tn + fp + fn + tp or 1
                self.hist["TRUE_NEGATIVE"][i] = tn / s
                self.hist["FALSE_POSITIVE"][i] = fp / s
                self.hist["FALSE_NEGATIVE"][i] = fn / s
                self.hist["TRUE_POSITIVE"][i] = tp / s
                self.hist["DIFF_POSITIVE"][i] = (tp - fp) / s
                self.hist["DIFF_NEGATIVE"][i] = (tn - fn) / s
                self.hist["ACCURACY"][i]      = (tp + tn) / s
        except Exception as exc:
            logging.exception("Recompute metrics failed: %s", exc)

    # ── callbacks de botões ─────────────────────────────────────────────────
    def load_new_csv(self, _):
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        file_path = filedialog.askopenfilename(title="Selecionar CSV", filetypes=[("CSV", "*.csv")])
        root.destroy()
        if not file_path: return
        X_new, y_new = load_data([Path(file_path)])
        self.X, self.y = X_new, y_new
        self.dataset_name = Path(file_path).name
        self._recompute_metrics()
        self.idx = 0
        self._draw()

    def restore_csv(self, _):
        self.X, self.y = self.X_base, self.y_base
        self.dataset_name = "Treinamento"
        self._recompute_metrics()
        self.idx = 0
        self._draw()

    def save_current(self, _):
        out = cfg.img_dir / f"{self.dataset_name.replace('.csv', '')}_fold{self.idx}.png"
        try:
            self.fig.savefig(out, dpi=150, bbox_inches="tight")
            print("Figura salva em", out)
            logging.info("Figura %s salva", out.name)
            self.saved = True
        except Exception as exc:
            print(f"Falha ao salvar figura: {exc}")

    # ── desenho ─────────────────────────────────────────────────────────────
    def _draw(self):
        self.ax_cm.clear(), self.ax_txt.clear()
        _plot_confusion_matrix(self.cms_current[self.idx], self.ax_cm)

        # destaque do botão da fold selecionada
        for i, btn in enumerate(self.fold_btns):
            btn.ax.set_facecolor("lightblue" if i == self.idx else "lightgrey")

        f1   = self.hist["F1_SCORE"][self.idx]
        prec = self.hist["PRECISION"][self.idx]
        rec  = self.hist["RECALL"][self.idx]
        vls  = self.hist["VAL_LOSS"][self.idx]
        acc  = self.hist["ACCURACY"][self.idx]

        txt = (
            f"F1-Score:\t\t{f1:.3f}\n"
            f"Precision:\t\t{prec:.3f}\n"
            f"Recall:\t\t\t{rec:.3f}\n"
            f"Accuracy:\t\t{acc:.3f}\n"
            f"Val Loss:\t\t{vls:.4f}\n{self.arch}\n"
        ).expandtabs(4)
        self.ax_txt.text(0.05, 0.95, txt, ha="left", va="top",
                         fontsize=10, transform=self.ax_txt.transAxes)

        self.fig.suptitle(
            f"Fold {self.idx}/{self.n - 1} – {self.dataset_name}  (0 = modelo completo)",
            fontsize=12, weight="bold"
        )
        self.fig.canvas.draw_idle()

    def _on_close(self, _event):
        if not self.saved:
            out = cfg.img_dir / f"{self.dataset_name.replace('.csv', '')}_fold{self.idx}.png"
            try:
                self.fig.savefig(out, dpi=150, bbox_inches="tight")
                logging.info("Auto-save figura %s", out.name)
            except Exception:
                logging.exception("Auto-save falhou")

# ──────────────────────────────────────────────────────────────────────────────
# TREINAMENTO
# ──────────────────────────────────────────────────────────────────────────────
def train_single_fold(
    Xtr: pd.DataFrame, ytr: np.ndarray, Xv: pd.DataFrame, yv: np.ndarray
) -> tuple[Pipeline, dict[str, float], np.ndarray, int]:
    imputer = make_imputer()
    steps = [] if imputer == "drop" else [("imputer", copy.deepcopy(imputer))]
    if cfg.normalize:
        steps.append(("scaler", StandardScaler()))
    clf = make_classifier()
    steps.append(("mlp", clf))
    pipe = Pipeline(steps)
    if imputer == "drop":
        Xtr, ytr = Xtr.dropna(), ytr[Xtr.index]
        Xv,  yv  = Xv.dropna(),  yv[Xv.index]

    best_pipe  = None
    best_score = np.inf if hp.watch_for.value == "min" else -np.inf
    stagnant   = 0
    trained_ep = 0
    classes    = np.unique(ytr)

    for epoch in range(1, hp.epochs + 1):
        trained_ep = epoch
        try:    pipe.named_steps["mlp"].partial_fit(Xtr, ytr, classes)
        except Exception: pipe.fit(Xtr, ytr)

        pred = pipe.predict(Xv)
        prob = pipe.predict_proba(Xv)
        score = _metric_value(hp.watch_for, yv, pred, prob)
        improve = (
            score + hp.min_delta < best_score if hp.watch_for.value == "min"
            else score - hp.min_delta > best_score
        )
        if improve:
            best_score = score
            best_pipe  = copy.deepcopy(pipe)
            stagnant   = 0
        else:
            stagnant += 1
            if stagnant >= hp.patience:
                break

    pipe = best_pipe if best_pipe else pipe
    pred, prob = pipe.predict(Xv), pipe.predict_proba(Xv)
    cm         = confusion_matrix(yv, pred)

    metrics = {
        "F1_SCORE"  : f1_score(yv, pred),
        "PRECISION" : precision_score(yv, pred),
        "RECALL"    : recall_score(yv, pred),
        "VAL_LOSS"  : log_loss(yv, prob),
    }
    tn, fp, fn, tp = cm.ravel(); s = tn + fp + fn + tp or 1
    metrics.update({
        "TRUE_NEGATIVE": tn / s, "FALSE_POSITIVE": fp / s,
        "FALSE_NEGATIVE": fn / s, "TRUE_POSITIVE": tp / s,
        "DIFF_POSITIVE": (tp - fp) / s, "DIFF_NEGATIVE": (tn - fn) / s,
        "ACCURACY": (tp + tn) / s,
    })
    return pipe, metrics, cm, trained_ep

def fold_scores(hist: dict[str, list[float]]) -> tuple[list[float], int]:
    n = len(next(iter(hist.values())))
    scores = [0.0] * n
    for rank, m in enumerate(BestBy):
        if m.name not in hist: continue
        vals = hist[m.name]
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        for i, v in enumerate(vals):
            norm = 0.5 if hi == lo else (v - lo) / (hi - lo)
            if m.value == "min": norm = 1 - norm
            scores[i] += (len(BestBy) - rank) * norm
    best = int(max(range(n), key=scores.__getitem__))
    return scores, best

def train(csv_paths: Optional[Sequence[str]]) -> Path:
    paths = [Path(p) for p in csv_paths] if csv_paths else None
    X, y  = load_data(paths)
    print(f"Conjunto de treino: {len(X)} amostras, {X.shape[1]} features")

    skf = StratifiedKFold(hp.k_folds, shuffle=True)
    hist = {m: [np.nan] * hp.k_folds for m in [m for m in BestBy.__members__.keys()]}
    cms, models, epochs_per_fold, val_indices = [], [], [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        Xtr, ytr = X.iloc[tr_idx], y[tr_idx]
        Xv,  yv  = X.iloc[val_idx], y[val_idx]

        model, metrics, cm, n_epochs = train_single_fold(Xtr, ytr, Xv, yv)
        models.append(model); cms.append(cm); epochs_per_fold.append(n_epochs)
        val_indices.append(val_idx)

        for k, v in metrics.items(): hist[k][fold - 1] = v
        print(
            f"\tFold {fold}:\tEpochs={n_epochs:<4}\t"
            f"F1={metrics['F1_SCORE']:.3f}\tAcc={metrics['ACCURACY']:.3f}\tLoss={metrics['VAL_LOSS']:.4f}"
        )

    # ── modelo completo ------------------------------------------------------
    imputer = make_imputer()
    steps = [] if imputer == "drop" else [("imputer", imputer)]
    if cfg.normalize: steps.append(("scaler", StandardScaler()))
    steps.append(("mlp", MLPClassifier(
        hidden_layer_sizes=tuple(u for _, u in hp.layers if u > 1),
        activation=hp.layers[0][0].value, solver="adam",
        learning_rate_init=hp.learning_rate, max_iter=hp.epochs,
        early_stopping=True)))
    pipe_full = Pipeline(steps); pipe_full.fit(X, y)

    pred_full, prob_full = pipe_full.predict(X), pipe_full.predict_proba(X)
    cm_full = confusion_matrix(y, pred_full)

    def _metric_dict(y_true, y_pred, prob, cm):
        tn, fp, fn, tp = cm.ravel(); s = tn + fp + fn + tp or 1
        return {
            "F1_SCORE": f1_score(y_true, y_pred),
            "PRECISION": precision_score(y_true, y_pred),
            "RECALL": recall_score(y_true, y_pred),
            "VAL_LOSS": log_loss(y_true, prob),
            "TRUE_NEGATIVE": tn / s, "FALSE_POSITIVE": fp / s,
            "FALSE_NEGATIVE": fn / s, "TRUE_POSITIVE": tp / s,
            "DIFF_POSITIVE": (tp - fp) / s, "DIFF_NEGATIVE": (tn - fn) / s,
            "ACCURACY": (tp + tn) / s,
        }
    metrics_full = _metric_dict(y, pred_full, prob_full, cm_full)

    # prefixa Fold 0
    for k in hist.keys(): hist[k] = [metrics_full[k]] + hist[k]
    cms.insert(0, cm_full); models.insert(0, pipe_full)

    mean_f1 = np.nanmean(hist["F1_SCORE"][1:])
    print(f"F1 médio (k-folds): {mean_f1:.4f}")

    scores, best_idx = fold_scores(hist)
    logging.info("Fold scores: %s; best=%d", scores, best_idx)

    arch = _architecture_string(X.shape[1])
    FoldViewer(cms, hist, hp.watch_for.name, hp.watch_for.value, arch,
               "Treinamento", models, X, y, best_idx, val_indices)

    # ── salvamento do modelo completo ---------------------------------------
    model_path = cfg.mdl_dir / f"mlp_{strftime('%Y%m%d_%H%M%S')}.joblib"
    joblib.dump(pipe_full, model_path)
    print("Modelo salvo em", model_path)
    return model_path

# ──────────────────────────────────────────────────────────────────────────────
# VALIDAÇÃO DE MODELOS
# ──────────────────────────────────────────────────────────────────────────────
def latest_model() -> Path:
    models = sorted(cfg.mdl_dir.glob("mlp_*.joblib"))
    if not models: sys.exit("Nenhum modelo em .models.")
    return models[-1]

def validate(model_path: Path, csv_paths: Optional[Sequence[str]]) -> None:
    paths = [Path(p) for p in csv_paths] if csv_paths else None
    X, y  = load_data(paths)
    pipe: Pipeline = joblib.load(model_path)

    pred = pipe.predict(X)
    print(classification_report(y, pred, digits=3))

    cm = confusion_matrix(y, pred)
    plt.figure(figsize=(4, 4)); _plot_confusion_matrix(cm, plt.gca())
    plt.tight_layout()
    out = cfg.img_dir / f"{model_path.stem}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show(block=True); plt.close()
    print("Gráfico salvo em", out)
    logging.info("Figura %s salva", out.name)

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> Namespace:
    p = ArgumentParser(description="MLP trainer / validator.")
    p.add_argument("--validate", nargs="?", const="", metavar="MODEL",
                   help="Valida o modelo informado (ou o mais recente se omitido).")
    p.add_argument("--csv", nargs="+", metavar="CSV",
                   help="Arquivos CSV específicos.")
    return p.parse_args()

def main() -> None:
    start = perf_counter()
    args = parse_args()
    if args.validate is not None:
        mdl = Path(args.validate) if args.validate else latest_model()
        validate(mdl, args.csv)
    else:
        train(args.csv)
    logging.info("Tempo total %.1fs", perf_counter() - start)
    logging.info("===== SESSION END =====")

if __name__ == "__main__":
    main()