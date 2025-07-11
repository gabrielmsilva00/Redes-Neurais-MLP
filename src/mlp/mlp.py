from __future__ import annotations
import sys, copy, logging, platform, shutil, warnings, traceback, os, math

from numbers import Number
from enum import Enum
from pathlib import Path
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, asdict
from time import perf_counter, strftime
from typing import Optional, Sequence, Any

import joblib as jb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as mgs
import numpy as np
import pandas as pd
from matplotlib.widgets import Button
from pureset import PureSet
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

class BestBy(Enum):
    VAL_LOSS = (0, "min")
    FALSE_NEGATIVE = (1, "min")
    FALSE_POSITIVE = (2, "min")
    TRUE_POSITIVE = (3, "max")
    TRUE_NEGATIVE = (4, "max")
    DIFF_POSITIVE = (5, "max")
    DIFF_NEGATIVE = (6, "max")
    F1_SCORE = (7, "max")
    ACCURACY = (8, "max")
    PRECISION = (9, "max")
    RECALL = (10, "max")


class FillStrategy(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "most_frequent"
    KEEP = "constant"
    ZERO = "zero"


class Activation(str, Enum):
    RELU = "relu"
    TANH = "tanh"
    LINEAR = "identity"
    SIGMOID = "logistic"


@dataclass(frozen=True, slots=True)
class HyperParams:
    test_split: float = 0.5
    k_folds: int = 4
    epochs: int = 1024
    patience: int = 64
    watch_for: tuple[BestBy, ...] = (
        BestBy.F1_SCORE,
        BestBy.FALSE_NEGATIVE,
        BestBy.PRECISION,
        BestBy.VAL_LOSS,
    )
    learning_rate: Number = 8e-4
    min_delta: Number = 4e-8
    activation: Activation = Activation.TANH
    layers: tuple[int, ...] = (8, 4, 2)


@dataclass(frozen=True, slots=True)
class Config:
    csv_dir: Path = Path(".data")
    log_dir: Path = Path(".log")
    img_dir: Path = Path(".img")
    mdl_dir: Path = Path(".models")
    cfg_dir: Path = Path(".config")
    csv_sep: str = ";"
    header: Optional[int] = None
    scramble_rows: bool = True
    normalize: bool = True
    null_value: Optional[Number] = 0
    exclude_cols: Optional[tuple[int, ...]] = (0,)
    fill_strategy: FillStrategy = FillStrategy.KEEP


cfg, hp = Config(), HyperParams()

plt.rcParams.update({"font.family": "monospace", "font.size": 12})
warnings.filterwarnings("ignore", category=UserWarning)
for d in (cfg.csv_dir, cfg.img_dir, cfg.mdl_dir, cfg.log_dir, cfg.cfg_dir):
    d.mkdir(exist_ok=True)

logging.basicConfig(
    filename=cfg.log_dir / f"{strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    encoding="utf-8",
)

for k, v in [
    ("===== SESSION START =====", ""),
    ("OS", platform.platform()),
    ("Python", sys.version.replace("\n", " ")),
    ("Arch", platform.machine()),
    ("Config", asdict(cfg)),
    ("HyperParams", asdict(hp)),
]:
    logging.info("%s: %s", k, v)


def load_data(csv_paths: Optional[Sequence[Path]] = None) -> tuple[pd.DataFrame, np.ndarray]:
    try:
        if csv_paths:
            files = list(csv_paths)
        elif (csv_files := sorted(list(cfg.csv_dir.glob("*.csv")))):
            files = csv_files
        else:
            root = list(Path().glob("*.csv"))
            if not root:
                sys.exit("Nenhum CSV encontrado.")
            for csv in root:
                shutil.move(csv, cfg.csv_dir / csv.name)
            sys.exit("CSV movidos para .data; execute novamente.")
        frames = [pd.read_csv(f, sep=cfg.csv_sep, dtype=str, header=cfg.header) for f in files]
        data = pd.concat(frames, ignore_index=True)
    except SystemExit:
        raise
    except Exception:
        sys.exit(f"Falha ao carregar CSV:\n{traceback.format_exc()}")

    data = data.apply(pd.to_numeric, errors="coerce")
    total_cols = data.shape[1]
    if not all(-total_cols <= i < total_cols for i in cfg.exclude_cols):
        sys.exit("Coluna de Exclusão fora do intervalo.")
    y = data.iloc[:, [-1]].values.ravel()
    predictors = data.copy()
    pred_cols = predictors.columns
    if cfg.null_value is not None:
        mask = predictors[pred_cols] == cfg.null_value
        if cfg.exclude_cols:
            mask.loc[:, cfg.exclude_cols] = False
        predictors.loc[mask.index, pred_cols] = predictors[pred_cols].where(~mask, np.nan)
    X = predictors[pred_cols].drop(columns=len(pred_cols) - 1)
    if X.empty:
        sys.exit("Sem colunas preditoras.")
    return X, y

def _metric_value(name: BestBy, y_true, y_pred, prob=None) -> float:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    s = tn + fp + fn + tp
    match name:
        case BestBy.F1_SCORE:
            return f1_score(y_true, y_pred)
        case BestBy.VAL_LOSS:
            return log_loss(y_true, prob)
        case BestBy.RECALL:
            return recall_score(y_true, y_pred)
        case BestBy.PRECISION:
            return precision_score(y_true, y_pred, zero_division=0)
        case BestBy.TRUE_NEGATIVE:
            return tn / s if s else 0
        case BestBy.TRUE_POSITIVE:
            return tp / s if s else 0
        case BestBy.FALSE_NEGATIVE:
            return fn / s if s else 0
        case BestBy.FALSE_POSITIVE:
            return fp / s if s else 0
        case BestBy.DIFF_POSITIVE:
            return (tp - fp) / s if s else 0
        case BestBy.DIFF_NEGATIVE:
            return (tn - fn) / s if s else 0
        case BestBy.ACCURACY:
            return (tp + tn) / s if s else 0
        case _:
            return 0.0


def _metric_dict(y_true, y_pred, prob, cm):
    tn, fp, fn, tp = cm.ravel()
    s = tn + fp + fn + tp or 1
    return {
        "F1_SCORE": f1_score(y_true, y_pred),
        "PRECISION": precision_score(y_true, y_pred, zero_division=0),
        "RECALL": recall_score(y_true, y_pred),
        "VAL_LOSS": log_loss(y_true, prob),
        "TRUE_NEGATIVE": tn / s,
        "FALSE_POSITIVE": fp / s,
        "FALSE_NEGATIVE": fn / s,
        "TRUE_POSITIVE": tp / s,
        "DIFF_POSITIVE": (tp - fp) / s,
        "DIFF_NEGATIVE": (tn - fn) / s,
        "ACCURACY": (tp + tn) / s,
    }


def _score_vector(y_true, y_pred, prob) -> list[float]:
    return [_metric_value(m, y_true, y_pred, prob) for m in hp.watch_for]


def _is_better(a: list[float], b: list[float], tol: float) -> bool:
    for m, av, bv in zip(hp.watch_for, a, b):
        if m.value[1] == "min":
            if av < bv - tol:
                return True
            if av > bv + tol:
                return False
        else:
            if av > bv + tol:
                return True
            if av < bv - tol:
                return False
    return False


class FoldViewer:
    def __init__(
        self,
        cms: list[np.ndarray],
        hist: dict[str, list[float]],
        metric: str,
        direction: str,
        arch: str,
        dataset: str,
        models: list[Pipeline],
        X: pd.DataFrame,
        y: np.ndarray,
        val_indices: list[np.ndarray],
        epochs: list[Any],
    ):
        self.cms_current = cms
        self.hist = hist
        self.metric = metric
        self.dir = direction
        self.arch = arch
        self.dataset_name = dataset
        self.models = models
        self.val_indices = val_indices
        self.X_base, self.y_base = X, y
        self.X, self.y = X, y
        self.epochs = epochs
        self.n = len(cms)
        self.idx = 0
        self.saved = False
        self.scroll = 0
        self.max_visible = 10
        self._make_figure()
        self._draw()
        self.fig.canvas.mpl_connect("close_event", self._on_close)
        plt.show()

    def _make_figure(self):
        self.fig = plt.figure(figsize=(12, 8), dpi=100)
        gs_main = self.fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.05)
        gs_left = mgs.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_main[0, 0], hspace=0.25, wspace=0.25)
        self.ax_menu = self.fig.add_subplot(gs_left[0, 0])
        self._blank(self.ax_menu)
        self.ax_cm = self.fig.add_subplot(gs_left[0, 1])
        self.ax_arch = self.fig.add_subplot(gs_left[1, 0])
        self._blank(self.ax_arch)
        self.ax_txt = self.fig.add_subplot(gs_left[1, 1])
        self._blank(self.ax_txt)
        self.ax_flds = self.fig.add_subplot(gs_main[0, 1])
        self._blank(self.ax_flds)
        self.fig.subplots_adjust(top=0.9, bottom=0.06)
        self.menu_btns = []
        menu_items = [("Carregar CSV", self.load_new_csv), ("Restaurar CSV", self.restore_csv), ("Salvar Gráfico", self.save_current)]
        total = len(menu_items)
        gap = 0.05
        btn_h = 0.20
        start_y = 0.5 + ((total - 1) / 2) * (btn_h + gap)
        for i, (lbl, cb) in enumerate(menu_items):
            y = start_y - i * (btn_h + gap)
            ax = self.fig.add_axes(self._rel(self.ax_menu, 0.2, y, 0.8, btn_h))
            btn = Button(ax, lbl)
            btn.on_clicked(cb)
            self.menu_btns.append(btn)
        self.ax_arch.text(0.20, 1.40, self.arch, va="top", ha="left", wrap=True)
        self._create_fold_buttons()

    def _blank(self, ax):
        ax.axis("off")
        [s.set_visible(False) for s in ax.spines.values()]

    def _rel(self, ax, x, y, w, h):
        bb = ax.get_position()
        return [bb.x0 + x * bb.width, bb.y0 + y * bb.height, w * bb.width, h * bb.height]

    def _create_fold_buttons(self):
        for btn in getattr(self, "fold_btn_objs", []):
            btn.ax.remove()
        self.fold_btn_objs = []
        if hasattr(self, "_scroll_patch"):
            self._scroll_patch.remove()
        if hasattr(self, "btn_up"):
            self.btn_up.ax.remove()
        if hasattr(self, "btn_down"):
            self.btn_down.ax.remove()
        btn_h = 0.06
        gap = 0.015
        up_ax = self.fig.add_axes(self._rel(self.ax_flds, 0.2, 0.93, 0.6, btn_h))
        dn_ax = self.fig.add_axes(self._rel(self.ax_flds, 0.2, 0.01, 0.6, btn_h))
        self.btn_up = Button(up_ax, "↑")
        self.btn_up.on_clicked(self._scroll_up)
        self.btn_down = Button(dn_ax, "↓")
        self.btn_down.on_clicked(self._scroll_down)
        vis = list(range(self.scroll, min(self.scroll + self.max_visible, self.n)))
        area_h = 0.92 - 2 * (btn_h + gap)
        slot_h = area_h / self.max_visible
        for row, i in enumerate(vis):
            y = 0.96 - (btn_h + gap) - (row + 1) * slot_h + slot_h * 0.2
            label = "Total" if i == 0 else f"Fold {i}"
            ax = self.fig.add_axes(self._rel(self.ax_flds, 0.05, y, 0.9, slot_h * 0.7))
            btn = Button(ax, label)
            btn.on_clicked(self._make_select_cb(i))
            self.fold_btn_objs.append(btn)
        self._draw_scrollbar()
        self._highlight_selected()

    def _draw_scrollbar(self):
        track_x0, track_y0 = 0.98, 0.02
        track_h = 0.9
        self._scroll_patch = self.ax_flds.add_patch(
            mpatches.Rectangle((track_x0, track_y0), 0.04, track_h, transform=self.ax_flds.transAxes, facecolor="#f0f0f0", edgecolor="k", linewidth=0.4)
        )
        handle_h = max(track_h * (self.max_visible / self.n), 0.05)
        denom = max(self.n - self.max_visible, 1)
        handle_y = track_y0 + (track_h - handle_h) * (self.scroll / denom)
        self.ax_flds.add_patch(
            mpatches.Rectangle((track_x0, handle_y), 0.04, handle_h, transform=self.ax_flds.transAxes, facecolor="#777", edgecolor="k", linewidth=0.4)
        )

    def _scroll_up(self, _):
        self._scroll_generic(-3)

    def _scroll_down(self, _):
        self._scroll_generic(3)

    def _scroll_generic(self, delta: int):
        self.scroll = max(0, min(self.scroll + delta, self.n - self.max_visible))
        self._create_fold_buttons()

    def _make_select_cb(self, idx: int):
        def _cb(_):
            self.idx = idx
            self._draw()
            self._highlight_selected()
        return _cb

    def _highlight_selected(self):
        for btn in self.fold_btn_objs:
            lbl = btn.label.get_text()
            i = 0 if lbl == "Total" else int(lbl.split()[-1])
            btn.ax.set_facecolor("lightblue" if i == self.idx else "lightgrey")
        self.fig.canvas.draw_idle()

    def _recompute_metrics(self):
        try:
            for i, model in enumerate(self.models):
                if i == 0:
                    X_eval, y_eval = self.X, self.y
                else:
                    idxs = [j for j in self.val_indices[i - 1] if j < len(self.X)]
                    X_eval, y_eval = (self.X.iloc[idxs], self.y[idxs]) if idxs else (self.X, self.y)
                pred = model.predict(X_eval)
                prob = model.predict_proba(X_eval)
                cm = confusion_matrix(y_eval, pred)
                self.cms_current[i] = cm
                self.hist["F1_SCORE"][i] = f1_score(y_eval, pred)
                self.hist["PRECISION"][i] = precision_score(y_eval, pred, zero_division=0)
                self.hist["RECALL"][i] = recall_score(y_eval, pred)
                self.hist["VAL_LOSS"][i] = log_loss(y_eval, prob)
                tn, fp, fn, tp = cm.ravel()
                s = tn + fp + fn + tp or 1
                self.hist["TRUE_NEGATIVE"][i] = tn / s
                self.hist["FALSE_POSITIVE"][i] = fp / s
                self.hist["FALSE_NEGATIVE"][i] = fn / s
                self.hist["TRUE_POSITIVE"][i] = tp / s
                self.hist["DIFF_POSITIVE"][i] = (tp - fp) / s
                self.hist["DIFF_NEGATIVE"][i] = (tn - fn) / s
                self.hist["ACCURACY"][i] = (tp + tn) / s
        except Exception as exc:
            logging.exception("Recompute metrics failed: %s", exc)

    def load_new_csv(self, _):
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="Selecionar CSV", filetypes=[("CSV", "*.csv")])
        root.destroy()
        logging.info("CSV selecionado: %s", file_path)
        if not file_path:
            return
        self.X, self.y = load_data([Path(file_path)])
        self.dataset_name = Path(file_path).name
        self._recompute_metrics()
        self.idx = 0
        self.scroll = 0
        self._create_fold_buttons()
        self._draw()

    def restore_csv(self, _):
        self.X, self.y = self.X_base, self.y_base
        self.dataset_name = "Treinamento"
        self._recompute_metrics()
        self.idx = 0
        self.scroll = 0
        self._create_fold_buttons()
        self._draw()

    def save_current(self, _):
        out = cfg.img_dir / f"{self.dataset_name.replace('.csv','')}_{'Modelo_Completo' if self.idx==0 else f'Fold_{self.idx}'}_{strftime('%Y%m%d_%H%M%S')}.png"
        try:
            self.fig.savefig(out, dpi=100, bbox_inches="tight")
            print("Figura salva em", out)
            logging.info("Figura %s salva", out.name)
            self.saved = True
        except Exception:
            print(f"Falha ao salvar figura: \n{traceback.format_exc()}")

    def _draw(self):
        self.ax_cm.clear()
        self.ax_txt.clear()
        self.ax_txt.axis("off")
        _plot_confusion_matrix(self.cms_current[self.idx], self.ax_cm)
        f1 = self.hist["F1_SCORE"][self.idx]
        prec = self.hist["PRECISION"][self.idx]
        rec = self.hist["RECALL"][self.idx]
        vls = self.hist["VAL_LOSS"][self.idx]
        acc = self.hist["ACCURACY"][self.idx]
        ep = self.epochs[self.idx]
        txt = (
            f"Épocas:\t\t{ep}\n\nF1-Score:\t{f1:.3f}\n\nPrecision:\t{prec:.3f}\n\n"
            f"Recall:\t\t{rec:.3f}\n\nAccuracy:\t{acc:.3f}\n\nVal Loss:\t{vls:.3f}"
        ).expandtabs(4)
        self.ax_txt.text(0.2, 0.95, txt, va="top", ha="left", wrap=True)
        self.fig.suptitle(
            f"{'Modelo Completo' if self.idx==0 else f'Fold {self.idx}'} | Validando: {self.dataset_name} | Entradas: {len(self.y)}",
            fontsize=14,
            weight="bold",
            ha="center",
            y=0.98,
        )
        self.fig.canvas.draw_idle()

    def _on_close(self, _event):
        if not self.saved:
            out = cfg.img_dir / f"{self.dataset_name.replace('.csv','')}_{'Modelo_Completo' if self.idx==0 else f'Fold_{self.idx}'}_{strftime('%Y%m%d_%H%M%S')}.png"
            try:
                self.fig.savefig(out, dpi=100, bbox_inches="tight")
                logging.info("Auto-save figura %s", out.name)
            except Exception:
                logging.exception("Auto-save falhou")


def _train_single_fold(fold_id: int, tr_idx: np.ndarray, val_idx: np.ndarray, X: pd.DataFrame, y: np.ndarray, classes: np.ndarray) -> dict[str, Any]:
    imputer = SimpleImputer(
        strategy=cfg.fill_strategy.value,
        fill_value=cfg.null_value if cfg.fill_strategy is FillStrategy.KEEP else None,
        add_indicator=True,
        keep_empty_features=True,
    )
    steps = [] if imputer == "drop" else [("imputer", copy.deepcopy(imputer))]
    if cfg.normalize:
        steps.append(("scaler", StandardScaler()))
    clf = MLPClassifier(
        max_iter=1,
        shuffle=True,
        solver="adam",
        learning_rate_init=hp.learning_rate,
        learning_rate="adaptive",
        activation=hp.activation.value,
        hidden_layer_sizes=hp.layers,
        warm_start=True,
        early_stopping=False,
    )
    steps.append(("mlp", clf))
    pipe = Pipeline(steps)
    best_scores = None
    best_pipe = None
    stagnant = 0
    tol = hp.learning_rate * hp.min_delta
    for epoch in range(1, hp.epochs + 1):
        try:
            pipe.named_steps["mlp"].partial_fit(X.iloc[tr_idx], y[tr_idx], classes)
        except Exception:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(X.iloc[tr_idx], y[tr_idx])
        pred_full = pipe.predict(X.iloc[tr_idx])
        prob_full = pipe.predict_proba(X.iloc[tr_idx])
        scores = _score_vector(y[tr_idx], pred_full, prob_full)
        if best_scores is None or _is_better(scores, best_scores, tol):
            best_scores = scores
            best_pipe = copy.deepcopy(pipe)
            stagnant = 0
        else:
            stagnant += 1
        if stagnant >= hp.patience:
            break
    if best_pipe is None:
        best_pipe = pipe
    watch_list = list(zip(hp.watch_for, best_scores))
    print(f"Fold {fold_id}: treino concluído em {epoch} épocas:")
    for w in watch_list:
        print(f"\t{w[0]}: {w[1]:.3f}")
    print()
    return {"pipe": best_pipe, "best_metric": best_scores, "epochs": epoch, "val_idx": val_idx}


def train(csv_paths: Optional[Sequence[str]]) -> Path:
    paths = [Path(p) for p in csv_paths] if csv_paths else None
    X, y = load_data(paths)
    dataset_names = [p.name for p in paths] if paths else [f.name for f in cfg.csv_dir.glob("*.csv")]
    print("Dataset(s):", ", ".join(dataset_names))
    print("Hiperparâmetros em uso:")
    for k, v in asdict(hp).items():
        if k == "watch_for":
            print(f"- {k}: {get_bestby_name()}")
            continue
        if isinstance(v, (tuple, list)):
            v = "\n\t- " + "\n\t- ".join(map(str, v))
        print(f"- {k}:\t{v}")
    print("\nPrimeiras 10 amostras:\n", X.head(10), "\n")
    print("Total de entradas:", len(X))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=hp.test_split, stratify=y)
    print(f"Divisão de Teste: {len(X_train)} ({hp.test_split*100:.2f}%)")
    print(f"Note: As entradas de teste NÃO influenciam no treino!\n")
    classes = np.unique(y_train)
    skf = StratifiedKFold(hp.k_folds, shuffle=True)
    fold_splits = list(skf.split(X_train, y_train))
    n_jobs = min(hp.k_folds, os.cpu_count() or 1)
    results = jb.Parallel(n_jobs=n_jobs)(
        jb.delayed(_train_single_fold)(i, tr, vl, X_train, y_train, classes) for i, (tr, vl) in enumerate(fold_splits, start=1)
    )
    best_global_scores = None
    best_pipe = None
    best_fold_idx = -1
    tol = hp.learning_rate * hp.min_delta
    for i, res in enumerate(results, start=1):
        pipe = res["pipe"]
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)
        scores = _score_vector(y_test, pred, prob)
        if best_global_scores is None or _is_better(scores, best_global_scores, tol):
            best_global_scores = scores
            best_pipe = pipe
            best_fold_idx = i
    pred_full = best_pipe.predict(X)
    prob_full = best_pipe.predict_proba(X)
    cm_full = confusion_matrix(y, pred_full)
    hist = {m: [np.nan] * (hp.k_folds + 1) for m in BestBy.__members__.keys()}
    metrics_full = _metric_dict(y, pred_full, prob_full, cm_full)
    for k in hist.keys():
        hist[k][0] = metrics_full[k]
    cms = [cm_full]
    epochs_list = ["/"]
    if best_fold_idx > 0:
        epochs_list[0] = f"{results[best_fold_idx - 1]['epochs']} (Fold {best_fold_idx})"
    for i, res in enumerate(results, start=1):
        pipe = res["pipe"]
        val_idx = res["val_idx"]
        Xv, yv = X_train.iloc[val_idx], y_train[val_idx]
        pred = pipe.predict(Xv)
        prob = pipe.predict_proba(Xv)
        cm = confusion_matrix(yv, pred)
        cms.append(cm)
        m = _metric_dict(yv, pred, prob, cm)
        for k in hist.keys():
            hist[k][i] = m[k]
        epochs_list.append(res["epochs"])
        print(f"\tFold {i} | Épocas={res['epochs']} | F1={m['F1_SCORE']:.3f}  Acc={m['ACCURACY']:.3f}  Loss={m['VAL_LOSS']:.3f}")
    arch = _architecture_string(X.shape[1])
    FoldViewer(
        cms,
        hist,
        hp.watch_for[0].name,
        hp.watch_for[0].value[1],
        arch,
        "Validação",
        [best_pipe] + [r["pipe"] for r in results],
        X,
        y,
        [r["val_idx"] for r in results],
        epochs_list,
    )
    model_path = cfg.mdl_dir / f"mlp_{strftime('%Y%m%d_%H%M%S')}.joblib"
    jb.dump(best_pipe, model_path)
    print("Modelo salvo em", model_path)
    return model_path


def latest_model() -> Path:
    models = sorted(cfg.mdl_dir.glob("mlp_*.joblib"))
    if not models:
        sys.exit("Nenhum modelo em .models.")
    return models[-1]


def validate(model_path: Path, csv_paths: Optional[Sequence[str]]) -> None:
    paths = [Path(p) for p in csv_paths] if csv_paths else None
    X, y = load_data(paths)
    pipe: Pipeline = jb.load(model_path)
    pred = pipe.predict(X)
    print(classification_report(y, pred, digits=3))
    cm = confusion_matrix(y, pred)
    plt.figure(figsize=(4, 4))
    _plot_confusion_matrix(cm, plt.gca())
    plt.tight_layout()
    out = cfg.img_dir / f"{model_path.stem}.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.show(block=True)
    plt.close()
    print("Gráfico salvo em", out)
    logging.info("Figura %s salva", out.name)


def _plot_confusion_matrix(cm: np.ndarray, ax: plt.Axes) -> None:
    ax.imshow(cm, cmap="Greens", vmin=0, vmax=cm.max(), interpolation="nearest")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=24)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["Real 0", "Real 1"])
    ax.set_xlabel("")
    ax.set_ylabel("")


def get_dataset_name() -> str:
    if not cfg.csv_dir.is_dir():
        return "Treinamento"
    csv_files = list(cfg.csv_dir.glob("*.csv"))
    if not csv_files:
        return "Treinamento"
    names = [f.stem for f in csv_files if f.is_file()]
    return "\n\t" + "\n\t".join(names) if names else "Treinamento"


def get_bestby_name() -> str:
    return "\n\t" + "\n\t".join((m.name + f" ({m.value[1]})") for m in hp.watch_for)


def _architecture_string(n_features: int) -> str:
    return (
        f"Dataset:{get_dataset_name()}\n"
        f"Features:\t\t{n_features}\n"
        f"Null Value:\t\t{cfg.null_value}\n"
        f"Imputer:\t\t{cfg.fill_strategy.name}\n\n"
        f"Max Epochs:\t\t{hp.epochs}\n"
        f"Train/Test:\t\t{int(hp.test_split*100)}%\n"
        f"Best By:{get_bestby_name()}\n"
        f"Patience:\t\t{hp.patience}\n"
        f"Learning Rate:\t{hp.learning_rate}\n"
        f"Min. Delta:\t\t{hp.min_delta}\n\n"
        f"Layers ({hp.activation.value}):" + ("\n\t" + " → ".join(str(u) for u in hp.layers))
    ).expandtabs(4)


def parse_args() -> Namespace:
    p = ArgumentParser(description="MLP trainer / validator.")
    p.add_argument("--validate", nargs="?", const="", metavar="MODEL", help="Valida o modelo informado (ou o mais recente se omitido).")
    p.add_argument("--csv", nargs="+", metavar="CSV", help="Arquivos CSV específicos.")
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