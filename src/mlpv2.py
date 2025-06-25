import sys
import copy
import logging
import platform
import shutil
import warnings
from enum import Enum
from pathlib import Path
from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass
from time import perf_counter, strftime
from typing import Optional, Sequence
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Enums para controle de métricas e estratégias de preenchimento
class BestBy(str, Enum):
    VAL_LOSS = "min"
    F1_SCORE = "max"
    ACCURACY = "max"

class FillStrategy(Enum):
    MEAN = 1
    MEDIAN = 2
    MODE = 3
    ITER = 4
    ZERO = 6

@dataclass(frozen=True, slots=True)
class HyperParams:
    k_folds: int = 5
    epochs: int = 2048
    patience: int = 64
    min_delta: float = 1e-4
    watch_for: BestBy = BestBy.F1_SCORE
    batch_size: int = 64
    learning_rate: float = 1e-3
    layers: tuple[int, ...] = (7, 7, 1)

@dataclass(frozen=True, slots=True)
class Config:
    csv_sep: str = ";"
    target_cols: tuple[int, ...] = (-1,)
    exclude_cols: tuple[int, ...] = (0,)
    csv_dir: Path = Path(".data")
    mdl_dir: Path = Path(".models")
    log_dir: Path = Path(".log")
    img_dir: Path = Path(".img")

    def setup_dirs(self):
        for d in [self.csv_dir, self.mdl_dir, self.log_dir, self.img_dir]:
            d.mkdir(exist_ok=True)

# Inicialização de configurações e parâmetros
cfg, hp = Config(), HyperParams()
cfg.setup_dirs()

logging.basicConfig(
    filename=cfg.log_dir / f"{strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    encoding="utf-8",
)

warnings.filterwarnings("ignore", category=UserWarning)

# Funções Utilitárias
def load_data(files: Optional[Sequence[Path]] = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Carrega dados do CSV e prepara matriz X (preditores) e vetor y (targets)."""
    files = files or list(cfg.csv_dir.glob("*.csv"))
    if not files:
        sys.exit("Nenhum arquivo CSV encontrado.")
    data = pd.concat([pd.read_csv(f, sep=cfg.csv_sep) for f in files], ignore_index=True)
    data = data.apply(pd.to_numeric, errors='coerce').dropna()
    target = data.iloc[:, list(cfg.target_cols)].values.ravel()
    predictors = data.drop(columns=data.columns[list(cfg.exclude_cols + cfg.target_cols)])
    return predictors, target

def make_pipeline() -> Pipeline:
    """Configura o pipeline do modelo, incluindo normalização e imputação."""
    steps = [("scaler", StandardScaler())]
    clf = MLPClassifier(
        hidden_layer_sizes=tuple(hp.layers),
        activation='relu',
        learning_rate_init=hp.learning_rate,
        max_iter=hp.epochs,
        early_stopping=True,
        batch_size=hp.batch_size,
    )
    steps.append(("mlp", clf))
    return Pipeline(steps)

# Treinamento e Validação
def train(csv_paths: Optional[Sequence[str]]) -> Path:
    """Treina o modelo com k-fold e salva o modelo otimizado."""
    X, y = load_data([Path(p) for p in csv_paths] if csv_paths else None)
    skf = StratifiedKFold(n_splits=hp.k_folds, shuffle=True)
    fold_scores = []
    models = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        pipeline = make_pipeline()
        pipeline.fit(X_train, y_train)
        models.append(pipeline)

        y_pred = pipeline.predict(X_val)
        f1 = f1_score(y_val, y_pred)
        fold_scores.append(f1)
        logging.info(f"Fold {fold_idx}: F1-Score={f1:.4f}")

    best_pipeline = models[np.argmax(fold_scores)]
    model_path = cfg.mdl_dir / f"mlp_{strftime('%Y%m%d_%H%M%S')}.joblib"
    joblib.dump(best_pipeline, model_path)
    logging.info(f"Modelo salvo em {model_path}")
    return model_path

def validate(model_path: Path, csv_paths: Optional[Sequence[str]] = None) -> None:
    """Valida o modelo salvo com um conjunto de teste."""
    X, y = load_data([Path(p) for p in csv_paths] if csv_paths else None)
    pipeline = joblib.load(model_path)
    y_pred = pipeline.predict(X)
    print(classification_report(y, y_pred))
    cm = confusion_matrix(y, y_pred)
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

# CLI Principal
def main() -> None:
    args = ArgumentParser(description="Treinador e validador de MLP").parse_args()
    model_path = train(args.csv) if args.validate is None else validate(Path(args.validate), args.csv)

if __name__ == "__main__":
    main()