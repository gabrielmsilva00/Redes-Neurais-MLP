"""
Arquivo: som.py

Implementação de um Mapa Auto-Organizável (Self-Organizing Map, SOM) voltado ao processamento e análise do conjunto
de dados Iris. O objetivo central deste programa é demonstrar a aplicação do SOM para tarefas de visualização,
prototipagem e classificação não-supervisionada e supervisionada de dados multivariados, com ênfase em ciência
de dados reprodutível, modular e adequada ao ambiente científico/engenharia de software.

Funcionalidades-chave:
- Carregamento automático ou via argumento de banco de dados (Iris ou arquivo customizado ZIP/CSV).
- Treinamento supervisionado e não-supervisionado do SOM com configuração via dataclass imutável.
- Avaliação de performance e métricas completas antes da visualização gráfica.
- Salvamento automático do modelo treinado e de gráficos ao disco com versionamento e timestamp.
- Validação/utilização de modelos salvos via flag de linha de comando.
- Plots não bloqueantes (salvos em .png/) e possibilidade de supressão de GUI interativa (-ng).
- Registro integral de logs, métricas e ambiente do experimento em .log/, facilitando rastreabilidade científica.
- Estrutura modular e extensível, adequada a pipelines científicos, automação e integração com workflows maiores.

===============================================================================================
Informações:
-----------------------------------------------------------------------------------------------
Autores: Gabriel Maia (@gabrielmsilva00)
Graduando em Engenharia Elétrica pela Universidade Estadual do Rio de Janeiro (UERJ), Brasil, 2025.

Como executar:
-------------
Dependências externas: NumPy, Pandas, MatPlotLib, SciKit-Learn, MiniSom, ucimlrepo
Treinamento padrão (com o Iris embutido):
    python som.py

Para utilizar um dataset ou modelo diferente:
    python som.py -data=meuarquivo.zip -m=15 -key=42
    python som.py -load=./.som/som-model-2024-07-09_23-41-00-0.987.som -ng

Parâmetros configuráveis por flags:
-----------------------------------
-m=           Número de linhas do grid SOM (default: 10)
-n=           Número de colunas do grid SOM (default: 10)
-lr=          Taxa de aprendizado (learning rate)
-iters=       Iterações máximas de treinamento
-radius=      Raio inicial da vizinhança SOM (default: max(m,n)/2)
-key=         Semente aleatória para reprodutibilidade
-ng           Desabilita a exibição gráfica (plot é salvo em .png/ apenas)
-load=        Caminho do modelo SOM pickle a ser avaliado
-data=        Caminho dataset alternativo (csv/zip)

Entradas:
---------
- Banco de dados Iris por padrão (fetch_ucirepo), ou arquivo informado via flag -data
- Possível entrada de modelo pré-treinado pickle (.som/model-...som) para validação apenas (flag -load)

Saídas:
-------
- Modelo SOM treinado (.som/som-model-AAAA-mm-dd_HH-MM-SS-n.nnn.som)
- Logs detalhados (.log/iris_som_AAAA-mm-dd_HH-MM-SS.log)
- Gráficos PMAPs (.png/som_AAAA-mm-dd_HH-MM-SS.png)

Referências:
-----------------------------------------------------------------------------------------------
[1] UCI Machine Learning Repository: Iris Data Set. https://archive.ics.uci.edu/ml/datasets/iris
[2] scikit-learn: Tools for Machine Learning in Python. https://scikit-learn.org/
[3] matplotlib: Visualization with Python. https://matplotlib.org/
[4] Vettigli, Giovanni. "MiniSom: minimalistic and NumPy-based implementation of the Self Organizing Map". 2018.
    https://github.com/JustGlowing/minisom
[5] TODO
[6] TODO
-----------------------------------------------------------------------------------------------
"""

import os
import sys
import random
import pickle
import platform
import glob

import numpy              as np
import pandas             as pd
import matplotlib.pyplot  as plt

from datetime import datetime
from typing   import (
  Any,
  Optional,
  Tuple,
  Mapping,
  Sequence,
)
from dataclasses import (
  dataclass,
  field,
  fields,
  replace,
)

from minisom                  import MiniSom
from matplotlib               import colormaps
from sklearn.preprocessing    import MinMaxScaler
from sklearn.model_selection  import train_test_split
from sklearn.metrics          import accuracy_score
from ucimlrepo                import fetch_ucirepo
from matplotlib.colors        import ListedColormap, BoundaryNorm

# ===== CONFIGS =====
@dataclass(frozen=True, slots=True)
class Config:
  iters:  int             = 2**16 # n_iterations;   Sugestões: 1000, 5000, 10000, 30000, 50000
  n:      int             = 10     # map_size;       Sugestões: 5, 10, 15
  m:      int             = 10     # map_size;       Sugestões: 5, 10, 15
  lr:     float           = 0.4  # learning_rate;  Sugestões: 0.5, 0.1, 0.01
  radius: Optional[float] = 5.0   # sigma_0;        Sugestões: 3, 5, 7 (Pode ser computado automaticamente)
  rand:   Optional[int]   = 52    # random_seed

  @staticmethod
  def from_args(args: dict[str, Any]) -> 'Config':
    d = {f.name: args.get(f.name, f.default) for f in fields(Config)}
    if d["radius"] is None:
      d["radius"] = max(d["m"], d["n"]) / 2
    if d["rand"] is None or d["rand"] == 0:
      d["rand"] = choose_random_key(None)
    return Config(**d)

LOG_DIR   = ".log"
MODEL_DIR = ".som"
PNG_DIR   = ".png"

# ===== ARGS =====
def parse_args(argv: list[str]) -> dict[str, Any]:
  cli = {}
  for arg in argv[1:]:
    if arg.startswith('-load='):
      cli['load'] = arg[len('-load='):]
    elif arg.startswith('-data='):
      cli['data'] = arg[len('-data='):]
    elif arg == '-ng':
      cli['ng'] = True
    else:
      for fld in fields(Config):
        prefix = f"-{fld.name}="
        if arg.startswith(prefix):
          val = arg[len(prefix):]
          if fld.type == int:
            cli[fld.name] = int(val)
          elif fld.type == float:
            cli[fld.name] = float(val)
          elif fld.type == Optional[int]:
            cli[fld.name] = int(val)
          elif fld.type == Optional[float]:
            cli[fld.name] = float(val)
  return cli

def ensure_dir(path: str) -> None:
  if not os.path.isdir(path):
    os.makedirs(path, exist_ok=True)

def current_timestr() -> str:
  return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

class DualLogger:
  def __init__(self, log_path: str):
    self.log_file = open(log_path, "a", encoding="utf-8")
    self.stdout = sys.stdout

  def write(self, msg: str):
    self.stdout.write(msg)
    self.log_file.write(msg)

  def flush(self):
    self.stdout.flush()
    self.log_file.flush()

  def close(self):
    self.log_file.close()

# ===== DADOS =====
def fetch_iris(data_path: Optional[str]=None) -> Tuple[pd.DataFrame, pd.Series]:
  if data_path:
    if data_path.endswith('.zip'):
      df = pd.read_csv(f"zip://{data_path}")
    else:
      df = pd.read_csv(data_path)
    features = df.select_dtypes(include=[np.number])
    targets = df.iloc[:, -1] if df.shape[1] > features.shape[1] else pd.Series(np.zeros(df.shape[0]), name="dummy")
    return features, targets
  iris = fetch_ucirepo(id=53)
  return iris.data.features, iris.data.targets

def scale_data(X: pd.DataFrame) -> np.ndarray:
  return MinMaxScaler().fit_transform(X)

def choose_random_key(key: Optional[int]) -> int:
  val = key if key is not None else random.randint(1, 99)
  print(f"[Chave Aleatória gerada: {val}]")
  return val

# ===== SOM =====
def train_som(X: np.ndarray, cfg: Config) -> MiniSom:
  som = MiniSom(
    x=cfg.m, y=cfg.n, input_len=X.shape[1], sigma=cfg.radius,
    learning_rate=cfg.lr, neighborhood_function="gaussian",
    random_seed=cfg.rand
  )
  som.random_weights_init(X)
  som.train(X, cfg.iters, verbose=False)
  return som

def extract_scalar_label(label: Any) -> Any:
  if isinstance(label, pd.Series):
    return label.iloc[0]
  if isinstance(label, (np.ndarray, list)) and len(label) == 1:
    return label[0]
  return label

def map_labels(X: np.ndarray, y: Any, som: MiniSom, cfg: Config) -> np.ndarray:
  label_grid = np.empty((cfg.m, cfg.n), dtype=object)
  for i in range(cfg.m):
    for j in range(cfg.n):
      label_grid[i, j] = []
  for idx, x in enumerate(X):
    row, col = som.winner(x)
    lbl = y.iloc[idx] if hasattr(y, "iloc") else y[idx]
    if hasattr(lbl, "__iter__") and not isinstance(lbl, str):
      lbl = lbl.iloc[0] if hasattr(lbl, "iloc") else lbl[0]
    label_grid[row, col].append(lbl)
  majority_labels = np.empty((cfg.m, cfg.n), dtype=object)
  for i in range(cfg.m):
    for j in range(cfg.n):
      labels = label_grid[i, j]
      if labels:
        unique_labels = set(labels)
        majority = max(unique_labels, key=labels.count)
        majority_labels[i, j] = majority
      else:
        majority_labels[i, j] = None
  return majority_labels

def map_unsupervised(X: np.ndarray, som: MiniSom, cfg: Config) -> np.ndarray:
  hits = np.zeros((cfg.m, cfg.n), dtype=int)
  for x in X:
    row, col = som.winner(x)
    hits[row, col] += 1
  return hits

def filter_valid_labels(labels: Sequence[Any]) -> list[Any]:
  return [
    val for val in labels
    if val is not None and not (isinstance(val, float) and np.isnan(val))
  ]

def flatten_labels(y: object) -> np.ndarray:
  if hasattr(y, "values"):
    arr = y.values.ravel()
  else:
    arr = np.asarray(y).ravel()
  return arr

def most_common_label(y: Sequence[Any]) -> Any:
  arr = np.array(y, dtype=object)
  arr = arr[arr != None]
  arr = arr[~pd.isnull(arr)]
  vals, counts = np.unique(arr, return_counts=True)
  return vals[np.argmax(counts)]

def predict_labels(
  X: np.ndarray, label_map: np.ndarray, som: MiniSom, fallback: Any = None
) -> np.ndarray:
  preds = []
  for x in X:
    row, col = som.winner(x)
    label = label_map[row, col]
    preds.append(label if label is not None else fallback)
  return np.array(preds, dtype=object)

# ===== PLOTTING =====
def to_rgba(arr: np.ndarray) -> np.ndarray:
  if arr.shape[-1] == 4:
    return arr
  return np.concatenate([arr, np.ones((*arr.shape[:-1], 1), dtype=arr.dtype)], axis=-1)

def plot_label_maps(
  unlab_map: np.ndarray,
  lab_map: np.ndarray,
  cfg: "Config",
  class_labels: Sequence[Any],
  save_path: Optional[str] = None,
  show: bool = True
) -> None:
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
  unl_max: float = float(unlab_map.max()) if np.any(unlab_map) else 1.0

  n_shades: int = int(unl_max) + 1
  ticks = np.arange(0, n_shades, n_shades // 5, dtype=int)
  blue_cmap = plt.get_cmap("Blues", n_shades + 1)
  blue_arr = to_rgba(blue_cmap(np.arange(2, n_shades + 1)))
  light_grey = np.array([[236/255, 236/255, 236/255, 1]])
  colors = np.vstack((light_grey, blue_arr))
  bmu_cmap = ListedColormap(colors)

  im1 = ax1.imshow(unlab_map, cmap=bmu_cmap, vmin=0, vmax=unl_max)
  ax1.set_title("Não Rotulado: BMU por Frequência")
  fig.colorbar(im1, ax=ax1, shrink=0.7, pad=0.02, ticks=ticks)
  ax1.set_xlabel("SOM X")
  ax1.set_ylabel("SOM Y")
  ax1.set_xticks(np.arange(cfg.n))
  ax1.set_yticks(np.arange(cfg.m))
  ax1.grid(False)

  flat_labels = filter_valid_labels([x for x in lab_map.ravel()])
  if not flat_labels:
    raise ValueError("Label map has no labels to plot.")

  class_set = sorted(set(flat_labels))
  class_to_int = {c: i for i, c in enumerate(class_set)}
  int_map = np.full(lab_map.shape, -1, dtype=int)
  for idx, val in np.ndenumerate(lab_map):
    int_map[idx] = class_to_int[val] if val in class_to_int else -1
  base_cmap = plt.get_cmap("tab10")
  class_colors = np.take(np.asarray(base_cmap.colors), list(range(len(class_set))), axis=0)
  class_colors_rgba = to_rgba(np.asarray(class_colors))
  sup_colors = np.vstack((light_grey, class_colors_rgba))
  sup_cmap = ListedColormap(sup_colors)
  plot_map = np.where(int_map == -1, 0, int_map + 1)

  im2 = ax2.imshow(plot_map, cmap=sup_cmap, vmin=0, vmax=sup_cmap.N - 1)
  ax2.set_title("Rotulado: BMU por Classe")
  ax2.set_xlabel("SOM X")
  ax2.set_ylabel("SOM Y")
  ax2.set_xticks(np.arange(cfg.n))
  ax2.set_yticks(np.arange(cfg.m))
  cbar = fig.colorbar(im2, ax=ax2, ticks=np.arange(1, len(class_set) + 1), pad=0.02, shrink=0.7)
  cbar.ax.set_yticklabels(class_set)
  ax2.grid(False)

  def overlay_hatch(ax, mask: np.ndarray, color: str = "#787878") -> None:
    for (i, j), flagged in np.ndenumerate(mask):
      if flagged:
        ax.add_patch(plt.Rectangle(
          (j - 0.5, i - 0.5), 1, 1,
          fill=False,
          edgecolor=color, linewidth=1.15, hatch='xxx', zorder=3
        ))

  overlay_hatch(ax1, (unlab_map == 0))
  overlay_hatch(ax2, (int_map == -1))

  fig.suptitle("SOM Visualization", fontsize=16)
  fig.text(0.5, 0.94, f"[Randomizer Key: {cfg.rand}]", ha="center", va="top", fontsize=9)
  plt.tight_layout(rect=[0, 0.03, 1, 0.95])
  if save_path:
    fig.savefig(save_path, dpi=200)
  if show:
    plt.show()
  plt.close(fig)

# ===== MÉTRICAS =====
def print_diagnostics(
  *, cfg: Config, acc: float, n_iter: int,
  log_path: str, os_name: str, os_version: str, python_version: str
) -> None:
  print("-" * 32)
  print("[SOM - Log de Informações]")
  print(f"Chave Aleatória         : {cfg.rand}")
  print(f"Iterações               : {n_iter}")
  print(f"Topologia               : {cfg.m} x {cfg.n}")
  print(f"Taxa de Aprendizado     : {cfg.lr}")
  print(f"Raio Inicial (Sigma)    : {cfg.radius:.2f}")
  print(f"Tau                     : {n_iter / np.log(cfg.radius):.2f}")
  print(f"Melhor Acurácia         : {acc:.3f}")
  print(f"Sistema Operacional     : {os_name} {os_version}")
  print(f"Python                  : {python_version}")
  print("-" * 32)

# ===== MODELO =====
def find_latest_model(model_dir: str) -> Optional[str]:
  candidates = sorted(
    glob.glob(os.path.join(model_dir, "som-model-*.som")),
    key=os.path.getmtime, reverse=True
  )
  return candidates[0] if candidates else None

def validate_model(
  model_path: str, data_path: Optional[str], cfg_override: dict[str, Any], ng: bool
) -> None:
  print(f"Carregando modelo: {model_path}")
  with open(model_path, "rb") as f:
    som = pickle.load(f)
  X, y = fetch_iris(data_path)
  X_scaled = scale_data(X)
  m, n = som.get_weights().shape[0:2]
  cfg: Config = Config(m=m, n=n, **{k: v for k, v in cfg_override.items() if k in {'lr','iters','rand','radius'}})
  print(f"Modelo tem uma topologia de {m} x {n}.")
  sup_map = map_labels(X_scaled, y, som, cfg)
  unsup_map = map_unsupervised(X_scaled, som, cfg)
  y_uniq = pd.unique(flatten_labels(y))
  y_uniq_filtered = filter_valid_labels(y_uniq)
  fallback = most_common_label(y)
  y_pred = predict_labels(X_scaled, sup_map, som, fallback=fallback)
  y_true = flatten_labels(y)
  y_pred_flat = flatten_labels(y_pred)
  acc = accuracy_score(y_true, y_pred_flat)
  os_name = platform.system()
  os_version = platform.version()
  python_version = platform.python_version()
  print_diagnostics(
    cfg=cfg, acc=acc, n_iter=cfg.iters,
    log_path="", os_name=os_name, os_version=os_version, python_version=python_version
  )
  ensure_dir(PNG_DIR)
  fig_path = os.path.join(PNG_DIR, f"validation-{current_timestr()}.png")
  plot_label_maps(unsup_map, sup_map, cfg, y_uniq_filtered, save_path=fig_path, show=not ng)
  print(f"Figura salva em: {fig_path}")

# ===== MAIN =====
def main() -> None:
  cli_args = parse_args(sys.argv)
  ensure_dir(LOG_DIR)
  ensure_dir(MODEL_DIR)
  ensure_dir(PNG_DIR)
  timestamp = current_timestr()
  log_path = os.path.join(LOG_DIR, f"iris_som_{timestamp}.log")
  logger = DualLogger(log_path)
  sys.stdout = sys.stderr = logger
  try:
    ng = cli_args.get("ng", False)
    if "load" in cli_args:
      model_path = cli_args["load"]
      if not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)
      if not os.path.exists(model_path):
        model_path = find_latest_model(MODEL_DIR)
        if model_path is None:
          print("Nenhum modelo encontrado em .som")
          return
        print(f"Nenhum modelo especificado; Utilizando o mais recente: {model_path}")
      validate_model(model_path, cli_args.get("data", None), cli_args, ng)
      return
    cfg = Config.from_args(cli_args)
    X, y = fetch_iris(cli_args.get("data"))
    X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=cfg.rand
    )
    X_train_scaled = scale_data(X_train)
    X_test_scaled = scale_data(X_test)
    som = train_som(X_train_scaled, cfg)
    unsup_map = map_unsupervised(X_train_scaled, som, cfg)
    sup_map = map_labels(X_train_scaled, y_train, som, cfg)
    y_train_uniq = pd.unique(flatten_labels(y_train))
    y_train_uniq_filtered = filter_valid_labels(y_train_uniq)
    fallback_label = most_common_label(y_train)
    y_pred = predict_labels(X_test_scaled, sup_map, som, fallback=fallback_label)
    y_true = flatten_labels(y_test)
    y_pred_flat = flatten_labels(y_pred)
    acc = accuracy_score(y_true, y_pred_flat)
    n_iter = cfg.iters
    os_name = platform.system()
    os_version = platform.version()
    python_version = platform.python_version()
    print_diagnostics(
      cfg=cfg, acc=acc, n_iter=n_iter,
      log_path=log_path, os_name=os_name,
      os_version=os_version, python_version=python_version
    )
    ensure_dir(PNG_DIR)
    fig_path = os.path.join(PNG_DIR, f"som_{timestamp}.png")
    plot_label_maps(
      unsup_map, sup_map, cfg, y_train_uniq_filtered,
      save_path=fig_path, show=not ng
    )
    print(f"Figura salva em: {fig_path}")
    model_path = os.path.join(
      MODEL_DIR, f"som-model-{timestamp}-{acc:.3f}.som"
    )
    with open(model_path, "wb") as f:
      pickle.dump(som, f)
    print(f"Modelo salvo em: {model_path}")
  finally:
    sys.stdout = logger.stdout
    sys.stderr = logger.stdout
    logger.close()

if __name__ == "__main__":
  main()