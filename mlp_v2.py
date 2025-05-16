# -*- coding: utf-8 -*-
"""
Criado em 2025.05.15 13:41 (YYYY.MM.DD HH:MM)

@autor: gabrielmsilva00 https://github.com/gabrielmsilva00
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 10 18:16:12 2025

@author: ADM
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 22:26:28 2024

@author: ADM
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 11:31:22 2024

@author: ADM
"""

# ------------------------------------------------------------------ #
#  0.0 ▸ Imports padrão
# ------------------------------------------------------------------ #
from io             import StringIO
from datetime       import datetime
from pathlib        import Path
from time           import perf_counter
from importlib.util import find_spec

from sys import (
  argv,
  executable,
)
from subprocess import (
  check_call,
  CalledProcessError,
)
from platform import (
  python_version,
  python_implementation,
  system,
  release,
  version,
  machine,
  processor,
)

# ------------------------------------------------------------------ #
#  0.1 ▸ Utilitários
# ------------------------------------------------------------------ #
tempo = perf_counter()
timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
script_name = Path(argv[0]).stem or "session"
log_dir  = Path.cwd() / ".log"
log_dir.mkdir(
  exist_ok=True,
  parents=False,
)
log_path = log_dir / f"{script_name}_{timestamp}.log"
log_buffer = StringIO()
log_buffer.write(
  f"===== EXECUÇÃO: {datetime.now()}\n"
  f"Script      : {script_name}\n"
  f"Python      : {python_version()} "
  f"({python_implementation()})\n"
  f"OS          : {system()} {release()} "
  f"{version()}\n"
  f"Machine     : {machine()}\n"
  f"Processor   : {processor()}\n"
  "============================================================\n"
)

def log(
  *msg,
  sep=' ',
  end='\n',
  log_buffer=log_buffer,
):
  """
  Escreve em stdout e em log_buffer
  """
  print(
    *msg,
    sep = sep,
    end = end
  )
  print(
    *msg,
    sep   = sep,
    end   = end,
    file  = log_buffer
  )

def summary_text():
    hyper_lines = []
    for key, val in Hyper.__dict__.items():
      if isinstance(val, (list, tuple)): val = "\n\t" + ",\n\t".join(map(str, val))
      if isinstance(val, dict): val = "\n".join(f"  {k}: {v}" for k, v in val.items())
      if not key.startswith("__"): hyper_lines.append(f"{key}: {val}")
    hyper_txt = (
        "Hiperparâmetros\n"
        "---------------\n"
        + "\n".join(hyper_lines)
    )
    return hyper_txt + f"\n\nTempo de Treinamento: {(perf_counter() - tempo):.2f}s"

def require(*pkgs, fatal=True):
  for pkg in pkgs:
    if find_spec(pkg) is None:
      try: check_call(
        [
          executable,
          "-m",
          "pip", 
          "install",
          "--quiet",
          pkg,
        ],
      )
      except CalledProcessError as e:
        print(f"Falha ao instalar dependência necessária: {pkg}\n{e}")
        if fatal: exit(1)

# ------------------------------------------------------------------ #
#  0.2 ▸ Dependências Externas
# ------------------------------------------------------------------ #
require(
  "matplotlib",
  "scikit-learn",
  "tensorflow",
  fatal = False,
)

from numpy import (
  unique,
  inf,
)
from matplotlib.pyplot import (
  subplots,
  tight_layout,
  show,
)
from sklearn.model_selection import (
  KFold,
  train_test_split,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.datasets           import make_classification
from sklearn.metrics            import (
  confusion_matrix,
  classification_report,
  accuracy_score,
)
from tensorflow.keras.models      import Sequential
from tensorflow.keras.layers      import Dense
from tensorflow.keras.metrics     import Precision, Recall
from tensorflow.keras.optimizers  import Adam
from tensorflow.keras.losses      import BinaryCrossentropy
from tensorflow.keras.callbacks   import EarlyStopping

# ------------------------------------------------------------------ #
#  1 ▸ Hiperparâmetros centralizados
# ------------------------------------------------------------------ #
class Hyper:
  # Dados
  n_samples     = 1_200
  n_features    = 20
  n_splits      = 5
  class_weights = (0.9, 0.1)
  test_size     = 0.3
  thresholds    = range(3, 9)
  random_state  = 42

  # Modelo
  max_epochs    = 1_200
  batch_size    = 100
  paciencia     = 60
  learning_rate = 1e-3
  layers        = (
    dict(
      units=12,
      activation='relu',
    ),
    dict(
      units=8,
      activation='relu',
    ),
    dict(
      units=1,
      activation='sigmoid',
    )
  )

class HyperDefault: # Evite modificar! Utilize como referência.
  # Dados
  n_samples     = 100
  n_features    = 20
  class_weights = [0.9, 0.1]
  test_size     = 0.30
  n_splits      = 5
  thresholds    = range(3, 9)
  random_state  = 42
  
  # Modelo
  max_epochs    = 50
  batch_size    = 10
  paciencia     = 5
  learning_rate = 0.001
  layers        = (
    dict(
      units=12,
      activation='relu',
    ),
    dict(
      units=8,
      activation='relu',
    ),
    dict(
      units=1,
      activation='sigmoid',
    )
  )

# ------------------------------------------------------------------ #
#  2 ▸ Geração e divisão de dados
# ------------------------------------------------------------------ #
(
  X,
  y
) = make_classification(
  n_samples     = Hyper.n_samples,
  n_features    = Hyper.n_features,
  n_classes     = 2,
  weights       = Hyper.class_weights,
  random_state  = Hyper.random_state
)

(
  X_train,
  X_test,
  y_train,
  y_test
) = train_test_split(
  X,
  y,
  test_size     = Hyper.test_size,
  random_state  = Hyper.random_state,
  stratify      = y
)

class_weights = compute_class_weight(
  class_weight  = 'balanced',
  classes       = unique(y_train),
  y             = y_train
)

class_weights_dict = {
  i: class_weights[i] for i in range(len(class_weights))
}

# ------------------------------------------------------------------ #
#  3.0 ▸ Configurações de treino
# ------------------------------------------------------------------ #
kf = KFold(
  n_splits      = Hyper.n_splits,
  shuffle       = True,
  random_state  = Hyper.random_state
)

early_stopping = EarlyStopping(
  monitor               = 'val_loss',
  patience              = Hyper.paciencia,
  restore_best_weights  = True,
  verbose               = 0
)

history_per_fold = []
metrics_per_fold = []
best_val_loss    = inf
best_model       = None

# ------------------------------------------------------------------ #
#  3.1 ▸ Loop de validação cruzada
# ------------------------------------------------------------------ #
for fold_no, (
  train_idx,
  val_idx
) in enumerate(
  kf.split(X_train),
  start=1
):

  (
    X_train_fold,
    X_val_fold
  ) = (
    X_train[train_idx],
    X_train[val_idx]
  )

  (
    y_train_fold,
    y_val_fold
  ) = (
    y_train[train_idx],
    y_train[val_idx]
  )

  model = Sequential()
  for layer in Hyper.layers:
    model.add(
      Dense(
        layer['units'],
        activation= layer['activation'],
        input_dim = (
          X_train.shape[1]
          if len(model.layers) == 0 else None
        )
      )
    )

  model.compile(
    loss            = BinaryCrossentropy(),
    optimizer       = Adam(
      learning_rate = Hyper.learning_rate
    ),
    metrics=[
      'accuracy',
      Precision(),
      Recall()
    ]
  )

  history = model.fit(
    X_train_fold,
    y_train_fold,
    epochs          = Hyper.max_epochs,
    batch_size      = Hyper.batch_size,
    validation_data = (
      X_val_fold,
      y_val_fold
    ),
    class_weight  = class_weights_dict,
    callbacks     = [early_stopping],
    verbose       = 0
  )

  scores = model.evaluate(
    X_val_fold,
    y_val_fold,
    verbose = 0
  )

  history_per_fold.append(history.history)
  metrics_per_fold.append(scores)

  log(
    f"Fold {fold_no:02d} ▸ "
    f"Loss={scores[0]:.4f} "
    f"Acc={scores[1]:.4f} "
    f"Prec={scores[2]:.4f} "
    f"Rec={scores[3]:.4f}"
  )

  if scores[0] < best_val_loss:
    best_val_loss = scores[0]
    best_model    = model

# ------------------------------------------------------------------ #
#  4 ▸ Avaliação final + thresholds
# ------------------------------------------------------------------ #
log("\n--- Avaliação no conjunto de TESTE ---")
test_scores = best_model.evaluate(
  X_test,
  y_test,
  verbose=0
)
log(
  f"Perda={test_scores[0]:.4f} | "
  f"Acurácia={test_scores[1]:.4f} | "
  f"Precisão={test_scores[2]:.4f} | "
  f"Recall={test_scores[3]:.4f}"
)

def evaluate_threshold(
  threshold: float
):
  y_pred = (
    best_model.predict(X_test) >
    threshold
  ).astype(int)

  cm = confusion_matrix(
    y_test,
    y_pred
  )
  cr = classification_report(
    y_test,
    y_pred
  )
  acc = accuracy_score(
    y_test,
    y_pred
  )

  log(
    f"\nThreshold > {threshold:.1f}"
  )
  log(
    "Confusion Matrix:\n",
    cm
  )
  log(
    "\nClassification Report:\n",
    cr
  )
  log(
    f"Acurácia: {acc:.4f}"
  )

for th in Hyper.thresholds: evaluate_threshold(th / 10)

# ------------------------------------------------------------------ #
#  5 ▸ Salvando o LOG completo
# ------------------------------------------------------------------ #
with log_path.open("w", encoding="utf-8") as f: f.write(log_buffer.getvalue())

# ------------------------------------------------------------------ #
#  6 ▸ Representação Gráfica (MatPlotLib)
# ------------------------------------------------------------------ #
(
  fig,
  axes
) = subplots(
  nrows   = 2,
  ncols   = 3,
  figsize = (20, 10),
  
)
axes = axes.flatten()

for i, (
  hist,
  ax
) in enumerate(
  zip(
    history_per_fold,
    axes
  )
):
  epochs = range(1, len(hist['loss']) + 1)

  ax.plot(
    epochs,
    hist['loss'],
    label='Treino – Loss',
  )
  ax.plot(
    epochs,
    hist['val_loss'],
    label='Val – Loss',
  )
  ax.plot(
    epochs,
    hist['accuracy'],
    label='Treino – Acc',
  )
  ax.plot(
    epochs,
    hist['val_accuracy'],
    label='Val – Acc',
  )

  ax.set_title(f"Fold {i + 1}")
  ax.set_xlabel("Épocas")
  ax.set_ylabel("Valor")
  ax.legend(
    loc='best',
    fontsize='small',
  )
  ax.grid(True)

# Subplot do painel de texto
text_ax = axes[-1]
text_ax.axis('off')
text_ax.text(
  0.0,
  1.0,
  summary_text().replace("\t","    "),
  va='top',
  ha='left',
  family='monospace',
  fontsize=12,
)

tight_layout()

# salva o gráfico em ./img
img_path = Path("img") / f"{log_path.stem}.png"
img_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(img_path, dpi=300, bbox_inches='tight')

show()
