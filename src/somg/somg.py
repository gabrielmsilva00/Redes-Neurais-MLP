import os
import pickle
import queue
import random
import threading
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from queue import Queue
from tkinter import (
  CENTER,
  BooleanVar,
  DoubleVar,
  Frame,
  IntVar,
  Label,
  StringVar,
  filedialog,
  messagebox,
)
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ttkbootstrap as ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, RegularPolygon
from matplotlib.widgets import CheckButtons
from minisom import MiniSom
from PIL import Image, ImageDraw, ImageFont, ImageGrab
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ttkbootstrap.constants import *
from ttkbootstrap.scrolled import ScrolledFrame, ScrolledText
from ttkbootstrap.tooltip import ToolTip
from ucimlrepo import fetch_ucirepo

# --- Configuração & Estruturas de Dados ---
@dataclass(frozen=True, slots=True)
class Config:
  """Configuração imutável para os parâmetros do SOM."""
  database_id: int
  m: int
  n: int
  iters: int
  lr: float
  rand_seed: int
  topology: str
  activation_distance: str
  neighborhood_function: str
  decay_function: str
  sigma_decay_function: str
  radius: Optional[float] = None

  def __post_init__(self: "Config") -> None:
    if self.radius is None:
      object.__setattr__(self, "radius", max(self.m, self.n) / 2.0)

@dataclass(slots=True)
class ModelData:
  """Container para dados do modelo SOM treinado."""
  som: MiniSom
  label_map: np.ndarray
  config: Config
  data_info: Dict[str, Any]

@dataclass(slots=True)
class Metrics:
  """Container para métricas de performance."""
  accuracy: float
  quantization_error: float
  topographic_error: float

PAD = dict(pady=2, padx=5)

# --- Módulos de Serviço ---
class DataManager:
  """Gerencia o carregamento e a preparação dos dados."""
  def __init__(self: "DataManager") -> None:
    self.title: str = "N/A"
    self.X: Optional[pd.DataFrame] = None
    self.y: Optional[pd.Series] = None
    self.meta: Optional[Dict[str, Any]] = None
    self.target_name: str = "Class"
    self.feature_names: List[str] = []

  def load_from_file(self: "DataManager", filepath: str) -> None:
    try:
      df = pd.read_csv(f"zip://{filepath}" if filepath.endswith(".zip") else filepath)
      if df.shape[1] < 2:
        raise ValueError("O dataset deve conter ao menos duas colunas (uma feature e um target).")
      
      self.y = df.iloc[:, -1]
      numeric_df = df.select_dtypes(include=np.number)
      
      self.X = numeric_df.drop(columns=[self.y.name], errors='ignore')
      if self.X.empty:
        raise ValueError("Nenhuma coluna numérica (diferente do target) foi encontrada para features.")
        
      self.title = os.path.basename(filepath)
      self.target_name = str(self.y.name)
      self.feature_names = self.X.columns.tolist()
      self.meta = {"source": "Local file", "description": f"Loaded from {self.title}"}

    except Exception as e:
      raise IOError(f"Falha ao ler ou processar o arquivo: {e}") from e

  def load_from_id(self: "DataManager", dataset_id: int) -> None:
    try:
      dataset = fetch_ucirepo(id=dataset_id)
      self.meta = dataset.metadata
      self.title = self.meta.get("name", f"Dataset ID {dataset_id}")
      self.X = dataset.data.features.select_dtypes(include=np.number)
      self.y = dataset.data.targets.iloc[:, 0]
      
      if self.X.empty:
        raise ValueError(f"O dataset '{self.title}' (ID: {dataset_id}) não possui features numéricas.")

      self.target_name = str(self.y.name)
      self.feature_names = self.X.columns.tolist()

    except Exception as e:
      raise ConnectionError(f"Não foi possível buscar o dataset ID {dataset_id}. Verifique a conexão ou o ID. Erro: {e}") from e

  def get_data_split(self: "DataManager", rand_seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if self.X is None or self.y is None:
      raise ValueError("Os dados não foram carregados.")
    X_scaled = MinMaxScaler().fit_transform(self.X)
    stratify_data = self.y if self.y.nunique() > 1 else None
    return train_test_split(
      X_scaled, self.y.values.ravel(), test_size=0.25, random_state=rand_seed, stratify=stratify_data
    )

  def get_scaled_data(self: "DataManager") -> np.ndarray:
    if self.X is None:
      raise ValueError("Os dados não foram carregados.")
    return MinMaxScaler().fit_transform(self.X)

  def get_info(self: "DataManager") -> Dict[str, Any]:
    if self.X is None or self.y is None:
      return {}
    return {
      "meta": self.meta,
      "title": self.title,
      "samples": len(self.X),
      "features": self.get_num_features(),
      "target": self.target_name,
      "classes": self.y.unique().tolist(),
    }

  def get_num_features(self: "DataManager") -> int:
    return self.X.shape[1] if self.X is not None else 0

class SOMModel:
  """Encapsula a lógica do modelo MiniSom."""
  def __init__(self: "SOMModel", config: Config, input_len: int) -> None:
    if input_len <= 0:
      raise ValueError("O comprimento de entrada (input_len) deve ser maior que zero.")
    self.config = config
    self.som = MiniSom(
      x=config.m, y=config.n, input_len=input_len,
      sigma=config.radius, learning_rate=config.lr,
      decay_function=config.decay_function,
      neighborhood_function=config.neighborhood_function,
      topology=config.topology,
      activation_distance=config.activation_distance,
      random_seed=config.rand_seed,
      sigma_decay_function=config.sigma_decay_function
    )
    self.label_map: Optional[np.ndarray] = None
    self.bmu_hits: Optional[np.ndarray] = None

  def train(
    self: "SOMModel", X_train: np.ndarray,
    update_callback: Optional[Callable[[float], None]] = None,
    stop_event: Optional[threading.Event] = None
  ) -> None:
    self.som.random_weights_init(X_train)
    total_iters = self.config.iters
    for i in range(total_iters):
      if stop_event and stop_event.is_set():
        break
      rand_idx = random.randint(0, len(X_train) - 1)
      self.som.update(X_train[rand_idx], self.som.winner(X_train[rand_idx]), i, total_iters)
      if update_callback and i % 100 == 0:
        progress = (i / total_iters) * 100
        update_callback(progress)

  def calculate_validation_metrics(
    self: "SOMModel", X_scaled: np.ndarray, y_full: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray
  ) -> Metrics:
    win_map_full = self._get_win_map(X_scaled)
    self.label_map, self.bmu_hits = self._create_label_and_hit_maps(win_map_full, y_full)
    y_pred = self._predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred) if len(y_test) > 0 else 0.0

    return Metrics(
      accuracy=accuracy,
      quantization_error=self.som.quantization_error(X_test),
      topographic_error=self.som.topographic_error(X_test),
    )

  def _get_win_map(self: "SOMModel", data: np.ndarray) -> Dict[Tuple[int, int], List[int]]:
    winmap = defaultdict(list)
    for i, x in enumerate(data):
      winmap[self.som.winner(x)].append(i)
    return winmap

  def _create_label_and_hit_maps(
    self: "SOMModel", win_map: Dict, labels: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray]:
    label_map = np.full((self.config.m, self.config.n), fill_value=None, dtype=object)
    bmu_hits = np.zeros((self.config.m, self.config.n), dtype=int)
    for pos, indices in win_map.items():
      if indices:
        bmu_hits[pos] = len(indices)
        label_map[pos] = Counter(labels[indices]).most_common(1)[0][0]
    return label_map, bmu_hits

  def _predict(self: "SOMModel", X: np.ndarray) -> np.ndarray:
    if self.label_map is None:
      raise RuntimeError("O mapa de rótulos não foi criado. Treine o modelo primeiro.")
    
    valid_labels = pd.Series(self.label_map.flatten()).dropna()
    if valid_labels.empty:
        return np.array(["N/A"] * len(X))

    fallback = valid_labels.mode()[0]
    
    def get_label(x: np.ndarray) -> Any:
      winner_pos = self.som.winner(x)
      label = self.label_map[winner_pos]
      return label if label is not None else fallback
      
    return np.array([get_label(x) for x in X])

class PlottingService:
  """Serviço centralizado para gerar todas as visualizações."""
  def __init__(self: "PlottingService", model: SOMModel, data_manager: DataManager) -> None:
    self.model = model
    self.data = data_manager
    self.class_labels = sorted(list(self.data.y.unique())) if self.data.y is not None else []
    self.class_to_int = {label: i for i, label in enumerate(self.class_labels)}
    self.cmap = plt.get_cmap("viridis", len(self.class_labels)) if self.class_labels else plt.get_cmap("viridis")

  def _create_base_figure(self: "PlottingService", figsize: Tuple[int, int] = (8, 7), **kwargs) -> Tuple[Figure, plt.Axes]:
    subplot_kw = kwargs.pop('subplot_kw', {})
    fig = Figure(figsize=figsize, dpi=100, **kwargs)
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    ax = fig.add_subplot(111, **subplot_kw)
    return fig, ax

  def _embed_figure(self: "PlottingService", fig: Figure, parent_frame: Frame) -> FigureCanvasTkAgg:
    canvas = FigureCanvasTkAgg(fig, master=parent_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
    return canvas

  def generate_all_plots(self: "PlottingService", parent_notebook: ttk.Notebook) -> None:
    plot_methods = {
      "Ativação": self.plot_bmu_hits, "Distanciamento": self.plot_umatrix,
      "Classificação": self.plot_label_map, "Clusterização": self.plot_pca_clusters_interactive,
      "Centróides": self.plot_centroids,
    }
    for name, method in plot_methods.items():
      tab = ttk.Frame(parent_notebook)
      parent_notebook.add(tab, text=name)
      try:
        method(tab)
      except Exception as e:
        Label(tab, text=f"Erro ao gerar gráfico '{name}':\n{e}", anchor=CENTER).pack(expand=True)
        import traceback; traceback.print_exc()

  def plot_generic_map(self: "PlottingService", parent_frame: Frame, title: str, data_map: np.ndarray, cmap: Any, norm: Any, cbar_labels: Optional[List[str]] = None) -> None:
    fig, ax = self._create_base_figure()
    ax.set_title(title, fontsize=12, pad=10)
    m, n = self.model.config.m, self.model.config.n
    
    is_hex = self.model.config.topology == 'hexagonal'
    
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    
    for i in range(m):
        for j in range(n):
            val = data_map[i, j]
            color = (0.9, 0.9, 0.9, 1) if val is None or (isinstance(val, float) and np.isnan(val)) else cmap(norm(val))
            
            if is_hex:
                x_coord = j + 0.5 if i % 2 == 1 else j
                y_coord = i * 0.866
                patch = RegularPolygon((x_coord, y_coord), numVertices=6, radius=0.577, facecolor=color, edgecolor='k', linewidth=0.5)
            else:
                patch = Rectangle((j-0.5, i-0.5), 1, 1, facecolor=color, edgecolor='k', linewidth=0.5)
            
            ax.add_patch(patch)

    ax.autoscale_view()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    if cbar_labels:
      ticks = np.linspace(0, len(cbar_labels) - 1, len(cbar_labels))
      cbar.set_ticks(ticks)
      cbar.set_ticklabels(cbar_labels)
    self._embed_figure(fig, parent_frame)

  def plot_label_map(self: "PlottingService", parent_frame: Frame) -> None:
    int_map = np.array([[self.class_to_int.get(l, -1) for l in row] for row in self.model.label_map])
    colors = np.vstack([np.array([0.9, 0.9, 0.9, 1]), self.cmap(np.arange(len(self.class_labels)))])
    custom_cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-1.5, len(self.class_labels) + 0.5), custom_cmap.N)
    self.plot_generic_map(parent_frame, "Mapa Supervisionado (Voto Majoritário)", int_map, custom_cmap, norm, self.class_labels)

  def plot_bmu_hits(self: "PlottingService", parent_frame: Frame) -> None:
    norm = Normalize(vmin=self.model.bmu_hits.min(), vmax=self.model.bmu_hits.max())
    self.plot_generic_map(parent_frame, "Contagem de Ativações (BMU Hits)", self.model.bmu_hits, plt.get_cmap('plasma'), norm)

  def plot_umatrix(self: "PlottingService", parent_frame: Frame) -> None:
    u_matrix = self.model.som.distance_map()
    norm = Normalize(vmin=u_matrix.min(), vmax=u_matrix.max())
    self.plot_generic_map(parent_frame, "Distância entre Neurônios (U-Matrix)", u_matrix, plt.get_cmap('bone_r'), norm)

  def plot_pca_clusters_interactive(self: "PlottingService", parent_frame: Frame) -> None:
    fig, ax = self._create_base_figure(figsize=(6, 5))
    canvas = self._embed_figure(fig, parent_frame)
    ax_check = fig.add_axes([0.75, 0.9, 0.2, 0.1], frame_on=False)

    X_scaled = self.data.get_scaled_data()
    pca = PCA(n_components=2, random_state=self.model.config.rand_seed)
    X_pca = pca.fit_transform(X_scaled)
    
    bmu_labels = None
    if self.model.label_map is not None and pd.Series(self.model.label_map.flatten()).notna().any():
      bmu_labels = self.model._predict(X_scaled)

    def redraw(show_labels: bool) -> None:
      ax.clear()
      if show_labels and self.class_labels and bmu_labels is not None:
        for i, label in enumerate(self.class_labels):
          indices = [j for j, l in enumerate(bmu_labels) if l == label]
          ax.scatter(X_pca[indices, 0], X_pca[indices, 1], c=[self.cmap(i)], label=label, s=20, alpha=0.7)
        ax.legend()
      else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=['#0000FF'], s=20, alpha=0.7)
      
      ax.set_title("Clusterização por Classe" if show_labels and bmu_labels is not None else "Clusterização", fontsize=12)
      ax.set_xlabel("Componente Principal 1"); ax.set_ylabel("Componente Principal 2")
      ax.grid(True, linestyle='--', alpha=0.6); canvas.draw_idle()

    check = CheckButtons(ax_check, [' Rótulos'], [False])
    check.on_clicked(lambda _: redraw(check.get_status()[0]))
    canvas.check = check 
    redraw(False)

  def plot_centroids(self: "PlottingService", parent_frame: Frame) -> None:
    fig, ax = self._create_base_figure(figsize=(6, 5), subplot_kw={'projection': 'polar'})
    canvas = self._embed_figure(fig, parent_frame)
    ax_check = fig.add_axes([0.75, 0.9, 0.2, 0.1], frame_on=False)

    weights = self.model.som.get_weights().reshape(-1, self.data.get_num_features())
    weights_scaled = MinMaxScaler().fit_transform(weights)
    angles = np.linspace(0, 2 * np.pi, self.data.get_num_features(), endpoint=False).tolist() + [0]
    ax.set_thetagrids(np.degrees(angles[:-1]), self.data.feature_names)

    lines, fills = [], []
    for i, label in enumerate(self.class_labels):
      neuron_indices = np.argwhere(self.model.label_map == label)
      if len(neuron_indices) > 0:
        mean_centroid = weights_scaled[neuron_indices[:, 0] * self.model.config.n + neuron_indices[:, 1]].mean(axis=0)
        data = np.concatenate((mean_centroid, [mean_centroid[0]]))
        line, = ax.plot(angles, data, label=label, linewidth=2, color=self.cmap(i))
        fill = ax.fill(angles, data, alpha=0.25, color=self.cmap(i))
        lines.append(line); fills.append(fill)

    def redraw(show_labels: bool) -> None:
      for line, fill_patches in zip(lines, fills):
        line.set_visible(show_labels)
        for patch in fill_patches:
          patch.set_color(line.get_color() if show_labels else '#0000FF')
      ax.legend(loc='upper left', bbox_to_anchor=(-0.1, 1.1))
      ax.get_legend().set_visible(show_labels)
      ax.set_title("Centróides por Classe" if show_labels else "Centróides", fontsize=12)
      canvas.draw_idle()

    check = CheckButtons(ax_check, [' Rótulos'], [False])
    check.on_clicked(lambda _: redraw(check.get_status()[0]))
    canvas.check = check
    redraw(False)

# --- Classe Principal da UI ---
class UIController(ttk.Window):
  """Controlador da UI, orquestra a aplicação."""
  def __init__(self: "UIController") -> None:
    super().__init__(themename="yeti")
    self.title("SOM - Análise de Dados com Mapas Auto-Organizáveis")
    self.geometry("1650x825")
    
    style = ttk.Style()
    style.configure('.', font=('Courier New', 10))
    style.configure('TCombobox', justify=CENTER)
    style.map('TCombobox', fieldbackground=[('readonly', 'white')])
    style.map('TCombobox', selectbackground=[('readonly', 'white')])
    style.map('TCombobox', selectforeground=[('readonly', 'black')])
    self.meta_font = ('Courier New', 14)
    self.report_font = ('Courier New', 14)
    
    self.data_manager = DataManager()
    self.som_model: Optional[SOMModel] = None
    self.task_queue = Queue()
    self.stop_event = threading.Event()
    self.current_worker: Optional[threading.Thread] = None

    self._setup_vars()
    self._setup_ui()
    self._reset_config_to_defaults()

    self.process_queue()
    self._load_data_by_id()

  def _validate_int(self: "UIController", value_if_allowed: str) -> bool:
    return value_if_allowed.isdigit() or value_if_allowed == ""

  def _validate_float(self: "UIController", value_if_allowed: str) -> bool:
    if not value_if_allowed: return True
    parts = value_if_allowed.split('.')
    if len(parts) > 2: return False
    return all(part.isdigit() for part in parts)

  def _open_combobox(self: "UIController", event: Any) -> None:
    event.widget.event_generate('<Down>')

  def _defocus_combobox(self: "UIController", event: Any) -> None:
    self.focus_set()

  def _setup_vars(self: "UIController") -> None:
    self.m_var = IntVar()
    self.n_var = IntVar()
    self.iters_var = IntVar()
    self.lr_var = DoubleVar()
    self.radius_var = DoubleVar()
    self.radius_is_auto = BooleanVar(value=True)
    self.rand_seed_var = IntVar()
    self.rand_seed_is_random = BooleanVar(value=True)
    self.database_id_var = IntVar()
    self.progress_var = DoubleVar()
    self.report_tab_exists = False
    
    self.topology_var = StringVar()
    self.activation_distance_var = StringVar()
    self.neighborhood_function_var = StringVar()
    self.decay_function_var = StringVar()
    self.sigma_decay_function_var = StringVar()

  def _setup_ui(self: "UIController") -> None:
    main_frame = ttk.Frame(self, padding=10)
    main_frame.pack(fill=BOTH, expand=True)
    
    control_frame = ttk.Frame(main_frame, width=380, style='Card.TFrame')
    control_frame.pack(side=LEFT, fill=Y, padx=(0, 10))
    control_frame.pack_propagate(False)
    
    scrolled_control = ScrolledFrame(control_frame, autohide=True)
    scrolled_control.pack(fill=BOTH, expand=True)
    
    self._create_status_panel(scrolled_control)
    self._create_data_panel(scrolled_control)
    self._create_config_panel(scrolled_control)

    plot_frame = ttk.Frame(main_frame)
    plot_frame.pack(side=RIGHT, fill=BOTH, expand=True)
    self.notebook = ttk.Notebook(plot_frame)
    self.notebook.pack(fill=BOTH, expand=True, pady=(0, 5))
    self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
    
    self._create_metadata_tab()
    
    self.save_graph_button_frame = ttk.Frame(plot_frame)
    self.save_graph_button = ttk.Button(self.save_graph_button_frame, text="Salvar Gráfico", command=self.save_current_graph)
    self.save_graph_button.pack(side=LEFT, fill=X, expand=True, padx=(0, 2))
    self.save_with_report_button = ttk.Button(self.save_graph_button_frame, text="Salvar Gráfico com Relatório", command=self.save_graph_with_report)
    self.save_with_report_button.pack(side=LEFT, fill=X, expand=True, padx=(2, 0))

  def _create_status_panel(self: "UIController", parent: Frame) -> None:
    panel = ttk.LabelFrame(parent, text="Status", padding=10)
    panel.pack(fill=X, pady=5, padx=5)
    self.status_label = ttk.Label(panel, text="Inicializando...", wraplength=340, justify=CENTER)
    self.status_label.pack(fill=X, pady=(0, 5))
    self.progress_bar = ttk.Progressbar(panel, variable=self.progress_var, mode='determinate')
    self.progress_bar.pack(fill=X, pady=5)
    self.stop_button = ttk.Button(panel, text="Interromper", command=self.stop_task, state=DISABLED)
    self.stop_button.pack(fill=X, pady=(5,0))

  def _create_data_panel(self: "UIController", parent: Frame) -> None:
    panel = ttk.LabelFrame(parent, text="Dados", padding=10)
    panel.pack(fill=X, pady=5, padx=5)
    
    self.db_title_label = ttk.Label(panel, text="Título: N/A", wraplength=320)
    self.db_title_label.pack(anchor='w', pady=(0, 5))
    
    id_frame = ttk.Frame(panel)
    id_frame.pack(fill=X, pady=(0, 5))
    ttk.Label(id_frame, text="ID:").pack(side=LEFT, padx=(0, 5))
    vcmd_int = (self.register(self._validate_int), '%P')
    self.db_id_entry = ttk.Entry(id_frame, textvariable=self.database_id_var, width=10, justify=CENTER, validate="key", validatecommand=vcmd_int)
    self.db_id_entry.pack(side=LEFT)
    self.load_id_button = ttk.Button(id_frame, text="Buscar Online", command=self._load_data_by_id)
    self.load_id_button.pack(side=RIGHT, fill=X, expand=True, padx=(5, 0))

    self.load_data_button = ttk.Button(panel, text="Carregar Arquivo Local", command=self._load_custom_data)
    self.load_data_button.pack(fill=X)

  def _create_config_panel(self: "UIController", parent: Frame) -> None:
    panel = ttk.LabelFrame(parent, text="Configuração do SOM", padding=10)
    panel.pack(fill=X, pady=5, padx=5)

    def create_spinbox_row(p, label_text, var, from_, to, increment):
        row = ttk.Frame(p); row.pack(fill=X, expand=True, pady=2)
        ttk.Label(row, text=label_text, width=18).pack(side=LEFT, anchor='w')
        spinbox = ttk.Spinbox(row, textvariable=var, from_=from_, to=to, increment=increment, width=20, justify=CENTER)
        spinbox.pack(side=RIGHT)
        return spinbox

    def create_combo_row(p, label_text, var, values):
        row = ttk.Frame(p); row.pack(fill=X, expand=True, pady=2)
        ttk.Label(row, text=label_text, width=18).pack(side=LEFT, anchor='w')
        combo = ttk.Combobox(row, textvariable=var, values=values, state='readonly', width=20, justify=CENTER)
        combo.bind('<Button-1>', self._open_combobox)
        combo.bind("<<ComboboxSelected>>", self._defocus_combobox)
        combo.pack(side=RIGHT)
        return combo

    self.spinboxes = [
        create_spinbox_row(panel, "Linhas", self.m_var, 1, 100, 1),
        create_spinbox_row(panel, "Colunas", self.n_var, 1, 100, 1),
        create_spinbox_row(panel, "Iterações", self.iters_var, 100, 1000000, 100),
        create_spinbox_row(panel, "Aprendizado", self.lr_var, 0.005, 5.0, 0.005)
    ]

    combos = {
        "Topologia": (self.topology_var, ['rectangular', 'hexagonal']),
        "Distância de Ativação": (self.activation_distance_var, ['euclidean', 'cosine', 'manhattan', 'chebyshev']),
        "Função de Vizinhança": (self.neighborhood_function_var, ['gaussian', 'mexican_hat', 'bubble', 'triangle']),
        "Decaimento do Aprend.": (self.decay_function_var, ['asymptotic_decay', 'inverse_decay_to_zero', 'linear_decay_to_zero']),
        "Decaimento do Sigma": (self.sigma_decay_function_var, ['asymptotic_decay', 'inverse_decay_to_one', 'linear_decay_to_one'])
    }
    self.comboboxes = [create_combo_row(panel, text, var, val) for text, (var, val) in combos.items()]

    def create_toggle_row(p, label_text, var, check_var, command, vcmd):
        row_frame = ttk.Frame(p); row_frame.pack(fill=X, expand=True, pady=2)
        ttk.Label(row_frame, text=label_text, width=18).pack(side=LEFT, anchor='w')
        right_frame = ttk.Frame(row_frame)
        right_frame.pack(side=RIGHT)
        ttk.Label(right_frame, text="Auto", anchor='center').pack(side=LEFT, fill=Y, padx=2)
        check = ttk.Checkbutton(right_frame, variable=check_var, command=command, style='Roundtoggle.Toolbutton')
        check.pack(side=LEFT, padx=(0, 5))
        entry = ttk.Entry(right_frame, textvariable=var, width=12, state=DISABLED, justify=CENTER, validate="key", validatecommand=vcmd)
        entry.pack(side=LEFT)
        return entry, check
    
    vcmd_int = (self.register(self._validate_int), '%P')
    vcmd_float = (self.register(self._validate_float), '%P')

    self.radius_entry, self.radius_check = create_toggle_row(panel, "Raio", self.radius_var, self.radius_is_auto, self._toggle_radius_entry, vcmd_float)
    self.seed_entry, self.seed_check = create_toggle_row(panel, "Semente", self.rand_seed_var, self.rand_seed_is_random, self._toggle_seed_entry, vcmd_int)
    
    action_frame = ttk.Frame(panel); action_frame.pack(fill=X, expand=True, pady=(10, 0))
    self.run_button = ttk.Button(action_frame, text="Executar Treinamento", command=self.start_training, style='Accent.TButton')
    self.run_button.pack(fill=X, pady=(0, 5))
    self.load_model_button = ttk.Button(action_frame, text="Carregar Modelo", command=self.load_model)
    self.load_model_button.pack(fill=X, pady=(0, 5))
    self.save_model_button = ttk.Button(action_frame, text="Salvar Modelo", command=self.save_model, state=DISABLED)
    self.save_model_button.pack(fill=X)

  def _create_metadata_tab(self: "UIController") -> None:
    self.metadata_tab = ttk.Frame(self.notebook)
    self.notebook.add(self.metadata_tab, text="Metadados")
    
    metadata_scrolled_text = ScrolledText(self.metadata_tab, wrap=WORD, font=self.meta_font)
    metadata_scrolled_text.pack(fill=BOTH, expand=True, pady=5, padx=5)
    self.metadata_text = metadata_scrolled_text.text
    self.metadata_text.config(state=DISABLED)
    
    btn_frame = ttk.Frame(self.metadata_tab)
    btn_frame.pack(fill=X, padx=5, pady=(0, 5))
    self.copy_meta_button = ttk.Button(btn_frame, text="Copiar Metadados", command=self._copy_metadata_text)
    self.copy_meta_button.pack(side=LEFT, expand=True, fill=X, padx=(0, 2))
    self.save_meta_button = ttk.Button(btn_frame, text="Salvar Metadados (.log)", command=self._save_metadata_log)
    self.save_meta_button.pack(side=LEFT, expand=True, fill=X, padx=(2, 0))
    
  def _create_report_tab(self: "UIController") -> None:
    if self.report_tab_exists: return
    self.report_tab = ttk.Frame(self.notebook)
    self.notebook.add(self.report_tab, text="Relatório")
    
    report_scrolled_text = ScrolledText(self.report_tab, wrap=WORD, font=self.report_font)
    report_scrolled_text.pack(fill=BOTH, expand=True, pady=5, padx=5)
    self.report_text = report_scrolled_text.text
    self.report_text.config(state=DISABLED)
    
    btn_frame = ttk.Frame(self.report_tab)
    btn_frame.pack(fill=X, padx=5, pady=(0, 5))
    self.copy_report_button = ttk.Button(btn_frame, text="Copiar Relatório", command=self._copy_report_text)
    self.copy_report_button.pack(side=LEFT, expand=True, fill=X, padx=(0, 2))
    self.save_report_button = ttk.Button(btn_frame, text="Salvar Relatório (.log)", command=self._save_report_log)
    self.save_report_button.pack(side=LEFT, expand=True, fill=X, padx=(2, 0))
    self.report_tab_exists = True
  
  def _on_tab_changed(self, event: Any) -> None:
      try:
          selected_tab_text = self.notebook.tab(self.notebook.select(), "text")
          if selected_tab_text in ["Metadados", "Relatório"]:
              self.save_graph_button_frame.pack_forget()
          else:
              self.save_graph_button_frame.pack(side=BOTTOM, fill=X, pady=(5,0))
      except Exception:
          self.save_graph_button_frame.pack_forget()

  def _reset_config_to_defaults(self: "UIController") -> None:
    self.database_id_var.set(109)
    self.m_var.set(15); self.n_var.set(15)
    self.iters_var.set(5000); self.lr_var.set(0.5)
    self.radius_is_auto.set(True)
    self.rand_seed_is_random.set(True)
    self.rand_seed_var.set(42)
    self.topology_var.set('rectangular')
    self.activation_distance_var.set('euclidean')
    self.neighborhood_function_var.set('gaussian')
    self.decay_function_var.set('asymptotic_decay')
    self.sigma_decay_function_var.set('asymptotic_decay')
    self._toggle_radius_entry()
    self._toggle_seed_entry()
    
  def _toggle_radius_entry(self: "UIController") -> None:
    self.radius_entry.config(state=DISABLED if self.radius_is_auto.get() else NORMAL)

  def _toggle_seed_entry(self: "UIController") -> None:
    self.seed_entry.config(state=DISABLED if self.rand_seed_is_random.get() else NORMAL)

  def _copy_text_from_widget(self, widget: ttk.Text, button: ttk.Button, original_text: str) -> None:
      self.clipboard_clear()
      self.clipboard_append(widget.get("1.0", END))
      button.config(text="Copiado!")
      self.after(2000, lambda: button.config(text=original_text))

  def _save_text_from_widget(self, widget: ttk.Text, title: str) -> None:
      filepath = filedialog.asksaveasfilename(
        defaultextension=".log", filetypes=[("Log Files", "*.log")], title=title
      )
      if not filepath: return
      try:
        with open(filepath, "w", encoding='utf-8') as f:
          f.write(widget.get("1.0", END))
        self._update_status(f"Arquivo salvo em {os.path.basename(filepath)}")
      except Exception as e:
        messagebox.showerror("Erro ao Salvar", f"Não foi possível salvar o arquivo: {e}")

  def _copy_metadata_text(self: "UIController") -> None:
    self._copy_text_from_widget(self.metadata_text, self.copy_meta_button, "Copiar Metadados")

  def _save_metadata_log(self: "UIController") -> None:
    self._save_text_from_widget(self.metadata_text, "Salvar Metadados Como...")

  def _copy_report_text(self: "UIController") -> None:
    self._copy_text_from_widget(self.report_text, self.copy_report_button, "Copiar Relatório")

  def _save_report_log(self: "UIController") -> None:
    self._save_text_from_widget(self.report_text, "Salvar Relatório Como...")

  def _set_ui_state(self: "UIController", is_running: bool) -> None:
    state = DISABLED if is_running else NORMAL
    widgets_to_toggle = [
        self.db_id_entry, self.load_id_button, self.radius_entry, self.seed_entry,
        self.radius_check, self.seed_check, self.run_button, self.load_data_button,
        self.load_model_button, self.save_model_button, *self.comboboxes, *self.spinboxes
    ]
    for widget in widgets_to_toggle:
        try: widget.config(state=state)
        except Exception: pass

    if not is_running:
        self._toggle_radius_entry()
        self._toggle_seed_entry()
        self.save_model_button.config(state=NORMAL if self.som_model else DISABLED)
        self._on_tab_changed(None)
    else:
        self.save_graph_button_frame.pack_forget()

    self.stop_button.config(state=NORMAL if is_running else DISABLED)

  def process_queue(self: "UIController") -> None:
    try:
      while True:
        task, args, kwargs = self.task_queue.get_nowait()
        task(*args, **kwargs)
    except queue.Empty:
      pass
    finally:
      self.after(100, self.process_queue)

  def _queue_update(self: "UIController", task: Callable, *args: Any, **kwargs: Any) -> None:
    self.task_queue.put((task, args, kwargs))

  def _execute_in_thread(self: "UIController", target_func: Callable, *args: Any, **kwargs: Any) -> None:
    self.stop_event.clear()
    self._queue_update(self._set_ui_state, is_running=True)
    self.current_worker = threading.Thread(target=target_func, args=args, kwargs=kwargs, daemon=True)
    self.current_worker.start()
  
  def _update_status(self: "UIController", message: str, progress: Optional[float] = None, style: str = "info") -> None:
    self.status_label.config(text=message)
    if progress is not None:
      self.progress_var.set(progress)
      self.progress_bar.config(bootstyle=style)

  def _update_data_info(self: "UIController") -> None:
    info = self.data_manager.get_info()
    title = info.get("title", "N/A")
    self.db_title_label.config(text=f"Título: {title}")
    
    for i in self.notebook.tabs()[1:]: self.notebook.forget(i)
    self.report_tab_exists = False
    
    metadata_report = self._get_metadata_text()
    
    self.metadata_text.config(state=NORMAL)
    self.metadata_text.delete("1.0", END)
    self.metadata_text.insert(END, metadata_report + "\n\n")
    self.metadata_text.config(state=DISABLED)

  def _load_data_by_id(self: "UIController") -> None:
      try:
          dataset_id = self.database_id_var.get()
          self._execute_in_thread(self._load_data_async, load_func=self.data_manager.load_from_id, source=dataset_id)
      except Exception:
          messagebox.showerror("ID Inválido", "Por favor, insira um ID de dataset numérico válido.")

  def _load_custom_data(self: "UIController") -> None:
    path = filedialog.askopenfilename(title="Selecione um arquivo .csv ou .zip", filetypes=[("CSV e ZIP", "*.csv *.zip")])
    if not path: return
    self._execute_in_thread(self._load_data_async, load_func=self.data_manager.load_from_file, source=path)

  def _load_data_async(self: "UIController", load_func: Callable, source: Any) -> None:
    status = f"Buscando ID {source}..." if isinstance(source, int) else f"Carregando '{os.path.basename(str(source))}'..."
    self._queue_update(self._update_status, status, 0)
    try:
      load_func(source)
      self._queue_update(self._update_data_info)
      self._queue_update(self._update_status, f"Dataset '{self.data_manager.title}' carregado.", 100, "success")
    except Exception as e:
      self._queue_update(messagebox.showerror, "Erro de Dados", str(e))
      self._queue_update(self._update_status, "Falha ao carregar dados.", 100, "danger")
    finally:
      self._queue_update(self._set_ui_state, is_running=False)

  def _parse_config(self: "UIController") -> Config:
    try:
      seed = random.randint(0, 100) if self.rand_seed_is_random.get() else self.rand_seed_var.get()
      radius = None if self.radius_is_auto.get() else self.radius_var.get()
      return Config(
          database_id=self.database_id_var.get(), m=self.m_var.get(), n=self.n_var.get(),
          iters=self.iters_var.get(), lr=self.lr_var.get(), radius=radius, rand_seed=seed,
          topology=self.topology_var.get(),
          activation_distance=self.activation_distance_var.get(),
          neighborhood_function=self.neighborhood_function_var.get(),
          decay_function=self.decay_function_var.get(),
          sigma_decay_function=self.sigma_decay_function_var.get()
      )
    except Exception as e:
      raise ValueError(f"Parâmetro de configuração inválido: {e}")

  def start_training(self: "UIController") -> None:
    if self.data_manager.X is None:
      messagebox.showwarning("Sem Dados", "Carregue um conjunto de dados antes de treinar.")
      return
    try:
      config = self._parse_config()
      self._execute_in_thread(self._training_worker, config)
    except ValueError as e:
      messagebox.showerror("Erro de Configuração", str(e))

  def _training_worker(self: "UIController", config: Config) -> None:
    try:
      self._set_random_seed(config.rand_seed)
      self._queue_update(self._update_status, "1/4 - Preparando dados...", 5)
      num_features = self.data_manager.get_num_features()
      X_train, X_test, _, y_test = self.data_manager.get_data_split(config.rand_seed)
      
      self.som_model = SOMModel(config, num_features)
      
      training_status_msg = "2/4 - Treinando o SOM..."
      self._queue_update(self._update_status, training_status_msg, 15)
      self.som_model.train(X_train, lambda p: self._queue_update(self._update_status, training_status_msg, progress=15 + p * 0.6), self.stop_event)
      if self.stop_event.is_set(): raise InterruptedError
      
      self._queue_update(self._update_status, "3/4 - Validando e calculando métricas...", 80)
      metrics = self.som_model.calculate_validation_metrics(self.data_manager.get_scaled_data(), self.data_manager.y.values.ravel(), X_test, y_test)
      
      self._queue_update(self._create_report_tab)
      report_text = self._get_training_report_text(metrics=metrics, config=config)
      self._queue_update(self._update_report_ui, report_text, clear=True)
      
      self._queue_update(self._update_status, "4/4 - Gerando visualizações...", 95)
      self._queue_update(self.display_results)
      self._queue_update(self._update_status, "Processo concluído.", 100, "success")

    except InterruptedError:
      self._queue_update(self._update_status, "Processo interrompido.", 100, "warning")
    except Exception as e:
      self._queue_update(self._update_status, f"Erro: {e}", 100, "danger")
      self._queue_update(messagebox.showerror, "Erro no Processo", f"Ocorreu um erro: {e}")
      import traceback; traceback.print_exc()
    finally:
      self._queue_update(self._set_ui_state, is_running=False)

  def stop_task(self: "UIController") -> None:
    if self.current_worker and self.current_worker.is_alive():
      self.stop_event.set()
      self._update_status("Interrompendo...", style="warning")
      self.stop_button.config(state=DISABLED)

  def _set_random_seed(self, seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
  def get_dict_text(self: "UIController", d: Dict[str, Any]) -> str:
    lines = []
    pointers = ('├', '└')

    for k, v in d.items():
      pointer = pointers[len(d) - 1 == list(d.keys()).index(k)]
      line = f"{pointer}─{k}:\n"

      if isinstance(v, dict):
        line += '\n'.join(f"\t{l}" for l in self.get_dict_text(v).splitlines()) + '\n'
      elif isinstance(v, (bytes, str)) and len(v) > 100:
        line += f"\t┊\n\t┊┈{v.rstrip("\n")}\n┊┈┈┈┈┈"
      else:
        line += f"\t└─{v}\n"
      lines.append(line.expandtabs(4))

    return '\n'.join(lines)

  def _get_metadata_text(self) -> str:
      info = self.data_manager.get_info()
      report_lines = [f"Metadados do Dataset: {info.get('title', 'N/A')}"]
      report_lines.append("=" * len(report_lines[0]))
      
      if info.get("meta"):
          report_lines.append(self.get_dict_text(info["meta"]))
      else:
          report_lines.append("Nenhum metadado disponível. Carregue um dataset.")
      return "\n".join(report_lines)

  def _get_training_report_text(self, metrics: Metrics, config: Config) -> str:
    report_lines = [f"Relatório de Treinamento: {self.data_manager.get_info().get('title', 'N/A')}"]
    report_lines.append("=" * len(report_lines[0]))
    
    radius_str = f"{config.radius:.2f}" if isinstance(config.radius, float) else "auto"
    report_lines.append("\nConfiguração do Treinamento\n" + "-" * 27)
    config_dict = {
        "Semente Aleatória": config.rand_seed, "Topologia": config.topology,
        "Dist. de Ativação": config.activation_distance, "Func. de Vizinhança": config.neighborhood_function,
        "Decaimento Aprend.": config.decay_function, "Decaimento Sigma": config.sigma_decay_function,
        "Iterações": config.iters, "Taxa de Aprendizado": config.lr, "Raio Inicial": radius_str
    }
    for k, v in config_dict.items():
        report_lines.append(f"● {k}:\n\t○ {v}".expandtabs(4))

    report_lines.append("\nMétricas de Performance\n" + "-" * 23)
    metrics_dict = {
        "Acurácia": f"{metrics.accuracy*100:.4f}",
        "Erro de Quantização": f"{metrics.quantization_error*100:.4f}%",
        "Erro Topográfico": f"{metrics.topographic_error*100:.4f}%"
    }
    for k, v in metrics_dict.items():
        report_lines.append(f"● {k}:\n\t○ {v}".expandtabs(4))
            
    return "\n".join(report_lines)

  def _update_report_ui(self: "UIController", text: str, clear: bool = False) -> None:
    self.report_text.config(state=NORMAL)
    if clear: self.report_text.delete("1.0", END)
    self.report_text.insert(END, text + "\n\n")
    self.report_text.config(state=DISABLED)

  def display_results(self: "UIController") -> None:
    if not self.som_model: return
    try:
      for i in self.notebook.tabs()[2:]: self.notebook.forget(i)
    except IndexError: pass
    plotter = PlottingService(self.som_model, self.data_manager)
    plotter.generate_all_plots(self.notebook)
    self.notebook.select(self.notebook.tabs()[1])

  def save_current_graph(self: "UIController") -> None:
    self.save_graph_with_report(include_report=False)

  def save_graph_with_report(self, include_report: bool = True) -> None:
    selected_tab_text = self.notebook.tab(self.notebook.select(), "text")
    if selected_tab_text in ["Metadados", "Relatório"]:
        messagebox.showwarning("Aba Inválida", "Selecione uma aba de gráfico para salvar.")
        return

    try:
        selected_tab = self.notebook.nametowidget(self.notebook.select())
        canvas = selected_tab.winfo_children()[0]
        x, y = canvas.winfo_rootx(), canvas.winfo_rooty()
        w, h = canvas.winfo_width(), canvas.winfo_height()
        graph_img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
    except (IndexError, AttributeError):
        messagebox.showerror("Erro", "Não foi possível capturar a imagem do gráfico.")
        return

    filepath = filedialog.asksaveasfilename(
        defaultextension=".png", filetypes=[("PNG", "*.png")], title="Salvar Imagem Como..."
    )
    if not filepath: return
    
    report_text = ""
    if include_report and self.som_model and self.report_tab_exists:
        report_text = self.report_text.get("1.0", END).strip()

    if not report_text:
        graph_img.save(filepath)
        self._update_status(f"Gráfico salvo em {os.path.basename(filepath)}")
        return

    try:
        report_width, padding, bg_color = 400, 20, "#FFFFFF"
        _, font_size = self.report_font
        try: font = ImageFont.truetype("cour.ttf", font_size)
        except IOError: font = ImageFont.load_default()

        temp_draw = ImageDraw.Draw(Image.new("RGB", (0, 0)))
        wrapped_text = ""
        for line in report_text.split('\n'):
            if temp_draw.textlength(line, font=font) <= report_width - 2 * padding:
                wrapped_text += line + '\n'
                continue
            
            new_line, words = "", line.split(' ')
            for word in words:
                if temp_draw.textlength(f"{new_line} {word}", font=font) < report_width - 2 * padding:
                    new_line += f" {word}"
                else:
                    wrapped_text += new_line.strip() + '\n'
                    new_line = f"  {word}"
            wrapped_text += new_line.strip() + '\n'

        wrapped_text = wrapped_text.strip()
        text_bbox = temp_draw.multiline_textbbox((0, 0), wrapped_text, font=font, align="left")
        text_height = text_bbox[3] - text_bbox[1]

        final_height = max(text_height + 2 * padding, graph_img.height)
        report_img = Image.new("RGB", (report_width, final_height), bg_color)
        draw = ImageDraw.Draw(report_img)

        text_y = (final_height - text_height) / 2
        draw.multiline_text((padding, text_y), wrapped_text, font=font, fill="black", align="left")

        # Adjust the final image creation to avoid overlap between the report text and the graph
        final_img = Image.new("RGB", (report_img.width + graph_img.width, final_height), bg_color)
        final_img.paste(graph_img, (0, 0))
        final_img.paste(report_img, (int(graph_img.width * 0.95), 0))
        final_img.save(filepath)
        self._update_status(f"Imagem composta salva em {os.path.basename(filepath)}")
    except Exception as e:
        messagebox.showerror("Erro ao Salvar", f"Falha ao criar imagem composta: {e}")
        import traceback; traceback.print_exc()

  def save_model(self: "UIController") -> None:
    if not self.som_model:
      messagebox.showerror("Erro", "Nenhum modelo treinado para salvar.")
      return
    filepath = filedialog.asksaveasfilename(
        defaultextension=".som", filetypes=[("Modelo SOM", "*.som")], title="Salvar Modelo Como..."
    )
    if not filepath: return
    model_data = ModelData(
      som=self.som_model.som, label_map=self.som_model.label_map,
      config=self.som_model.config, data_info=self.data_manager.get_info()
    )
    try:
      with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
      self._update_status(f"Modelo salvo em {os.path.basename(filepath)}")
    except Exception as e:
      messagebox.showerror("Erro ao Salvar", f"Não foi possível salvar o modelo: {e}")

  def load_model(self: "UIController") -> None:
    if not self.data_manager.X:
      messagebox.showwarning("Sem Dados", "Carregue dados para validação antes de carregar um modelo.")
      return
    filepath = filedialog.askopenfilename(filetypes=[("Modelo SOM", "*.som")], title="Carregar Modelo")
    if not filepath: return
    try:
      with open(filepath, 'rb') as f:
        model_data: ModelData = pickle.load(f)
      
      num_features = self.data_manager.get_num_features()
      model_features = model_data.som.get_weights().shape[-1]
      if model_features != num_features:
        raise ValueError(f"Incompatibilidade! Modelo treinado com {model_features} features, dados atuais têm {num_features}.")

      cfg = model_data.config
      self.m_var.set(cfg.m); self.n_var.set(cfg.n)
      self.iters_var.set(cfg.iters); self.lr_var.set(cfg.lr)
      self.rand_seed_is_random.set(False); self.rand_seed_var.set(cfg.rand_seed)
      self.radius_is_auto.set(cfg.radius is None)
      if cfg.radius is not None: self.radius_var.set(cfg.radius)
      
      self.topology_var.set(cfg.topology)
      self.activation_distance_var.set(cfg.activation_distance)
      self.neighborhood_function_var.set(cfg.neighborhood_function)
      self.decay_function_var.set(cfg.decay_function)
      self.sigma_decay_function_var.set(cfg.sigma_decay_function)

      self._toggle_radius_entry(); self._toggle_seed_entry()
      
      self.som_model = SOMModel(cfg, num_features)
      self.som_model.som = model_data.som
      
      self._update_status("Modelo carregado. Re-validando com dados atuais...")
      self._execute_in_thread(self._validation_worker_from_load, model_data)

    except Exception as e:
      messagebox.showerror("Erro ao Carregar", f"Arquivo inválido ou erro: {e}")
      self._update_status(f"Falha ao carregar modelo.", 100, "danger")

  def _validation_worker_from_load(self: "UIController", model_data: ModelData) -> None:
    try:
        self._set_random_seed(model_data.config.rand_seed)
        self._queue_update(self._update_status, "1/3 - Preparando dados para validação...", 10)
        _, X_test, _, y_test = self.data_manager.get_data_split(model_data.config.rand_seed)
        
        self.som_model.label_map = model_data.label_map
        
        self._queue_update(self._update_status, "2/3 - Calculando métricas...", 50)
        win_map = self.som_model._get_win_map(self.data_manager.get_scaled_data())
        _, self.som_model.bmu_hits = self.som_model._create_label_and_hit_maps(win_map, self.data_manager.y.values.ravel())
        
        y_pred = self.som_model._predict(X_test)
        metrics = Metrics(
            accuracy=accuracy_score(y_test, y_pred) if len(y_test) > 0 else 0.0,
            quantization_error=self.som_model.som.quantization_error(X_test),
            topographic_error=self.som_model.som.topographic_error(X_test)
        )
        self._queue_update(self._create_report_tab)
        report = self._get_training_report_text(metrics=metrics, config=model_data.config)
        self._queue_update(self._update_report_ui, report, clear=True)
        
        self._queue_update(self._update_status, "3/3 - Gerando visualizações...", 90)
        self._queue_update(self.display_results)
        self._queue_update(self._update_status, "Modelo revalidado com sucesso.", 100, "success")

    except Exception as e:
        self._queue_update(self._update_status, f"Erro na validação: {e}", 100, "danger")
        self._queue_update(messagebox.showerror, "Erro na Validação", f"Ocorreu um erro: {e}")
    finally:
        self._queue_update(self._set_ui_state, is_running=False)

  def on_closing(self: "UIController") -> None:
    if messagebox.askokcancel("Sair", "Deseja realmente sair?"):
      self.stop_task()
      if self.current_worker and self.current_worker.is_alive():
        self.current_worker.join(timeout=1.0)
      self.destroy()

if __name__ == "__main__":
  warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
  app = UIController()
  app.protocol("WM_DELETE_WINDOW", app.on_closing)
  app.mainloop()