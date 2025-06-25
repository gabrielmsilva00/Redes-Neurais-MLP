"""
mlp.py
=======
Autoria:    Gabriel Maia, em https://github.com/gabrielmsilva00
Repo:       https://github.com/gabrielmsilva00/Redes-Neurais-MLP

Treina, avalida e inspeciona visualmente um classificador *Multilayer-Perceptron*
a partir de conjuntos CSV binários.  

O script pode ser usado tanto via linha de comando quanto de forma interativa,
exibindo um painel gráfico com métricas e matrizes de confusão para cada *fold*.

FUNCIONALIDADES PRINCIPAIS
--------------------------
• Treinamento k-fold estratificado com parada antecipada por métrica (BestBy).  
• Cálculo e registro de múltiplas métricas (F1, loss, precisão, recall, etc.).  
• Navegação pelos *folds* através de botões, *scrollbar* e salvar a figura
  atual (.png).  
• Validação de modelos já treinados, com geração da matriz de
  confusão correspondente.  
• Log completo de sessão, salvamento automático de figuras e *joblib* do
  modelo completo.

USO RÁPIDO NA CLI
-----------------
Treinar um novo modelo com todos os CSV em `.data/` (ou mover do diretório
atual caso não existam arquivos lá):

    python mlp.py

Treinar usando arquivos específicos:

    python mlp.py --csv base1.csv base2.csv

Validar o modelo mais recente:

    python mlp.py --validate

Validar um modelo indicado:

    python mlp.py --validate .models/mlp_20250101_120000.joblib --csv teste.csv

PASTAS PADRÃO
-------------
.data   CSV de entrada  
.img    Imagens geradas (confusão e painéis)  
.models Modelos salvos (`joblib`)  
.log    Arquivos de log

PARÂMETROS AJUSTÁVEIS
---------------------
Config (dados) e HyperParams (rede) estão definidos como *dataclasses*  
e podem ser alterados antes da execução:

    cfg.csv_sep           Separador de coluna (padrão ';')
    cfg.exclude_cols      Colunas a ignorar
    cfg.fill_strategy     Estratégia de imputação (mean, median, mode, etc.)
    cfg.normalize         Aplica `StandardScaler`
    
    hp.k_folds            Nº de *folds* de validação cruzada
    hp.epochs             Épocas máximas por *fold*
    hp.patience           Tolerância sem melhora para *early-stopping*
    hp.watch_for          Métrica monitorada (F1, loss, etc.)
    hp.layers             Lista de tuplas (ativação, unidades) incluindo
                          a saída (unidade = 1)

DEPENDÊNCIAS
------------
Python ≥3.10, scikit-learn ≥1.3, pandas, numpy, matplotlib, jb.

Execute `python mlp.py -h` para ajuda nas flags da CLI.
"""

from __future__ import annotations

import sys, copy, logging, platform, shutil, warnings, traceback

from typing             import Optional, Sequence
from enum               import Enum
from pathlib            import Path
from argparse           import ArgumentParser, Namespace
from dataclasses        import dataclass, asdict
from time               import perf_counter, strftime
from pprint             import pprint

import joblib               as jb
import matplotlib.pyplot    as plt
import matplotlib.patches   as mpatches
import matplotlib.gridspec  as mgs
import numpy                as np
import pandas               as pd

from matplotlib.widgets     import Button

from sklearn.experimental   import enable_iterative_imputer

from sklearn.impute         import IterativeImputer, SimpleImputer
from sklearn.metrics        import (
    classification_report,
    confusion_matrix,
    f1_score, log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network  import MLPClassifier
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# ENUMS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
class BestBy(Enum):
    VAL_LOSS        = (0, "min")
    FALSE_NEGATIVE  = (1, "min")
    FALSE_POSITIVE  = (2, "min")
    TRUE_POSITIVE   = (3, "max")
    TRUE_NEGATIVE   = (4, "max")
    DIFF_POSITIVE   = (5, "max")
    DIFF_NEGATIVE   = (6, "max")
    F1_SCORE        = (7, "max")
    ACCURACY        = (8, "max")
    PRECISION       = (9, "max")
    RECALL          = (10, "max")

class FillStrategy(Enum):
    MEAN=1; MEDIAN=2; MODE=3; ITER=4; KEEP=5; ZERO=6

class Activation(str, Enum):
    RELU="relu"; TANH="tanh"; LINEAR="identity"; SIGMOID="logistic"
    @classmethod
    def from_str(cls, name:str)->"Activation":
        try: return cls[name.upper()]
        except KeyError as exc: raise ValueError(f"Invalid activation: {name}") from exc

@dataclass(frozen=True, slots=True)
class HyperParams:
    k_folds:        int     = 6
    epochs:         int     = 512
    patience:       int     = 16
    watch_for:      BestBy  = BestBy.PRECISION
    min_delta:      float   = 1e-5
    batch_size:     int     = 64
    learning_rate:  float   = 5e-4
    
    layers: tuple[tuple[Activation,int],...] = (
        (Activation.TANH,7),
        (Activation.TANH,7),
        (Activation.SIGMOID,1)
    )

@dataclass(frozen=True, slots=True)
class Config:
    csv_dir:        Path    = Path(".data")
    log_dir:        Path    = Path(".log")
    img_dir:        Path    = Path(".img")
    mdl_dir:        Path    = Path(".models")
    csv_sep:        str     = ";"
    scramble_rows:  bool    = True
    normalize:      bool    = True
    null_value:     Optional[float] = 0
    fill_strategy:  FillStrategy = FillStrategy.MODE
    exclude_cols:   tuple[int,...] = (0,)

cfg, hp = Config(), HyperParams()

# ─────────────────────────────────────────────────────────────────────────────

for d in (cfg.csv_dir,cfg.img_dir,cfg.mdl_dir,cfg.log_dir): d.mkdir(exist_ok=True)

logging.basicConfig(
    filename=cfg.log_dir/f"{strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    encoding="utf-8"
)
for k,v in [("===== SESSION START =====",""),
            ("OS",platform.platform()),("Python",sys.version.replace("\n"," ")),
            ("Arch",platform.machine()),("Config",asdict(cfg)),
            ("HyperParams",asdict(hp))]: logging.info("%s: %s",k,v)

plt.rcParams.update({"font.family":"monospace","font.size":12})
warnings.filterwarnings("ignore",category=UserWarning)

def _collect_csv_files()->list[Path]:
    csv_files = sorted(list(cfg.csv_dir.glob("*.csv")))
    if csv_files: return csv_files
    root=list(Path().glob("*.csv"))
    if not root: sys.exit("Nenhum CSV encontrado.")
    for csv in root: shutil.move(csv, cfg.csv_dir/csv.name)
    sys.exit("CSV movidos para .data; execute novamente.")

def _validate_indices(idxs:Sequence[int], total:int, label:str)->None:
    if not all(-total<=i<total for i in idxs): sys.exit(f"{label} fora do intervalo.")

def load_data(csv_paths:Optional[Sequence[Path]]=None)->tuple[pd.DataFrame,np.ndarray]:
    try:
        files=list(csv_paths) if csv_paths else _collect_csv_files()
        frames=[pd.read_csv(f,sep=cfg.csv_sep,dtype=str) for f in files]
        data=pd.concat(frames,ignore_index=True)
    except Exception: sys.exit(f"Falha ao carregar CSV:\n{traceback.format_exc()}")

    if cfg.scramble_rows: data=data.sample(frac=1,ignore_index=True)
    data=data.apply(pd.to_numeric,errors="coerce")

    total_cols=data.shape[1]
    _validate_indices((-1,),total_cols,"Target")
    _validate_indices(cfg.exclude_cols,total_cols,"exclude_cols")

    y=data.iloc[:,list((-1,))].values.ravel()
    predictors=data.copy()
    drop=set(cfg.exclude_cols)|set((-1,))
    pred_cols=predictors.columns.difference(predictors.columns[list(drop)])

    if cfg.null_value is not None:
        mask=predictors[pred_cols]==cfg.null_value
        predictors.loc[mask.index,pred_cols]=predictors[pred_cols].where(~mask,np.nan)

    X=predictors[pred_cols]
    if X.empty: sys.exit("Sem colunas preditoras.")
    return X,y

def make_imputer()->IterativeImputer|SimpleImputer|str:
    return {
        FillStrategy.ITER:IterativeImputer(),
        FillStrategy.MEDIAN:SimpleImputer(strategy="median"),
        FillStrategy.MODE:SimpleImputer(strategy="most_frequent"),
        FillStrategy.KEEP:SimpleImputer(strategy="constant",fill_value=cfg.null_value),
        FillStrategy.ZERO:SimpleImputer(strategy="constant",fill_value=0),
    }.get(cfg.fill_strategy,SimpleImputer(strategy="mean"))

def make_classifier()->MLPClassifier:
    hidden=tuple(u for _,u in hp.layers if u>1)
    return MLPClassifier(
        hidden_layer_sizes=hidden,activation=hp.layers[0][0].value,
        solver="sgd",learning_rate_init=hp.learning_rate,
        learning_rate="adaptive",warm_start=True,
        batch_size=hp.batch_size,max_iter=1)

def _metric_value(name:BestBy,y_true,y_pred,prob=None)->float:
    cm=confusion_matrix(y_true,y_pred)
    tn,fp,fn,tp=cm.ravel(); s=tn+fp+fn+tp
    match name:
        case BestBy.F1_SCORE:        return f1_score(y_true,y_pred)
        case BestBy.VAL_LOSS:        return log_loss(y_true,prob)
        case BestBy.RECALL:          return recall_score(y_true,y_pred)
        case BestBy.PRECISION:       return precision_score(y_true,y_pred)
        case BestBy.TRUE_NEGATIVE:   return tn/s if s else 0
        case BestBy.TRUE_POSITIVE:   return tp/s if s else 0
        case BestBy.FALSE_NEGATIVE:  return fn/s if s else 0
        case BestBy.FALSE_POSITIVE:  return fp/s if s else 0
        case BestBy.DIFF_POSITIVE:   return (tp-fp)/s if s else 0
        case BestBy.DIFF_NEGATIVE:   return (tn-fn)/s if s else 0
        case BestBy.ACCURACY:        return (tp+tn)/s if s else 0
        case _:                      return 0.0

def _plot_confusion_matrix(cm:np.ndarray,ax:plt.Axes)->None:
    ax.imshow(cm,cmap="Greens",vmin=0,vmax=cm.max(),interpolation="nearest")
    for i in range(2):
        for j in range(2):
            ax.text(j,i,int(cm[i,j]),ha="center",va="center",fontsize=24)
    ax.set_xticks([0,1]),ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred 0","Pred 1"]); ax.set_yticklabels(["Real 0","Real 1"])
    ax.set_xlabel(""),ax.set_ylabel("")

def get_dataset_name()->str:
    if not cfg.csv_dir.is_dir(): return "Treinamento"
    csv_files=list(cfg.csv_dir.glob("*.csv"))
    if not csv_files: return "Treinamento"
    names=[f.stem for f in csv_files if f.is_file()]
    return "\n\t"+"\n\t".join(names) if len(names)>1 else names[0] if names else "Treinamento"

def _architecture_string(n_features:int)->str:
    return (
        f"Dataset:{get_dataset_name()}\n\n"
        f"Features:\t{n_features}\n\n"
        f"[Architecture]\n\nMax Epochs:\t\t{hp.epochs}\n"
        f"Best By ({hp.watch_for.value[1]}):\t{hp.watch_for.name}\n"
        f"Patience:\t\t{hp.patience}\n"
        f"Learning Rate:\t{hp.learning_rate}\n"
        f"Min. Delta:\t\t{hp.min_delta}\n"
        f"Layers:{"\n\t"+"\n\t".join(f"{u} {a.value}" for a,u in hp.layers)}"
    ).expandtabs(4)

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZAÇÃO INTERATIVA
# ─────────────────────────────────────────────────────────────────────────────
class FoldViewer:
    def __init__(
        self,cms:list[np.ndarray],hist:dict[str,list[float]],
        metric:str,direction:str,arch:str,dataset:str,
        models:list[Pipeline],X:pd.DataFrame,y:np.ndarray,
        init_idx:int,val_indices:list[np.ndarray]
    ):
        self.cms_current=cms; self.hist=hist; self.metric=metric
        self.dir=direction; self.arch=arch; self.dataset_name=dataset
        self.models=models; self.val_indices=val_indices
        self.X_base,self.y_base=X,y; self.X,self.y=X,y
        self.n=len(cms); self.idx=0; self.saved=False
        self.scroll=0
        self.max_visible=10

        self._make_figure(); self._draw()
        self.fig.canvas.mpl_connect("close_event",self._on_close)
        plt.show()

    def _make_figure(self):
        self.fig=plt.figure(figsize=(10,6),dpi=100)
        gs_main=self.fig.add_gridspec(1,2,width_ratios=[4,1],wspace=0.05)

        gs_left=mgs.GridSpecFromSubplotSpec(
            2,2,subplot_spec=gs_main[0,0],hspace=0.25,wspace=0.25
        )
        self.ax_menu=self.fig.add_subplot(gs_left[0,0]); self._blank(self.ax_menu)
        self.ax_cm  =self.fig.add_subplot(gs_left[0,1])
        self.ax_arch=self.fig.add_subplot(gs_left[1,0]); self._blank(self.ax_arch)
        self.ax_txt =self.fig.add_subplot(gs_left[1,1]); self._blank(self.ax_txt)

        self.ax_flds=self.fig.add_subplot(gs_main[0,1]); self._blank(self.ax_flds)

        self.fig.subplots_adjust(top=0.92,bottom=0.06)

        self.menu_btns=[]
        menu_items=[("Carregar CSV",self.load_new_csv),
                    ("Restaurar CSV",self.restore_csv),
                    ("Salvar Gráfico",self.save_current)]
        total=len(menu_items); gap=0.05; btn_h=0.20
        start_y=0.5+((total-1)/2)*(btn_h+gap)
        for i,(lbl,cb) in enumerate(menu_items):
            y=start_y - i*(btn_h+gap)
            ax=self.fig.add_axes(self._rel(self.ax_menu,0.2,y,0.8,btn_h))
            btn=Button(ax,lbl); btn.on_clicked(cb)
            self.menu_btns.append(btn)

        self.ax_arch.text(0.20,1.40,self.arch,va="top",ha="left", wrap=True)
        self._create_fold_buttons()

    def _blank(self,ax):
        ax.axis("off")
        for s in ax.spines.values(): s.set_visible(False)

    def _rel(self,ax:plt.Axes,x:float,y:float,w:float,h:float):
        bb=ax.get_position()
        return [bb.x0+x*bb.width, bb.y0+y*bb.height, w*bb.width, h*bb.height]

    def _create_fold_buttons(self):
        for btn in getattr(self,"fold_btn_objs",[]): btn.ax.remove()
        self.fold_btn_objs=[]
        if hasattr(self,"_scroll_patch"): self._scroll_patch.remove()
        if hasattr(self,"btn_up"):   self.btn_up.ax.remove()
        if hasattr(self,"btn_down"): self.btn_down.ax.remove()

        btn_h=0.06; gap=0.015
        up_ax=self.fig.add_axes(self._rel(self.ax_flds,0.2,0.93,0.6,btn_h))
        dn_ax=self.fig.add_axes(self._rel(self.ax_flds,0.2,0.01,0.6,btn_h))
        self.btn_up  =Button(up_ax,"↑"); self.btn_up.on_clicked(self._scroll_up)
        self.btn_down=Button(dn_ax,"↓"); self.btn_down.on_clicked(self._scroll_down)

        vis=list(range(self.scroll,min(self.scroll+self.max_visible,self.n)))
        area_h=0.92-2*(btn_h+gap)
        slot_h=area_h/self.max_visible
        for row,i in enumerate(vis):
            y= 0.96 - (btn_h+gap) - (row+1)*slot_h + slot_h*0.2
            label="Total" if i==0 else f"Fold {i}"
            ax=self.fig.add_axes(self._rel(self.ax_flds,0.05,y,0.9,slot_h*0.7))
            btn=Button(ax,label); btn.on_clicked(self._make_select_cb(i))
            self.fold_btn_objs.append(btn)

        self._draw_scrollbar()
        self._highlight_selected()

    def _draw_scrollbar(self):
        track_x0,track_y0=0.98,0.02
        track_h=0.9
        self._scroll_patch=self.ax_flds.add_patch(
            mpatches.Rectangle((track_x0,track_y0),0.04,track_h,
                               transform=self.ax_flds.transAxes,
                               facecolor="#f0f0f0",edgecolor="k",linewidth=0.4))
        handle_h=max(track_h*(self.max_visible/self.n),0.05)
        denom=max(self.n-self.max_visible,1)
        handle_y=track_y0 + (track_h-handle_h)*(self.scroll/denom)
        self.ax_flds.add_patch(
            mpatches.Rectangle((track_x0,handle_y),0.04,handle_h,
                               transform=self.ax_flds.transAxes,
                               facecolor="#777",edgecolor="k",linewidth=0.4))

    def _scroll_up(self,_):   self._scroll_generic(-3)
    def _scroll_down(self,_): self._scroll_generic(3)
    def _scroll_generic(self,delta:int):
        self.scroll=max(0,min(self.scroll+delta,self.n-self.max_visible))
        self._create_fold_buttons()

    def _make_select_cb(self,idx:int):
        def _cb(_): self.idx=idx; self._highlight_selected(); self._draw()
        return _cb

    def _highlight_selected(self):
        for btn in self.fold_btn_objs:
            lbl=btn.label.get_text()
            i=0 if lbl=="Total" else int(lbl.split()[-1])
            btn.ax.set_facecolor("lightblue" if i==self.idx else "lightgrey")
        self.fig.canvas.draw_idle()

    def _recompute_metrics(self):
        try:
            for i,model in enumerate(self.models):
                if i==0:
                    X_eval,y_eval=self.X,self.y
                else:
                    idxs=[j for j in self.val_indices[i-1] if j<len(self.X)]
                    X_eval,y_eval=(self.X.iloc[idxs],self.y[idxs]) if idxs else (self.X,self.y)
                pred=model.predict(X_eval); prob=model.predict_proba(X_eval)
                cm=confusion_matrix(y_eval,pred); self.cms_current[i]=cm
                self.hist["F1_SCORE"][i]=f1_score(y_eval,pred)
                self.hist["PRECISION"][i]=precision_score(y_eval,pred)
                self.hist["RECALL"][i]=recall_score(y_eval,pred)
                self.hist["VAL_LOSS"][i]=log_loss(y_eval,prob)
                tn,fp,fn,tp=cm.ravel(); s=tn+fp+fn+tp or 1
                self.hist["TRUE_NEGATIVE"][i]=tn/s; self.hist["FALSE_POSITIVE"][i]=fp/s
                self.hist["FALSE_NEGATIVE"][i]=fn/s; self.hist["TRUE_POSITIVE"][i]=tp/s
                self.hist["DIFF_POSITIVE"][i]=(tp-fp)/s; self.hist["DIFF_NEGATIVE"][i]=(tn-fn)/s
                self.hist["ACCURACY"][i]=(tp+tn)/s
        except Exception as exc: logging.exception("Recompute metrics failed: %s",exc)

    def load_new_csv(self,_):
        import tkinter as tk; from tkinter import filedialog
        root=tk.Tk(); root.withdraw()
        file_path=filedialog.askopenfilename(title="Selecionar CSV",filetypes=[("CSV","*.csv")])
        root.destroy();  logging.info("CSV selecionado: %s",file_path)
        if not file_path: return
        self.X,self.y=load_data([Path(file_path)]); self.dataset_name=Path(file_path).name
        self._recompute_metrics(); self.idx=0; self.scroll=0
        self._create_fold_buttons(); self._draw()

    def restore_csv(self,_):
        self.X,self.y=self.X_base,self.y_base; self.dataset_name="Treinamento"
        self._recompute_metrics(); self.idx=0; self.scroll=0
        self._create_fold_buttons(); self._draw()

    def save_current(self,_):
        out=cfg.img_dir/f"{self.dataset_name.replace('.csv','')}_{"Modelo_Completo" if self.idx==0 else f"Fold_{self.idx}"}_{strftime('%Y%m%d_%H%M%S')}.png"
        try:
            self.fig.savefig(out,dpi=100,bbox_inches="tight")
            print("Figura salva em",out); logging.info("Figura %s salva",out.name)
            self.saved=True
        except Exception as exc: print(f"Falha ao salvar figura: {exc}")

    # ── desenho principal ──────────────────────────────────────────────────
    def _draw(self):
        self.ax_cm.clear(); self.ax_txt.clear(); self.ax_txt.axis("off")
        _plot_confusion_matrix(self.cms_current[self.idx],self.ax_cm)
        f1=self.hist["F1_SCORE"][self.idx]; prec=self.hist["PRECISION"][self.idx]
        rec=self.hist["RECALL"][self.idx]; vls=self.hist["VAL_LOSS"][self.idx]
        acc=self.hist["ACCURACY"][self.idx]
        txt=(f"F1-Score:\t{f1:.3f}\n\nPrecision:\t{prec:.3f}\n\n"
             f"Recall:\t\t{rec:.3f}\n\nAccuracy:\t{acc:.3f}\n\n"
             f"Val Loss:\t{vls:.3f}").expandtabs(4)
        self.ax_txt.text(0.2,0.95,txt,va="top",ha="left",wrap=True)
        self.fig.suptitle(f"{'Modelo Completo' if self.idx==0 else f'Fold {self.idx}'}",
                          fontsize=16,weight="bold", ha="center", va="top", y=1)
        self.fig.canvas.draw_idle()

    def _on_close(self,_event):
        if not self.saved:
            out=cfg.img_dir/f"{self.dataset_name.replace('.csv','')}_{'Modelo Completo' if self.idx==0 else f'Fold_{self.idx}'}_{strftime('%Y%m%d_%H%M%S')}.png"
            try:
                self.fig.savefig(out,dpi=100,bbox_inches="tight")
                logging.info("Auto-save figura %s",out.name)
            except Exception: logging.exception("Auto-save falhou")

# ─────────────────────────────────────────────────────────────────────────────
# TREINAMENTO
# ─────────────────────────────────────────────────────────────────────────────
def train_single_fold(Xtr,ytr,Xv,yv)->tuple[Pipeline,dict[str,float],np.ndarray,int]:
    imputer=make_imputer(); steps=[] if imputer=="drop" else [("imputer",copy.deepcopy(imputer))]
    if cfg.normalize: steps.append(("scaler",StandardScaler()))
    clf=make_classifier(); steps.append(("mlp",clf)); pipe=Pipeline(steps)
    if imputer=="drop":
        Xtr,ytr=Xtr.dropna(),ytr[Xtr.index]; Xv,yv=Xv.dropna(),yv[Xv.index]

    best_pipe,best_score=None,(np.inf if hp.watch_for.value[1]=="min" else -np.inf)
    stagnant=0; trained_ep=0; classes=np.unique(ytr)

    for epoch in range(1,hp.epochs+1):
        trained_ep=epoch
        try: pipe.named_steps["mlp"].partial_fit(Xtr,ytr,classes)
        except Exception: pipe.fit(Xtr,ytr)
        pred=pipe.predict(Xv); prob=pipe.predict_proba(Xv)
        score=_metric_value(hp.watch_for,yv,pred,prob)
        improve=(score+hp.min_delta<best_score if hp.watch_for.value[1]=="min"
                 else score-hp.min_delta>best_score)
        if improve: best_score, best_pipe, stagnant=score,copy.deepcopy(pipe),0
        else:
            stagnant+=1
            if stagnant>=hp.patience: break

    pipe=best_pipe if best_pipe else pipe
    pred,prob=pipe.predict(Xv),pipe.predict_proba(Xv); cm=confusion_matrix(yv,pred)
    metrics={"F1_SCORE":f1_score(yv,pred),"PRECISION":precision_score(yv,pred),
             "RECALL":recall_score(yv,pred),"VAL_LOSS":log_loss(yv,prob)}
    tn,fp,fn,tp=cm.ravel(); s=tn+fp+fn+tp or 1
    metrics.update({"TRUE_NEGATIVE":tn/s,"FALSE_POSITIVE":fp/s,
                    "FALSE_NEGATIVE":fn/s,"TRUE_POSITIVE":tp/s,
                    "DIFF_POSITIVE":(tp-fp)/s,"DIFF_NEGATIVE":(tn-fn)/s,
                    "ACCURACY":(tp+tn)/s})
    return pipe,metrics,cm,trained_ep

def fold_scores(hist:dict[str,list[float]])->tuple[list[float],int]:
    n=len(next(iter(hist.values()))); scores=[0.0]*n
    for rank,m in enumerate(BestBy):
        if m.name not in hist: continue
        vals=hist[m.name]; lo,hi=np.nanmin(vals),np.nanmax(vals)
        for i,v in enumerate(vals):
            norm=0.5 if hi==lo else (v-lo)/(hi-lo); norm=1-norm if m.value=="min" else norm
            scores[i]+=(len(BestBy)-rank)*norm
    best=int(max(range(n),key=scores.__getitem__)); return scores,best

def train(csv_paths:Optional[Sequence[str]])->Path:
    paths=[Path(p) for p in csv_paths] if csv_paths else None
    X,y=load_data(paths); print(f"Conjunto de treino: {len(X)} amostras, {X.shape[1]} features, alvo: {hp.watch_for.name} ({hp.watch_for.value[1]})")
    skf=StratifiedKFold(hp.k_folds,shuffle=True)
    hist={m:[np.nan]*hp.k_folds for m in [m for m in BestBy.__members__.keys()]}
    cms,models,epochs_per_fold,val_indices=[],[],[],[]

    for fold,(tr_idx,val_idx) in enumerate(skf.split(X,y),start=1):
        Xtr,ytr=X.iloc[tr_idx],y[tr_idx]; Xv,yv=X.iloc[val_idx],y[val_idx]
        model,metrics,cm,n_epochs=train_single_fold(Xtr,ytr,Xv,yv)
        models.append(model); cms.append(cm); epochs_per_fold.append(n_epochs); val_indices.append(val_idx)
        for k,v in metrics.items(): hist[k][fold-1]=v
        print(f"\tFold {fold}:\tEpochs={n_epochs:<4}\tF1={metrics['F1_SCORE']:.3f}"
              f"\tAcc={metrics['ACCURACY']:.3f}\tLoss={metrics['VAL_LOSS']:.3f}")

    imputer=make_imputer(); steps=[] if imputer=="drop" else [("imputer",imputer)]
    if cfg.normalize: steps.append(("scaler",StandardScaler()))
    steps.append(("mlp",MLPClassifier(hidden_layer_sizes=tuple(u for _,u in hp.layers if u>1),
        activation=hp.layers[0][0].value,solver="adam",
        learning_rate_init=hp.learning_rate,max_iter=hp.epochs,early_stopping=True)))
    pipe_full=Pipeline(steps); pipe_full.fit(X,y)
    pred_full,prob_full=pipe_full.predict(X),pipe_full.predict_proba(X)
    cm_full=confusion_matrix(y,pred_full)

    def _metric_dict(y_true,y_pred,prob,cm):
        tn,fp,fn,tp=cm.ravel(); s=tn+fp+fn+tp or 1
        return {"F1_SCORE":f1_score(y_true,y_pred),"PRECISION":precision_score(y_true,y_pred),
                "RECALL":recall_score(y_true,y_pred),"VAL_LOSS":log_loss(y_true,prob),
                "TRUE_NEGATIVE":tn/s,"FALSE_POSITIVE":fp/s,"FALSE_NEGATIVE":fn/s,"TRUE_POSITIVE":tp/s,
                "DIFF_POSITIVE":(tp-fp)/s,"DIFF_NEGATIVE":(tn-fn)/s,"ACCURACY":(tp+tn)/s}
    metrics_full=_metric_dict(y,pred_full,prob_full,cm_full)

    for k in hist.keys(): hist[k]=[metrics_full[k]]+hist[k]
    cms.insert(0,cm_full); models.insert(0,pipe_full)

    mean_f1=np.nanmean(hist["F1_SCORE"][1:]); print(f"F1 médio (k-folds): {mean_f1:.3f}")
    scores,best_idx=fold_scores(hist); logging.info("Fold scores: %s; best=%d",scores,best_idx)
    pprint(f"Métricas do modelo completo: {metrics_full.items()}")
    pprint(f"Matriz de confusão do modelo completo:\n{cm_full}")

    arch=_architecture_string(X.shape[1])
    FoldViewer(cms,hist,hp.watch_for.name,hp.watch_for.value[1],arch,
               "Treinamento",models,X,y,best_idx,val_indices)

    model_path=cfg.mdl_dir/f"mlp_{strftime('%Y%m%d_%H%M%S')}.joblib"
    jb.dump(pipe_full,model_path); print("Modelo salvo em",model_path)
    return model_path

def latest_model()->Path:
    models=sorted(cfg.mdl_dir.glob("mlp_*.joblib"))
    if not models: sys.exit("Nenhum modelo em .models.")
    return models[-1]

def validate(model_path:Path,csv_paths:Optional[Sequence[str]])->None:
    paths=[Path(p) for p in csv_paths] if csv_paths else None
    X,y=load_data(paths); pipe:Pipeline=jb.load(model_path)
    pred=pipe.predict(X); print(classification_report(y,pred,digits=3))
    cm=confusion_matrix(y,pred)

    plt.figure(figsize=(4,4)); _plot_confusion_matrix(cm,plt.gca()); plt.tight_layout()
    out=cfg.img_dir/f"{model_path.stem}.png"; plt.savefig(out,dpi=100,bbox_inches="tight")
    plt.show(block=True); plt.close(); print("Gráfico salvo em",out); logging.info("Figura %s salva",out.name)

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args()->Namespace:
    p=ArgumentParser(description="MLP trainer / validator.")
    p.add_argument("--validate",nargs="?",const="",metavar="MODEL",
                   help="Valida o modelo informado (ou o mais recente se omitido).")
    p.add_argument("--csv",nargs="+",metavar="CSV",help="Arquivos CSV específicos.")
    return p.parse_args()

def main()->None:
    start=perf_counter(); args=parse_args()
    if args.validate is not None:
        mdl=Path(args.validate) if args.validate else latest_model(); validate(mdl,args.csv)
    else: train(args.csv)
    logging.info("Tempo total %.1fs",perf_counter()-start); logging.info("===== SESSION END =====")

if __name__=="__main__": main()