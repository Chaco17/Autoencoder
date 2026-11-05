"""Helpers used by the notebook to pull results back from W&B (curves, images, t-SNE, tables).

Every figure shown in the notebook's results sections comes from what train.py logged to W&B,
or is built from W&B data and logged back to W&B (analysis run) before being displayed.
"""
import base64
import json
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import wandb
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score

MODELS = ["vae", "u-net"]
LOSSES = ["l1", "l2", "ssim", "ssim_l1"]
LOSS_COLORS = {"l1": "#1f77b4", "l2": "#ff7f0e", "ssim": "#2ca02c", "ssim_l1": "#d62728"}
SCORE_NAMES = {
    "score_l1": "error L1 medio",
    "score_mse": "error cuadrático medio",
    "score_ssim": "1 - SSIM",
    "score_max": "máximo del mapa de error suavizado",
}


# ---------------------------------------------------------------- runs

def _model_loss(run):
    """(model, loss) of a training run, from its config (fallback: its tags)."""
    cfg = run.config
    model = (cfg.get("model") or {}).get("name") if isinstance(cfg.get("model"), dict) else cfg.get("model/name")
    loss = (cfg.get("loss") or {}).get("name") if isinstance(cfg.get("loss"), dict) else cfg.get("loss/name")
    if model is None or loss is None:
        tags = set(run.tags)
        model = next((m for m in MODELS if m in tags), model)
        loss = next((l for l in LOSSES if l in tags), loss)
    return model, loss


def run_trial(run):
    """Trial (set of hyperparameters, conf/trial/) of a training run; runs from before trials existed are 'base'."""
    trial = run.config.get("trial")
    return trial.get("name", "base") if isinstance(trial, dict) else run.config.get("trial/name", "base")


def get_all_runs(project, entity=None, state="finished"):
    """Latest finished training run for each (model, loss, trial) -> {(model, loss, trial): run}."""
    api = wandb.Api(timeout=60)
    path = f"{entity}/{project}" if entity else project
    runs = {}
    # lazy=False: config and summary come with each page of runs (lazy runs make one extra request per run)
    for run in api.runs(path, filters={"state": state}, order="-created_at", lazy=False):
        if run.job_type == "analysis":  # runs created by the notebook itself
            continue
        key = (*_model_loss(run), run_trial(run))
        if None not in key and key not in runs:  # newest first: keep the latest of each config
            runs[key] = run
    return runs


def get_runs(project, entity=None, state="finished", trial="base", all_runs=None):
    """Latest finished training run for each (model, loss) of one trial -> {(model, loss): run}.

    all_runs: result of get_all_runs() when it is already loaded (avoids listing the project again).
    """
    all_runs = get_all_runs(project, entity, state) if all_runs is None else all_runs
    return {(m, l): run for (m, l, t), run in all_runs.items() if t == trial}


def trials_table(runs, keys):
    """List of dicts (one per (model, loss, trial) run) with the final value of each summary key."""
    rows = []
    for (model, loss, trial), run in sorted(runs.items()):
        row = {"model": model, "loss": loss, "trial": trial, "run": run.name}
        for key in keys:
            row[key] = run.summary.get(key)
        rows.append(row)
    return rows


def summary_table(runs, keys):
    """List of dicts (one per (model, loss) run) with the final value of each summary key."""
    rows = []
    for (model, loss), run in sorted(runs.items(), key=lambda kv: (MODELS.index(kv[0][0]), LOSSES.index(kv[0][1]))):
        row = {"model": model, "loss": loss, "run": run.name}
        for key in keys:
            row[key] = run.summary.get(key)
        rows.append(row)
    return rows


# ---------------------------------------------------------------- curves

def metric_history(run, key):
    """(epochs, values) of an epoch-level metric such as 'train/loss' or 'val/loss' (epochs start at 1)."""
    rows = [r for r in run.scan_history(keys=["epoch", key]) if r.get(key) is not None]
    rows.sort(key=lambda r: r["epoch"])
    return [r["epoch"] + 1 for r in rows], [r[key] for r in rows]


def train_val_figure(runs, model, log_y=True):
    """One subplot per loss function with its train/loss (solid) and val/loss (dashed) curves.

    Each subplot has its own y axis: every loss has its own scale, so only train vs val of the
    same run (convergence and overfitting) is compared here.
    """
    fig = make_subplots(rows=1, cols=len(LOSSES), subplot_titles=LOSSES, horizontal_spacing=0.06)
    for col, loss in enumerate(LOSSES, start=1):
        run = runs.get((model, loss))
        if run is None:
            continue
        for key, dash in (("train/loss", "solid"), ("val/loss", "dash")):
            epochs, values = metric_history(run, key)
            fig.add_trace(
                go.Scatter(x=epochs, y=values, mode="lines", name=key, legendgroup=key, showlegend=col == 1,
                           line=dict(color=LOSS_COLORS[loss], dash=dash)),
                row=1, col=col,
            )
        fig.update_xaxes(title_text="época", row=1, col=col)
        if log_y:
            fig.update_yaxes(type="log", row=1, col=col)
    fig.update_layout(title=f"{model}: pérdida de entrenamiento (continua) y validación (discontinua)",
                      template="plotly_white", height=380)
    return fig


def loss_curves_figure(runs, model, keys=("val/l1", "val/mse", "val/ssim")):
    """One subplot per key, one line per loss function, for a given model (metrics on a common scale)."""
    fig = make_subplots(rows=1, cols=len(keys), subplot_titles=list(keys))
    for col, key in enumerate(keys, start=1):
        for loss in LOSSES:
            run = runs.get((model, loss))
            if run is None:
                continue
            epochs, values = metric_history(run, key)
            fig.add_trace(
                go.Scatter(x=epochs, y=values, mode="lines", name=loss, legendgroup=loss,
                           showlegend=col == 1, line=dict(color=LOSS_COLORS[loss])),
                row=1, col=col,
            )
        fig.update_xaxes(title_text="época", row=1, col=col)
    fig.update_layout(title=f"{model}: {' / '.join(keys)} por función de pérdida", template="plotly_white", height=380)
    return fig


# ---------------------------------------------------------------- media

def _media_files(value):
    """File paths inside a logged media value (image list, plotly or table)."""
    if not isinstance(value, dict):
        return []
    if "filenames" in value:  # list of images (WandbLogger.log_image)
        return list(value["filenames"])
    return [value["path"]] if "path" in value else []


def download(run, file_path, root="wandb_downloads"):
    """Download one run file (cached) and return its local path."""
    local = Path(root) / run.id / file_path
    if not local.exists():
        run.file(file_path).download(root=str(Path(root) / run.id), replace=True)
    return local


def media_history(run, key):
    """[(step, media_value)] for every time `key` was logged, ordered by step."""
    rows = [(r["_step"], r[key]) for r in run.scan_history(keys=["_step", key]) if r.get(key) is not None]
    return sorted(rows, key=lambda r: r[0])


def image_history(run, key, root="wandb_downloads"):
    """[(epoch, local_png_path)] of an image key such as 'val/reconstructions' (epoch from the caption, from 1)."""
    out = []
    for step, value in media_history(run, key):
        captions = value.get("captions") or [""]
        match = re.search(r"Epoch (\d+)", captions[0])
        epoch = int(match.group(1)) + 1 if match else step
        for path in _media_files(value):
            out.append((epoch, download(run, path, root)))
    return out


def plotly_history(run, key, root="wandb_downloads"):
    """[(step, plotly Figure)] of a plotly key such as 'val/tsne'."""
    out = []
    for step, value in media_history(run, key):
        for path in _media_files(value):
            out.append((step, pio.read_json(download(run, path, root))))
    return out


def first_image(run, key, root="wandb_downloads"):
    """Local path of the first image logged under `key` (downloads only that one)."""
    history = media_history(run, key)
    files = _media_files(history[0][1]) if history else []
    return download(run, files[0], root) if files else None


def last_image(run, key, root="wandb_downloads"):
    """Local path of the last image logged under `key` (from the run summary)."""
    files = _media_files(run.summary.get(key))
    return download(run, files[-1], root) if files else None


def last_plotly(run, key, root="wandb_downloads"):
    files = _media_files(run.summary.get(key))
    return pio.read_json(download(run, files[-1], root)) if files else None


def load_table(run, key="test/scores", root="wandb_downloads"):
    """A logged wandb.Table as a list of dicts."""
    files = _media_files(run.summary.get(key))
    if not files:
        raise KeyError(f"{run.name}: no table '{key}' in summary")
    data = json.loads(download(run, files[-1], root).read_text(encoding="utf-8"))
    return [dict(zip(data["columns"], row)) for row in data["data"]]


# ---------------------------------------------------------------- t-SNE

def _array(value):
    """Plotly >= 6 stores arrays in JSON as {'dtype', 'bdata'} (base64); returns a numpy array."""
    if isinstance(value, dict) and "bdata" in value:
        return np.frombuffer(base64.b64decode(value["bdata"]), dtype=np.dtype(value["dtype"]))
    return np.asarray(value)


def tsne_grid(figures, titles, cols=4, title="", height_per_row=330):
    """Several logged t-SNE figures as subplots of a single figure (same colors/symbols, one shared legend)."""
    n_rows = -(-len(figures) // cols)
    grid = make_subplots(rows=n_rows, cols=cols, subplot_titles=titles, horizontal_spacing=0.03, vertical_spacing=0.08)
    shown = set()
    for i, fig in enumerate(figures):
        r, c = i // cols + 1, i % cols + 1
        for trace in fig.data:
            grid.add_trace(
                go.Scatter(x=_array(trace.x), y=_array(trace.y), mode="markers", name=trace.name,
                           legendgroup=trace.name, showlegend=trace.name not in shown,
                           marker=dict(size=4, opacity=0.75, color=trace.marker.color, symbol=trace.marker.symbol)),
                row=r, col=c,
            )
            shown.add(trace.name)
    grid.update_xaxes(showticklabels=False)
    grid.update_yaxes(showticklabels=False)
    grid.update_layout(title=title, template="plotly_white", height=height_per_row * n_rows)
    return grid


def tsne_epoch(fig):
    """Epoch (from 1) written in the title of a validation t-SNE figure."""
    match = re.search(r"epoch (\d+)", fig.layout.title.text or "")
    return int(match.group(1)) + 1 if match else None


# ---------------------------------------------------------------- anomaly scores

def _auroc(good, bad):
    return roc_auc_score([0] * len(good) + [1] * len(bad), list(good) + list(bad))


def defect_table(rows, score="score_ssim"):
    """One dict per (category, defect) of the test set: number of images, mean error and AUROC vs the good images."""
    out = []
    for category in sorted({r["category"] for r in rows}):
        cat_rows = [r for r in rows if r["category"] == category]
        good = [r[score] for r in cat_rows if r["label"] == 0]
        for defect in sorted({r["defect"] for r in cat_rows} - {"good"}):
            bad = [r[score] for r in cat_rows if r["defect"] == defect]
            out.append({"category": category, "defect": defect, "n": len(bad),
                        "error good": float(np.mean(good)), "error defect": float(np.mean(bad)),
                        "razón defect/good": float(np.mean(bad) / np.mean(good)), "auroc": _auroc(good, bad)})
    return out


def error_histograms(rows, category, score="score_ssim", title_prefix=""):
    """Histograms of the reconstruction error: good vs each defect type of one category.

    Normalized as probability (there are ~3 times more good images than images of each defect) and with
    the same bins in both, so the overlap reads directly; the AUROC of each defect is in its subplot title.
    """
    cat_rows = [r for r in rows if r["category"] == category]
    good = [r[score] for r in cat_rows if r["label"] == 0]
    defects = sorted({r["defect"] for r in cat_rows} - {"good"})
    values = [r[score] for r in cat_rows]
    bins = dict(start=min(values), end=max(values), size=(max(values) - min(values)) / 25 or 1e-6)
    titles = [f"{d} (n={sum(r['defect'] == d for r in cat_rows)}, AUROC {_auroc(good, [r[score] for r in cat_rows if r['defect'] == d]):.2f})"
              for d in defects]
    cols = min(3, len(defects))
    n_rows = -(-len(defects) // cols)
    fig = make_subplots(rows=n_rows, cols=cols, subplot_titles=titles, vertical_spacing=0.12)
    for i, defect in enumerate(defects):
        r, c = i // cols + 1, i % cols + 1
        bad = [x[score] for x in cat_rows if x["defect"] == defect]
        fig.add_trace(go.Histogram(x=good, name=f"good (n={len(good)})", marker_color="#2ca02c", opacity=0.6,
                                   xbins=bins, histnorm="probability", legendgroup="good", showlegend=i == 0), row=r, col=c)
        fig.add_trace(go.Histogram(x=bad, name="con defecto", marker_color="#d62728", opacity=0.6,
                                   xbins=bins, histnorm="probability", legendgroup="defective", showlegend=i == 0),
                      row=r, col=c)
    fig.update_xaxes(title_text=SCORE_NAMES.get(score, score), row=n_rows)
    fig.update_layout(barmode="overlay", template="plotly_white", height=260 * n_rows + 80,
                      title=f"{title_prefix}{category}: error de reconstrucción ({SCORE_NAMES.get(score, score)}), good vs cada defecto")
    return fig


def good_vs_defective_means(rows, score="score_ssim"):
    """Mean error of good and defective test images, per category and overall."""
    out = {}
    for category in sorted({r["category"] for r in rows}) + ["all"]:
        sel = rows if category == "all" else [r for r in rows if r["category"] == category]
        good = [r[score] for r in sel if r["label"] == 0]
        bad = [r[score] for r in sel if r["label"] == 1]
        out[category] = (sum(good) / len(good), sum(bad) / len(bad))
    return out
