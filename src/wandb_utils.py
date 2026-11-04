"""Helpers used by the notebook to pull results back from W&B (curves, images, t-SNE, tables).

Every figure shown in the notebook's results sections comes from what train.py logged to W&B,
or is built from W&B data and logged back to W&B (analysis run) before being displayed.

PRELIMINARY: written before the runs exist; check keys/paths against the first finished runs.
"""
import json
import re
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import wandb
from plotly.subplots import make_subplots

MODELS = ["vae", "u-net"]
LOSSES = ["l1", "l2", "ssim", "ssim_l1"]
LOSS_COLORS = {"l1": "#1f77b4", "l2": "#ff7f0e", "ssim": "#2ca02c", "ssim_l1": "#d62728"}


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
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = {}
    for run in api.runs(path, filters={"state": state}, order="-created_at"):
        if run.job_type == "analysis":  # runs created by the notebook itself
            continue
        key = (*_model_loss(run), run_trial(run))
        if None not in key and key not in runs:  # newest first: keep the latest of each config
            runs[key] = run
    return runs


def get_runs(project, entity=None, state="finished", trial="base"):
    """Latest finished training run for each (model, loss) of one trial -> {(model, loss): run}."""
    return {(m, l): run for (m, l, t), run in get_all_runs(project, entity, state).items() if t == trial}


def trials_table(runs, keys):
    """List of dicts (one per (model, loss, trial) run) with the final value of each summary key."""
    rows = []
    for (model, loss, trial), run in sorted(runs.items()):
        row = {"model": model, "loss": loss, "trial": trial, "run": run.name}
        for key in keys:
            row[key] = run.summary.get(key)
        rows.append(row)
    return rows


# ---------------------------------------------------------------- scalars

def metric_history(run, key):
    """(epochs, values) of an epoch-level metric such as 'train/loss' or 'val/loss'."""
    rows = [r for r in run.scan_history(keys=["epoch", key]) if r.get(key) is not None]
    rows.sort(key=lambda r: r["epoch"])
    return [r["epoch"] for r in rows], [r[key] for r in rows]


def loss_curves_figure(runs, model, keys=("train/loss", "val/loss")):
    """One subplot per key, one line per loss function, for a given model."""
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
        fig.update_xaxes(title_text="epoch", row=1, col=col)
    fig.update_layout(title=f"{model}: {' / '.join(keys)} por función de pérdida", template="plotly_white", height=400)
    return fig


def summary_table(runs, keys):
    """List of dicts (one per run) with the final value of each summary key."""
    rows = []
    for (model, loss), run in sorted(runs.items()):
        row = {"model": model, "loss": loss, "run": run.name}
        for key in keys:
            row[key] = run.summary.get(key)
        rows.append(row)
    return rows


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
    """[(epoch, local_png_path)] of an image key such as 'val/reconstructions' (epoch read from the caption)."""
    out = []
    for step, value in media_history(run, key):
        captions = value.get("captions") or [""]
        match = re.search(r"Epoch (\d+)", captions[0])
        epoch = int(match.group(1)) if match else step
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


# ---------------------------------------------------------------- anomaly-score plots

def error_histograms(rows, category, score="score_l1", title_prefix=""):
    """Histogram of the reconstruction error: good vs each defect type of one category."""
    cat_rows = [r for r in rows if r["category"] == category]
    good = [r[score] for r in cat_rows if r["defect"] == "good"]
    defects = sorted({r["defect"] for r in cat_rows} - {"good"})
    cols = min(3, len(defects))
    n_rows = -(-len(defects) // cols)
    fig = make_subplots(rows=n_rows, cols=cols, subplot_titles=defects)
    for i, defect in enumerate(defects):
        r, c = i // cols + 1, i % cols + 1
        bad = [x[score] for x in cat_rows if x["defect"] == defect]
        fig.add_trace(go.Histogram(x=good, name="good", marker_color="#2ca02c", opacity=0.6,
                                   legendgroup="good", showlegend=i == 0), row=r, col=c)
        fig.add_trace(go.Histogram(x=bad, name="defective", marker_color="#d62728", opacity=0.6,
                                   legendgroup="defective", showlegend=i == 0), row=r, col=c)
    fig.update_layout(barmode="overlay", template="plotly_white", height=300 * n_rows,
                      title=f"{title_prefix}{category}: error de reconstrucción ({score}) good vs defecto")
    return fig


def good_vs_defective_means(rows, score="score_l1"):
    """Mean error of good and defective test images, per category and overall."""
    out = {}
    for category in sorted({r["category"] for r in rows}) + ["all"]:
        sel = rows if category == "all" else [r for r in rows if r["category"] == category]
        good = [r[score] for r in sel if r["label"] == 0]
        bad = [r[score] for r in sel if r["label"] == 1]
        out[category] = (sum(good) / len(good), sum(bad) / len(bad))
    return out
