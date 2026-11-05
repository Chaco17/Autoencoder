"""Helpers used by the notebook to pull results back from W&B (metrics, images, t-SNE, tables).

Nothing is logged back to W&B: the images and t-SNE figures are shown exactly as train.py logged
them, and the metrics / test/scores table are only turned into tables.
"""
import json
import re
from pathlib import Path

import numpy as np
import plotly.io as pio
import wandb
from sklearn.metrics import roc_auc_score

MODELS = ["vae", "u-net"]
LOSSES = ["l1", "l2", "ssim", "ssim_l1"]
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
    for (model, loss), run in _ordered(runs):
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


def _ordered(runs):
    """Runs of one trial ordered as MODELS x LOSSES."""
    return sorted(runs.items(), key=lambda kv: (MODELS.index(kv[0][0]), LOSSES.index(kv[0][1])))


def metric_table(runs, key, epochs):
    """List of dicts (one per (model, loss) run) with the value of `key` logged at each of the given epochs."""
    rows = []
    for (model, loss), run in _ordered(runs):
        by_epoch = dict(zip(*metric_history(run, key)))
        rows.append({"model": model, "loss": loss, **{epoch: by_epoch.get(epoch) for epoch in epochs}})
    return rows


# ---------------------------------------------------------------- media

def _media_files(value):
    """File paths inside a logged media value (image list, plotly or table)."""
    # run.summary returns SummarySubDict, which is dict-like but not a dict subclass
    if value is not None and not isinstance(value, dict) and hasattr(value, "keys"):
        value = {k: value[k] for k in value.keys()}
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


def good_vs_defective_means(rows, score="score_ssim"):
    """Mean error of good and defective test images, per category and overall."""
    out = {}
    for category in sorted({r["category"] for r in rows}) + ["all"]:
        sel = rows if category == "all" else [r for r in rows if r["category"] == category]
        good = [r[score] for r in sel if r["label"] == 0]
        bad = [r[score] for r in sel if r["label"] == 1]
        out[category] = (sum(good) / len(good), sum(bad) / len(bad))
    return out
