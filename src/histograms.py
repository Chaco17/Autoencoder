"""Histograms of the reconstruction error (good vs each defect type) from the rows of a test/scores table.

They are logged into the training runs themselves (scripts/log_histograms.py, or the notebook when a run
does not have them yet) and the notebook downloads them from there.
"""
import time

import plotly.graph_objects as go
import wandb
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score

from src.wandb_utils import load_table

SCORES = ["score_l1", "score_mse", "score_ssim", "score_max"]
SCORE_NAMES = {
    "score_l1": "error L1 medio",
    "score_mse": "error cuadrático medio",
    "score_ssim": "1 - SSIM",
    "score_max": "máximo del mapa de error suavizado",
}


def _auroc(good, bad):
    return roc_auc_score([0] * len(good) + [1] * len(bad), list(good) + list(bad))


def error_histograms(rows, category, score="score_ssim", title_prefix=""):
    """One subplot per defect type of `category`: error of the good images (green) vs that defect (red).

    Normalized as probability (there are ~3 times more good images than images of each defect) and with
    the same bins in both, so the overlap reads directly; the AUROC of each defect is in its subplot title.
    """
    cat_rows = [r for r in rows if r["category"] == category]
    good = [r[score] for r in cat_rows if r["label"] == 0]
    defects = sorted({r["defect"] for r in cat_rows} - {"good"})
    values = [r[score] for r in cat_rows]
    bins = dict(start=min(values), end=max(values), size=(max(values) - min(values)) / 25 or 1e-6)
    titles = []
    for defect in defects:
        bad = [r[score] for r in cat_rows if r["defect"] == defect]
        titles.append(f"{defect} (n={len(bad)}, AUROC {_auroc(good, bad):.2f})")
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
                      title=f"{title_prefix}{category}: error de reconstrucción ({SCORE_NAMES.get(score, score)}), "
                            "good vs cada defecto")
    return fig


def histogram_key(score, category):
    """W&B key under which the histograms of one score and category are logged."""
    return f"test/histograms/{score}/{category}"


def missing_keys(run, rows):
    """Histogram keys (every score x category of `rows`) that the run does not have yet."""
    categories = sorted({r["category"] for r in rows})
    present = set(run.summary.keys())
    return [histogram_key(s, c) for s in SCORES for c in categories if histogram_key(s, c) not in present]


def log_histograms(run, rows, dir=None):
    """Log every histogram (all scores x categories) into the finished training run itself (resumed, same id)."""
    categories = sorted({r["category"] for r in rows})
    figures = {histogram_key(s, c): error_histograms(rows, c, s, title_prefix=f"{run.name} - ")
               for s in SCORES for c in categories}
    # resume="must": appends to the existing run (same id, config and metrics); fails instead of creating a new one
    resumed = wandb.init(entity=run.entity, project=run.project, id=run.id, resume="must", dir=dir,
                         settings=wandb.Settings(silent=True))
    resumed.log({key: wandb.Plotly(fig) for key, fig in figures.items()})
    resumed.finish()


def _fresh(run):
    """The run as it is now in W&B (a new Api each time: wandb.Api caches every run it returns)."""
    return wandb.Api().run(f"{run.entity}/{run.project}/{run.id}")


def ensure_histograms(run, retries=40, wait=3):
    """The run with its histograms in W&B: if any is missing, logs them first and returns the refreshed run."""
    rows = load_table(run)
    if not missing_keys(run, rows):
        return run
    run = _fresh(run)  # the given object can be older than a previous upload: check W&B before logging again
    if not missing_keys(run, rows):
        return run
    print(f"  {run.name}: subiendo sus histogramas a W&B...", flush=True)
    log_histograms(run, rows)
    for _ in range(retries):  # the summary of the run can take a few seconds to show the new keys
        fresh = _fresh(run)
        if not missing_keys(fresh, rows):
            return fresh
        time.sleep(wait)
    raise RuntimeError(f"{run.name}: los histogramas se subieron pero aún no aparecen en el summary de W&B; "
                       "volver a ejecutar la celda en unos segundos")
