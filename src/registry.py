"""Identify each experiment by its hyperparameters and find the runs that were already done.

run_key: short hash of the settings that define a training run (data, model, loss, trainer and seed).
Settings that do not change the result (dataset path, workers, progress bar, logging frequency,
W&B mode, console output) and the names (run name, trial name) are left out, so the same experiment
always gets the same key wherever it was launched and whatever it was called. The original 8 runs
(before trials existed) therefore have the same key as the "base" trial.

A run counts as done when:
- outputs/ or multirun/ contain a results.json with the same key (written at the very end of
  train.py, so interrupted runs do not count), or
- the W&B project has a *finished* run with the same key (runs launched by other team members).
For runs saved before the key existed, the key is computed from their .hydra/config.yaml or W&B config.
"""
import hashlib
import json
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

KEY_GROUPS = ("data", "model", "loss", "trainer")
IGNORED_KEYS = {
    "data": {"root", "num_workers"},
    "trainer": {"_target_", "accelerator", "devices", "log_every_n_steps", "enable_progress_bar", "enable_model_summary"},
}


def _drop_none(value):
    # Optional settings left as null (e.g. no lr scheduler) must not change the key
    if isinstance(value, dict):
        return {k: _drop_none(v) for k, v in value.items() if v is not None}
    return value


def experiment_params(cfg):
    """Settings that define a run, from a Hydra config (DictConfig) or a plain dict (W&B run config)."""
    if isinstance(cfg, DictConfig):
        params = {group: OmegaConf.to_container(cfg[group], resolve=True) for group in KEY_GROUPS}
    else:
        params = {group: dict(cfg[group]) for group in KEY_GROUPS}
    for group, ignored in IGNORED_KEYS.items():
        params[group] = {k: v for k, v in params[group].items() if k not in ignored}
    params["seed"] = cfg["seed"]
    return _drop_none(params)


def run_key(cfg):
    """12-character hash of experiment_params (same settings -> same key)."""
    text = json.dumps(experiment_params(cfg), sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------- local runs (outputs/, multirun/)

def _results_files(root):
    files = [f for folder in ("outputs", "multirun") for f in (Path(root) / folder).glob("**/results.json")]
    return sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)  # newest first


def _local_key(results_file, results):
    if results.get("run_key"):
        return results["run_key"]
    config_file = results_file.parent / ".hydra" / "config.yaml"  # runs saved before the key existed
    if not config_file.exists():
        return None
    try:
        return run_key(OmegaConf.load(config_file))
    except Exception:  # config of an older layout: cannot be compared
        return None


def local_runs(root, online_only=True):
    """{run_key: results dict} of the finished runs saved in outputs/ and multirun/ (newest per key).

    online_only: count only runs logged to W&B online (skips the short offline tests).
    """
    runs = {}
    for results_file in _results_files(root):
        try:
            results = json.loads(results_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if online_only and not results.get("wandb_url"):
            continue
        key = _local_key(results_file, results)
        if key is not None and key not in runs:
            runs[key] = results
    return runs


# ---------------------------------------------------------------- W&B runs

def wandb_runs(project, entity=None):
    """{run_key: wandb run} of the finished training runs of the W&B project (newest per key)."""
    import wandb

    api = wandb.Api(timeout=60)
    path = f"{entity}/{project}" if entity else project
    runs = {}
    for run in api.runs(path, filters={"state": "finished"}, order="-created_at"):
        if run.job_type == "analysis":  # runs created by the notebook itself
            continue
        config = run.config
        key = config.get("run_key")
        if key is None:  # runs logged before the key existed
            try:
                key = run_key(config)
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
        runs.setdefault(key, run)
    return runs


# ---------------------------------------------------------------- lookup used by train.py and the notebook

def find_existing(cfg, root, local=None, remote=None):
    """Description of an already finished run with the same experiment as cfg, or None.

    local / remote: results of local_runs() / wandb_runs() when they are already loaded (the notebook
    checks many configs at once); when missing they are loaded here.
    """
    key = run_key(cfg)
    online = cfg.logger.wandb.mode == "online"

    if local is None:
        local = local_runs(root, online_only=online)
    if key in local:
        results = local[key]
        folder = Path(results.get("output_dir", "?"))
        try:
            folder = folder.relative_to(Path(root).resolve())
        except ValueError:
            pass
        return f"local {folder}" + (f" | {results['wandb_url']}" if results.get("wandb_url") else "")

    if online and cfg.skip_existing.check_wandb:
        if remote is None:
            remote = wandb_runs(cfg.logger.wandb.project, cfg.logger.wandb.entity)
        if key in remote:
            run = remote[key]
            return f"W&B {run.name} | {run.url}"
    return None
