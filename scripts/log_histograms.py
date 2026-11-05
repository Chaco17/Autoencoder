"""Add the reconstruction-error histograms to the finished training runs in W&B (no retraining).

For every finished training run of the project (newest per model/loss/trial), downloads its test/scores
table, builds the histograms (good vs each defect type, one figure per class and score) and logs them
into that same run under test/histograms/<score>/<category>. The notebook then only downloads them.

    .venv\\Scripts\\python scripts/log_histograms.py --dry-run   # list what would be done
    .venv\\Scripts\\python scripts/log_histograms.py             # all trials
    .venv\\Scripts\\python scripts/log_histograms.py --trial e200_cos

Runs that already have every histogram are skipped, so it can be launched again safely. The notebook
(section 8.2) also logs them by itself for the runs it shows, so running this script is optional.
Do not launch it while train.py or the notebook are running: while a run is being resumed its
state is "running", and both only look at finished runs.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))  # appended (not inserted) so the local wandb/ run folder never shadows the package

from omegaconf import OmegaConf  # noqa: E402

from src import wandb_utils as wu  # noqa: E402
from src.histograms import log_histograms, missing_keys  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trial", action="append", help="only these trials (repeatable); default: all")
    parser.add_argument("--dry-run", action="store_true", help="only list the runs that would be updated")
    args = parser.parse_args()

    cfg = OmegaConf.load(ROOT / "conf" / "logger" / "wandb.yaml").wandb
    all_runs = wu.get_all_runs(cfg.project, cfg.entity)
    todo = []
    for (model, loss, trial), run in sorted(all_runs.items()):
        if args.trial and trial not in args.trial:
            continue
        if "test/scores" not in run.summary.keys():
            print(f"  sin test/scores, se omite: {run.name}")
            continue
        rows = wu.load_table(run)
        missing = missing_keys(run, rows)
        status = "pendiente" if missing else "ya tiene los histogramas"
        print(f"{run.name:<28} {run.id}  {status}")
        if missing:
            todo.append((run, rows))

    print(f"\n{len(todo)} corridas por actualizar en {cfg.entity}/{cfg.project}")
    if args.dry_run or not todo:
        return

    for i, (run, rows) in enumerate(todo, start=1):
        print(f"[{i}/{len(todo)}] {run.name} ({run.id})", flush=True)
        log_histograms(run, rows, dir=str(ROOT))
    print("Listo: volver a ejecutar en el notebook la celda que carga las corridas (sección 7) y luego la 8.2.")


if __name__ == "__main__":
    main()
