"""Thin MLflow wrapper: local SQLite store, logs params, metrics, git SHA and the config used."""
import subprocess
from contextlib import contextmanager
from pathlib import Path

import mlflow


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@contextmanager
def start_run(experiment: str, run_name: str, params: dict, config_path: str | None = None,
              tracking_uri: str = "sqlite:///mlflow.db"):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("git_sha", _git_sha())
        mlflow.log_params(params)
        if config_path and Path(config_path).exists():
            mlflow.log_artifact(config_path)
        yield run


def log_metrics(metrics: dict, step: int | None = None) -> None:
    # MLflow metric names disallow "@" (e.g. "ndcg@10"), so spell it "_at_".
    mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in metrics.items()}, step=step)
