import mlflow

from finqa.utils.tracking import log_metrics, start_run


def test_run_logs_params_and_metrics(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    with start_run("smoke", "t", {"lr": 0.1}, tracking_uri=uri) as run:
        log_metrics({"ndcg": 0.5})
    data = mlflow.get_run(run.info.run_id).data
    assert data.params["lr"] == "0.1"
    assert data.metrics["ndcg"] == 0.5
    assert "git_sha" in data.tags
