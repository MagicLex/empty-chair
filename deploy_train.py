"""Register the T job `train-chair` (PYTHON, pandas-training-pipeline). Idempotent.
Run:  hops job run train-chair
"""
from __future__ import annotations

from pathlib import Path

import hopsworks

JOB_NAME = "train-chair"
ENV_NAME = "empty-chair-train"
_rel = str(Path(__file__).resolve()).split("/hopsfs/", 1)[1].rsplit("/", 1)[0]


def main():
    project = hopsworks.login()
    ja = project.get_job_api()
    cfg = ja.get_configuration("PYTHON")
    cfg["appPath"] = f"hdfs:///Projects/{project.name}/{_rel}/train_chair.py"
    cfg["environmentName"] = ENV_NAME
    cfg["resourceConfig"]["memory"] = 8192
    job = ja.get_job(JOB_NAME)
    if job is not None:  # config property lost its setter; recreate instead
        job.delete(); print(f"deleted stale {JOB_NAME}")
    ja.create_job(JOB_NAME, cfg); print(f"created {JOB_NAME}")
    print(cfg["appPath"])


if __name__ == "__main__":
    main()
