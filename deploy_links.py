"""Register the ingest-links job (PYTHON, pandas-training-pipeline). Idempotent.
Run:  hops job run ingest-links
"""
from __future__ import annotations

from pathlib import Path

import hopsworks

JOB_NAME = "ingest-links"
ENV_NAME = "pandas-training-pipeline"
_here = Path(__file__).resolve()
_rel = str(_here).split("/hopsfs/", 1)[1].rsplit("/", 1)[0]
_data = str(_here.parent / "data")


def main():
    project = hopsworks.login()
    ja = project.get_job_api()
    cfg = ja.get_configuration("PYTHON")
    cfg["appPath"] = f"hdfs:///Projects/{project.name}/{_rel}/ingest_links.py"
    cfg["environmentName"] = ENV_NAME
    cfg["defaultArgs"] = f"--data-dir {_data}"
    cfg["resourceConfig"]["memory"] = 16384
    job = ja.get_job(JOB_NAME)
    if job is not None:  # config property lost its setter; recreate instead
        job.delete(); print(f"deleted stale {JOB_NAME}")
    ja.create_job(JOB_NAME, cfg); print(f"created {JOB_NAME}")
    print(cfg["appPath"])


if __name__ == "__main__":
    main()
