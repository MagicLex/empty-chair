"""Register the I1 job `score-universe` (PYTHON, pandas-training-pipeline).

Bakes the FUSE data dir into args; 16GB for the PSC record dict over ~5M companies.
Run:  hops job run score-universe
"""
from __future__ import annotations

from pathlib import Path

import hopsworks

JOB_NAME = "score-universe"
ENV_NAME = "pandas-training-pipeline"
_here = Path(__file__).resolve()
_rel = str(_here).split("/hopsfs/", 1)[1].rsplit("/", 1)[0]
_data = str(_here.parent / "data")


def main():
    project = hopsworks.login()
    ja = project.get_job_api()
    cfg = ja.get_configuration("PYTHON")
    cfg["appPath"] = f"hdfs:///Projects/{project.name}/{_rel}/score_universe.py"
    cfg["environmentName"] = ENV_NAME
    cfg["defaultArgs"] = f"--data-dir {_data}"
    cfg["resourceConfig"]["memory"] = 16384
    job = ja.get_job(JOB_NAME)
    if job is None:
        job = ja.create_job(JOB_NAME, cfg); print(f"created {JOB_NAME}")
    else:
        job.config = cfg; job.save(); print(f"updated {JOB_NAME}")
    print(cfg["appPath"], "| args:", cfg["defaultArgs"])


if __name__ == "__main__":
    main()
