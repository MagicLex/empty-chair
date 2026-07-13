"""Register the F2 job `ingest-registry` as a Hopsworks PYTHON job.

Points appPath at the FUSE-staged script (one source of truth, no copy) and bakes
the resolved FUSE data dir into --args, since a job pod may stage the entry script
outside /hopsfs while still mounting the FUSE home. Idempotent.

Run:  hops job run ingest-registry
"""
from __future__ import annotations

from pathlib import Path

import hopsworks

JOB_NAME = "ingest-registry"
ENV_NAME = "python-feature-pipeline"

_here = Path(__file__).resolve()
_rel = str(_here).split("/hopsfs/", 1)[1].rsplit("/", 1)[0]  # Users/<me>/011_empty_chair
_data = str(_here.parent / "data")  # absolute FUSE path, mounted in job pods too


def main() -> None:
    project = hopsworks.login()
    ja = project.get_job_api()
    app_path = f"hdfs:///Projects/{project.name}/{_rel}/ingest_registry.py"

    cfg = ja.get_configuration("PYTHON")
    cfg["appPath"] = app_path
    cfg["environmentName"] = ENV_NAME
    cfg["defaultArgs"] = f"--data-dir {_data}"
    cfg["resourceConfig"]["memory"] = 8192  # address dict + positive rows headroom

    job = ja.get_job(JOB_NAME)
    if job is None:
        job = ja.create_job(JOB_NAME, cfg)
        print(f"created job {job.name} on {ENV_NAME}", flush=True)
    else:
        job.config = cfg
        job.save()
        print(f"updated job {job.name}", flush=True)
    print(f"appPath={app_path}\nargs=--data-dir {_data}", flush=True)


if __name__ == "__main__":
    main()
