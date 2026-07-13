"""Insert one experiment row into autoresearch_experiments_jul14 v1.

Usage: python log_row.py <commit> <val_metric> <peak_memory_gb> <status> <description>
"""

import sys

import pandas as pd
import hopsworks


def main():
    commit, val_metric, peak_gb, status, description = sys.argv[1:6]
    project = hopsworks.login()
    fs = project.get_feature_store()
    fg = fs.get_feature_group("autoresearch_experiments_jul14", version=1)
    row = pd.DataFrame([{
        "commit": commit,
        "val_metric": float(val_metric),
        "peak_memory_gb": float(peak_gb),
        "status": status,
        "description": description,
        "ts": pd.Timestamp.now(),
    }])
    fg.insert(row, wait=True)
    print(f"logged {commit} {status} val_metric={val_metric}")


if __name__ == "__main__":
    main()
