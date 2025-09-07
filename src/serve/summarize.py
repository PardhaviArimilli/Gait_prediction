import os
import glob
import argparse
import json
import pandas as pd
from pandas.errors import EmptyDataError


LABELS = ["StartHesitation", "Turn", "Walking"]


def summarize_intervals(dir_path: str) -> dict:
    out = {}
    files = sorted(glob.glob(os.path.join(dir_path, "*_intervals.csv")))
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0].replace("_intervals", "")
        try:
            df = pd.read_csv(f)
            if df.empty:
                raise EmptyDataError("empty")
            stats = (
                df.groupby("class")["end_s"]
                .count()
                .rename("count")
                .to_frame()
                .join((df["end_s"] - df["start_s"]).groupby(df["class"]).sum().rename("total_duration_s"))
                .fillna(0)
            )
            classes = sorted(set(df["class"].unique()) | set(LABELS))
        except (EmptyDataError, FileNotFoundError):
            stats = pd.DataFrame(columns=["count", "total_duration_s"])  # no rows
            classes = LABELS
        out[base] = {cls: {"count": int(stats.loc[cls, "count"]) if cls in stats.index else 0,
                            "total_duration_s": float(stats.loc[cls, "total_duration_s"]) if cls in stats.index else 0.0}
                     for cls in classes}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory with *_intervals.csv")
    ap.add_argument("--out", default="summary.json")
    ap.add_argument("--csv", default="", help="Optional CSV path to write tabular summary")
    args = ap.parse_args()
    summary = summarize_intervals(args.dir)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    if args.csv:
        rows = []
        for base, classes in summary.items():
            for cls, stats in classes.items():
                rows.append({
                    "file": base,
                    "class": cls,
                    "count": stats.get("count", 0),
                    "total_duration_s": stats.get("total_duration_s", 0.0),
                })
        import pandas as pd
        pd.DataFrame(rows).to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()


