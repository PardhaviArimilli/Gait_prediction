import os
import glob
import argparse
import pandas as pd


def build_submission(intervals_dir: str, out_csv: str) -> None:
    rows = []
    files = sorted(glob.glob(os.path.join(intervals_dir, "*_intervals.csv")))
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0].replace("_intervals", "")
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=["class", "start_s", "end_s", "score"])  # empty
        # Example schema: file, class, start_s, end_s, score
        if not df.empty and {"class","start_s","end_s","score"}.issubset(df.columns):
            df2 = df.copy()
            df2.insert(0, "file", base)
            rows.append(df2[["file", "class", "start_s", "end_s", "score"]])
        else:
            rows.append(pd.DataFrame(columns=["file", "class", "start_s", "end_s", "score"]))
    out = pd.concat(rows, axis=0) if rows else pd.DataFrame(columns=["file", "class", "start_s", "end_s", "score"])
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out.to_csv(out_csv, index=False)


def main():
    ap = argparse.ArgumentParser(description="Create submission CSV from intervals directory")
    ap.add_argument("--dir", required=True, help="Directory with *_intervals.csv")
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()
    build_submission(args.dir, args.out)


if __name__ == "__main__":
    main()


