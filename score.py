"""Score the classifier's predictions against the Gui et al. expert-labelled Instagram set.

Truth comes from the authors' Dataset.csv (column `category`), which is not in this repo:
download it from https://github.com/HaoyangGui/Evaluating-LLM-Generated-Legal-Explanations
(dataset/Dataset.csv). Predictions come from predictions.csv (row_id, flagged_ad), where
flagged_ad = 1 means the classifier gave its advertising/promotion output.

Main test, the 551 expert-labelled posts (category undisclosed = hidden ad, 127; organic, 424):
  precision, recall and accuracy with Wilson 95% intervals; F1 with a 95% percentile
  interval from a bootstrap over the 551 posts (2,000 draws, numpy default_rng seed 51).
  Context only: the F1 of always answering "ad" and the accuracy of always answering "organic".
Secondary, the 592 disclosed ads (category disclosed), reported apart and never mixed in:
  recall with a Wilson 95% interval.

Usage:
    python score.py --dataset data/Dataset.csv --predictions predictions.csv [--json out.json]
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

EXPECTED_MD5 = "aae56369561b23a0be5864159264eee5"
BOOT, SEED = 2000, 51


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, type=Path, help="the authors' Dataset.csv")
    ap.add_argument("--predictions", required=True, type=Path, help="predictions.csv")
    ap.add_argument("--json", type=Path, help="also write the results as JSON to this path")
    args = ap.parse_args()

    md5 = hashlib.md5(args.dataset.read_bytes()).hexdigest()
    if md5 != EXPECTED_MD5:
        print(f"WARNING: Dataset.csv md5 is {md5}, expected {EXPECTED_MD5}. "
              "Numbers may differ from the published ones.\n")

    with args.predictions.open(encoding="utf-8", newline="") as fh:
        pred = {r["row_id"]: r["flagged_ad"] == "1" for r in csv.DictReader(fh)}
    with args.dataset.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))  # file order is kept: the bootstrap depends on it
    missing = [r["id"] for r in rows if r["id"] not in pred]
    if missing:
        raise SystemExit(f"{len(missing)} dataset ids have no prediction, e.g. {missing[:5]}")

    exp = [r for r in rows if r["category"] in ("undisclosed", "organic")]
    dis = [r for r in rows if r["category"] == "disclosed"]
    y = np.array([r["category"] == "undisclosed" for r in exp])
    yhat = np.array([pred[r["id"]] for r in exp])
    tp = int((y & yhat).sum()); fp = int((~y & yhat).sum())
    fn = int((y & ~yhat).sum()); tn = int((~y & ~yhat).sum())
    n = len(exp)
    prec, rec = tp / (tp + fp) if tp + fp else float("nan"), tp / (tp + fn)
    acc = (tp + tn) / n
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(BOOT):
        i = rng.integers(0, n, n)
        a, b = y[i], yhat[i]
        boots.append(f1(int((a & b).sum()), int((~a & b).sum()), int((a & ~b).sum())))
    f1_lo, f1_hi = np.percentile(boots, [2.5, 97.5])
    n_pos = int(y.sum())
    always_ad_f1 = f1(n_pos, n - n_pos, 0)
    majority_acc = (n - n_pos) / n
    dk = sum(pred[r["id"]] for r in dis)

    res = {
        "n_expert": n, "n_hidden_ad": n_pos, "n_organic": n - n_pos, "n_disclosed": len(dis),
        "confusion_binary": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": [prec, *wilson(tp, tp + fp)], "recall": [rec, *wilson(tp, tp + fn)],
        "f1": [f1(tp, fp, fn), float(f1_lo), float(f1_hi)],
        "accuracy": [acc, *wilson(tp + tn, n)],
        "always_ad_f1": always_ad_f1, "always_organic_accuracy": majority_acc,
        "disclosed_recall": [dk / len(dis) if dis else float("nan"), *wilson(dk, len(dis))],
        "dataset_md5": md5, "bootstrap": {"draws": BOOT, "seed": SEED},
    }
    if args.json:
        args.json.write_text(json.dumps(res, indent=2), encoding="utf-8")

    pct = lambda t: f"{t[0]:.1%} (95% {t[1]:.1%} to {t[2]:.1%})"
    print("\n".join([
        f"Main test: {n} expert-labelled posts ({n_pos} hidden ads, {n - n_pos} organic)",
        "",
        "| Metric | Value |", "|---|---|",
        f"| Precision | {pct(res['precision'])} |",
        f"| Recall | {pct(res['recall'])} |",
        f"| F1 | {res['f1'][0]:.3f} (95% bootstrap {f1_lo:.3f} to {f1_hi:.3f}) |",
        f"| Accuracy | {pct(res['accuracy'])} |",
        f"| Context: F1 of always answering ad | {always_ad_f1:.3f} |",
        f"| Context: accuracy of always answering organic | {majority_acc:.1%} |",
        "",
        f"Binary confusion: TP {tp}, FP {fp}, FN {fn}, TN {tn}.",
        "",
        f"Secondary, {len(dis)} disclosed ads (kept apart): recall {pct(res['disclosed_recall'])}, "
        f"{dk} of {len(dis)}.",
    ]))


if __name__ == "__main__":
    main()
