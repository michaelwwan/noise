#!/usr/bin/env python3
"""
Compare _masking_coord_ outputs from two NOISE inference runs (e.g. old_inference
vs new_inference) on a given dataset (human/mouse) using standard object
detection Average Precision (AP25, AP50, AP75; optionally COCO-style
mAP@[.5:.95]), plus count- and area-agreement statistics (Bland-Altman,
Dice, relative error, KS test, paired t-test/Wilcoxon signed-rank).

Expects a directory layout of:
    <data-root>/<model>_masking_coord_<dataset>/<image_id>.txt

where each .txt file has a header line followed by one row per detection:
    box_x1,box_y1,box_x2,box_y2,objectness_score,mask_x1,mask_y1,mask_x2,mask_y2,...
(trailing columns are a flat x,y polygon of arbitrary length), or a single
line reading "No osteoclasts detected" for images with zero detections.

By default, a row per run is appended to <data-root>/equiv-results.csv
(pass --csv to override the path, or --no-csv to skip it), and a single
summary figure is written to <data-root>/equiv-plots_<dataset>.png (pass
--plot to override, or --no-plot to skip it).

Usage:
    python val_equiv.py --data-root "/path/to/7-31-26-equiv-test" --dataset human
    python val_equiv.py --data-root "/path/to/7-31-26-equiv-test" --dataset mouse \
        --model-a old_inference --model-b new_inference --coco-map
"""
import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import ks_2samp, ttest_rel, wilcoxon
from shapely.geometry import Polygon

NO_DETECTIONS_MARKER = "no osteoclasts detected"


def parse_masking_coord_file(path):
    """Parse one <image_id>.txt file into a list of detections."""
    dets = []
    lines = path.read_text().splitlines()
    for line in lines[1:]:  # skip header
        line = line.strip()
        if not line or line.lower() == NO_DETECTIONS_MARKER:
            continue
        parts = line.split(",")
        box = tuple(float(v) for v in parts[0:4])
        score = float(parts[4])
        flat = [float(v) for v in parts[5:]]
        polygon = None
        if len(flat) >= 6:  # need >= 3 points to form a polygon
            pts = list(zip(flat[0::2], flat[1::2]))
            polygon = Polygon(pts)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
        dets.append({"box": box, "score": score, "polygon": polygon})
    return dets


def load_run(directory):
    """Map image_id -> detections for every .txt file in a masking_coord directory."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"masking_coord directory not found: {directory}")
    run = {}
    for p in sorted(directory.glob("*.txt")):
        stem = p.stem.strip()  # tolerate filenames like "25 .txt"
        image_id = int(stem) if stem.isdigit() else stem
        run[image_id] = parse_masking_coord_file(p)
    return run


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def iou(det_a, det_b, metric):
    if metric == "box" or det_a["polygon"] is None or det_b["polygon"] is None:
        return box_iou(det_a["box"], det_b["box"])
    poly_a, poly_b = det_a["polygon"], det_b["polygon"]
    if poly_a.is_empty or poly_b.is_empty:
        return box_iou(det_a["box"], det_b["box"])
    try:
        inter = poly_a.intersection(poly_b).area
        union = poly_a.union(poly_b).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return box_iou(det_a["box"], det_b["box"])


def detection_area(det):
    if det["polygon"] is not None and not det["polygon"].is_empty:
        return det["polygon"].area
    x1, y1, x2, y2 = det["box"]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def average_precision(runs_a, runs_b, image_ids, threshold, metric):
    """COCO/VOC-style AP treating model A's detections as reference (ground truth)
    and model B's detections as predictions ranked by objectness_score."""
    per_image_iou = {}
    n_gt = 0
    predictions = []  # (score, image_id, det_index_in_b)
    for image_id in image_ids:
        dets_a = runs_a.get(image_id, [])
        dets_b = runs_b.get(image_id, [])
        n_gt += len(dets_a)
        if dets_a and dets_b:
            mat = np.zeros((len(dets_a), len(dets_b)))
            for i, da in enumerate(dets_a):
                for j, db in enumerate(dets_b):
                    mat[i, j] = iou(da, db, metric)
            per_image_iou[image_id] = mat
        for j, db in enumerate(dets_b):
            predictions.append((db["score"], image_id, j))

    if n_gt == 0:
        return 0.0

    predictions.sort(key=lambda x: x[0], reverse=True)
    gt_used = {image_id: set() for image_id in image_ids}
    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))
    for k, (_, image_id, j) in enumerate(predictions):
        mat = per_image_iou.get(image_id)
        best_iou, best_i = 0.0, -1
        if mat is not None:
            for i in range(mat.shape[0]):
                if i in gt_used[image_id]:
                    continue
                if mat[i, j] > best_iou:
                    best_iou, best_i = mat[i, j], i
        if best_iou >= threshold:
            tp[k] = 1
            gt_used[image_id].add(best_i)
        else:
            fp[k] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / n_gt
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(float).eps)

    # All-point interpolation (Pascal VOC 2010+ / monotone precision envelope).
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    idx = np.where(recalls[1:] != recalls[:-1])[0]
    ap = float(np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1]))
    return ap


def match_instances(runs_a, runs_b, image_ids, metric, match_iou):
    """Optimal one-to-one IoU matching per image (Hungarian algorithm).

    Returns a list of (image_id, iou) for every matched pair with
    iou >= match_iou -- used to pair up individual detections for the
    area-agreement statistics below.
    """
    matches = []
    for image_id in image_ids:
        dets_a = runs_a.get(image_id, [])
        dets_b = runs_b.get(image_id, [])
        if not dets_a or not dets_b:
            continue
        mat = np.zeros((len(dets_a), len(dets_b)))
        for i, da in enumerate(dets_a):
            for j, db in enumerate(dets_b):
                mat[i, j] = iou(da, db, metric)
        row_ind, col_ind = linear_sum_assignment(-mat)
        for i, j in zip(row_ind, col_ind):
            if mat[i, j] >= match_iou:
                matches.append((dets_a[i], dets_b[j], mat[i, j]))
    return matches


def paired_stats(a, b):
    """MAE, RMSE, Bland-Altman bias/limits-of-agreement, KS test, paired
    t-test, and Wilcoxon signed-rank test between two equal-length paired
    arrays (a treated as reference, b as comparison)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = b - a
    n = len(a)
    stats = {
        "mae": float(np.mean(np.abs(diff))) if n else float("nan"),
        "rmse": float(np.sqrt(np.mean(diff ** 2))) if n else float("nan"),
        "ba_bias": float(np.mean(diff)) if n else float("nan"),
    }
    sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    stats["ba_loa_low"] = stats["ba_bias"] - 1.96 * sd
    stats["ba_loa_high"] = stats["ba_bias"] + 1.96 * sd

    ks_stat, ks_p = ks_2samp(a, b) if n else (float("nan"), float("nan"))
    stats["ks_stat"], stats["ks_pvalue"] = float(ks_stat), float(ks_p)

    try:
        stats["ttest_pvalue"] = float(ttest_rel(a, b).pvalue) if n > 1 else float("nan")
    except Exception:
        stats["ttest_pvalue"] = float("nan")
    try:
        stats["wilcoxon_pvalue"] = float(wilcoxon(a, b).pvalue) if n > 1 and np.any(diff) else float("nan")
    except Exception:
        stats["wilcoxon_pvalue"] = float("nan")
    return stats


def make_figure(out_path, dataset, model_a, model_b, counts_a, counts_b,
                 matched_areas_a, matched_areas_b, all_areas_a, all_areas_b,
                 count_stats, area_stats):
    counts_a, counts_b = np.asarray(counts_a, dtype=float), np.asarray(counts_b, dtype=float)
    matched_areas_a = np.asarray(matched_areas_a, dtype=float)
    matched_areas_b = np.asarray(matched_areas_b, dtype=float)
    all_areas_a, all_areas_b = np.asarray(all_areas_a, dtype=float), np.asarray(all_areas_b, dtype=float)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"{model_a} vs {model_b} — {dataset} dataset equivalence", fontsize=14)

    # (0,0) Count scatter
    ax = axes[0, 0]
    lim = [0, max(counts_a.max(), counts_b.max()) * 1.1 + 1] if len(counts_a) else [0, 1]
    ax.plot(lim, lim, "k--", linewidth=1, label="y = x")
    ax.scatter(counts_a, counts_b, alpha=0.7)
    ax.set_title("Detections per Image")
    ax.set_xlabel(f"{model_a} count")
    ax.set_ylabel(f"{model_b} count")
    ax.legend()

    # (0,1) Count Bland-Altman
    ax = axes[0, 1]
    mean_counts = (counts_a + counts_b) / 2
    diff_counts = counts_b - counts_a
    ax.scatter(mean_counts, diff_counts, alpha=0.7)
    ax.axhline(count_stats["ba_bias"], color="red", linewidth=1.5, label=f"bias = {count_stats['ba_bias']:.2f}")
    ax.axhline(count_stats["ba_loa_low"], color="gray", linestyle="--", linewidth=1, label="±1.96 SD")
    ax.axhline(count_stats["ba_loa_high"], color="gray", linestyle="--", linewidth=1)
    ax.set_title("Count Bland-Altman")
    ax.set_xlabel("Mean count (A, B)")
    ax.set_ylabel(f"Difference ({model_b} − {model_a})")
    ax.legend()

    # (0,2) Count distribution
    ax = axes[0, 2]
    max_count = int(max(counts_a.max(), counts_b.max())) if len(counts_a) else 0
    bins = np.arange(-0.5, max_count + 1.5, 1)
    ax.hist(counts_a, bins=bins, alpha=0.5, label=model_a)
    ax.hist(counts_b, bins=bins, alpha=0.5, label=model_b)
    ax.set_title(f"Count Distribution (KS p={count_stats['ks_pvalue']:.3f})")
    ax.set_xlabel("Detections per image")
    ax.set_ylabel("Number of images")
    ax.legend()

    # (1,0) Matched-instance area scatter (log-log)
    ax = axes[1, 0]
    if len(matched_areas_a):
        lo = max(min(matched_areas_a.min(), matched_areas_b.min()), 1e-6)
        hi = max(matched_areas_a.max(), matched_areas_b.max()) * 1.1
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
        ax.scatter(matched_areas_a, matched_areas_b, alpha=0.5, s=12)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
    ax.set_title("Matched Instance Mask Area")
    ax.set_xlabel(f"{model_a} area (px²)")
    ax.set_ylabel(f"{model_b} area (px²)")

    # (1,1) Matched-instance area Bland-Altman
    ax = axes[1, 1]
    if len(matched_areas_a):
        mean_areas = (matched_areas_a + matched_areas_b) / 2
        diff_areas = matched_areas_b - matched_areas_a
        ax.scatter(mean_areas, diff_areas, alpha=0.5, s=12)
        ax.axhline(area_stats["ba_bias"], color="red", linewidth=1.5, label=f"bias = {area_stats['ba_bias']:.1f}")
        ax.axhline(area_stats["ba_loa_low"], color="gray", linestyle="--", linewidth=1, label="±1.96 SD")
        ax.axhline(area_stats["ba_loa_high"], color="gray", linestyle="--", linewidth=1)
        ax.legend()
    ax.set_title("Matched Instance Area Bland-Altman")
    ax.set_xlabel("Mean area (px²)")
    ax.set_ylabel(f"Difference ({model_b} − {model_a})")

    # (1,2) Area distribution (all detections) -- log-spaced bins since mask
    # area spans orders of magnitude (matches the log-log scatter above).
    ax = axes[1, 2]
    if len(all_areas_a) or len(all_areas_b):
        combined = np.concatenate([a for a in (all_areas_a, all_areas_b) if len(a)])
        combined = combined[combined > 0]
        bins = np.logspace(np.log10(combined.min()), np.log10(combined.max()), 30)
        ax.hist(all_areas_a, bins=bins, alpha=0.5, label=model_a)
        ax.hist(all_areas_b, bins=bins, alpha=0.5, label=model_b)
        ax.set_xscale("log")
        ax.legend()
    ax.set_title(f"Area Distribution, All Detections (KS p={area_stats['ks_pvalue']:.3f})")
    ax.set_xlabel("Mask area (px², log scale)")
    ax.set_ylabel("Number of detections")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def sig(x, digits=3):
    """Round a number to a fixed number of significant figures (not decimal places)."""
    if x is None:
        return x
    try:
        x = float(x)
    except (TypeError, ValueError):
        return x
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, digits - int(math.floor(math.log10(abs(x)))) - 1)


CSV_FIELDS = [
    "timestamp", "model_a", "model_b", "dataset", "metric",
    "ap25", "ap50", "ap75", "map_coco",
    "count_mae", "count_rmse", "count_ba_bias", "count_ba_loa_low", "count_ba_loa_high",
    "count_ks_stat", "count_ks_pvalue", "count_ttest_pvalue", "count_wilcoxon_pvalue",
    "n_matched_instances", "area_dice_mean", "area_relative_error_mean",
    "area_ba_bias", "area_ba_loa_low", "area_ba_loa_high",
    "area_ks_stat", "area_ks_pvalue", "area_ttest_pvalue", "area_wilcoxon_pvalue",
]


def write_csv(csv_path, args, ap25, ap50, ap75, map_coco, count_stats, area_stats, n_matched):
    """Append one row for this run to csv_path, writing a header only if new."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model_a": args.model_a,
            "model_b": args.model_b,
            "dataset": args.dataset,
            "metric": args.metric,
            "ap25": sig(ap25),
            "ap50": sig(ap50),
            "ap75": sig(ap75),
            "map_coco": sig(map_coco) if map_coco is not None else "",
            "count_mae": sig(count_stats["mae"]),
            "count_rmse": sig(count_stats["rmse"]),
            "count_ba_bias": sig(count_stats["ba_bias"]),
            "count_ba_loa_low": sig(count_stats["ba_loa_low"]),
            "count_ba_loa_high": sig(count_stats["ba_loa_high"]),
            "count_ks_stat": sig(count_stats["ks_stat"]),
            "count_ks_pvalue": sig(count_stats["ks_pvalue"]),
            "count_ttest_pvalue": sig(count_stats["ttest_pvalue"]),
            "count_wilcoxon_pvalue": sig(count_stats["wilcoxon_pvalue"]),
            "n_matched_instances": n_matched,
            "area_dice_mean": sig(area_stats["dice_mean"]),
            "area_relative_error_mean": sig(area_stats["relative_error_mean"]),
            "area_ba_bias": sig(area_stats["ba_bias"]),
            "area_ba_loa_low": sig(area_stats["ba_loa_low"]),
            "area_ba_loa_high": sig(area_stats["ba_loa_high"]),
            "area_ks_stat": sig(area_stats["ks_stat"]),
            "area_ks_pvalue": sig(area_stats["ks_pvalue"]),
            "area_ttest_pvalue": sig(area_stats["ttest_pvalue"]),
            "area_wilcoxon_pvalue": sig(area_stats["wilcoxon_pvalue"]),
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, help="Directory containing <model>_masking_coord_<dataset>/ folders")
    parser.add_argument("--model-a", default="old_inference", help="Reference model name (default: old_inference)")
    parser.add_argument("--model-b", default="new_inference", help="Comparison model name (default: new_inference)")
    parser.add_argument("--dataset", default="human", choices=["human", "mouse"], help="Dataset to compare (default: human)")
    parser.add_argument("--folder-pattern", default="{model}_masking_coord_{dataset}", help="Folder naming pattern under --data-root")
    parser.add_argument("--metric", default="mask", choices=["mask", "box"], help="IoU basis (default: mask polygon IoU, falls back to box IoU per-detection when a mask is missing/invalid)")
    parser.add_argument("--coco-map", action="store_true", help="Also compute COCO-style mAP averaged over IoU 0.5:0.95:0.05")
    parser.add_argument("--match-iou", type=float, default=0.5, help="IoU threshold for pairing individual detections for area-agreement stats (default: 0.5)")
    parser.add_argument("--json", default=None, help="Optional path to dump full results as JSON")
    parser.add_argument("--csv", default=None, help="Path to append CSV results to (default: <data-root>/equiv-results.csv)")
    parser.add_argument("--no-csv", action="store_true", help="Disable CSV output")
    parser.add_argument("--plot", default=None, help="Path to write the summary figure (default: <data-root>/equiv-plots_<dataset>.png)")
    parser.add_argument("--no-plot", action="store_true", help="Disable figure output")
    args = parser.parse_args()

    dir_a = Path(args.data_root) / args.folder_pattern.format(model=args.model_a, dataset=args.dataset)
    dir_b = Path(args.data_root) / args.folder_pattern.format(model=args.model_b, dataset=args.dataset)
    runs_a = load_run(dir_a)
    runs_b = load_run(dir_b)

    image_ids = sorted(set(runs_a) | set(runs_b), key=lambda x: (isinstance(x, str), x))
    only_in_a = sorted(set(runs_a) - set(runs_b))
    only_in_b = sorted(set(runs_b) - set(runs_a))

    print(f"Comparing {args.model_a!r} vs {args.model_b!r} on dataset {args.dataset!r}")
    print(f"  {dir_a}\n  {dir_b}")
    print(f"  {len(image_ids)} images total ({len(runs_a)} in A, {len(runs_b)} in B)")
    if only_in_a:
        print(f"  WARNING: {len(only_in_a)} image(s) only in {args.model_a}: {only_in_a}")
    if only_in_b:
        print(f"  WARNING: {len(only_in_b)} image(s) only in {args.model_b}: {only_in_b}")
    print()

    # --- Detection AP ---
    ap25 = average_precision(runs_a, runs_b, image_ids, 0.25, args.metric)
    ap50 = average_precision(runs_a, runs_b, image_ids, 0.50, args.metric)
    ap75 = average_precision(runs_a, runs_b, image_ids, 0.75, args.metric)
    print(f"AP25 = {sig(ap25)}   AP50 = {sig(ap50)}   AP75 = {sig(ap75)}")

    map_coco = None
    if args.coco_map:
        coco_thresholds = np.arange(0.5, 1.0, 0.05)
        aps = [average_precision(runs_a, runs_b, image_ids, t, args.metric) for t in coco_thresholds]
        map_coco = float(np.mean(aps))
        print(f"mAP@[.5:.95] = {sig(map_coco)}")

    # --- Count agreement ---
    counts_a = [len(runs_a.get(i, [])) for i in image_ids]
    counts_b = [len(runs_b.get(i, [])) for i in image_ids]
    count_stats = paired_stats(counts_a, counts_b)
    print(f"\nCount agreement (n={len(image_ids)} images):")
    print(f"  MAE={sig(count_stats['mae'])}  RMSE={sig(count_stats['rmse'])}"
          f"  Bland-Altman bias={sig(count_stats['ba_bias'])}"
          f" (LoA {sig(count_stats['ba_loa_low'])} to {sig(count_stats['ba_loa_high'])})")
    print(f"  KS stat={sig(count_stats['ks_stat'])} (p={sig(count_stats['ks_pvalue'])})"
          f"  paired t-test p={sig(count_stats['ttest_pvalue'])}"
          f"  Wilcoxon p={sig(count_stats['wilcoxon_pvalue'])}")

    # --- Area agreement ---
    matches = match_instances(runs_a, runs_b, image_ids, args.metric, args.match_iou)
    matched_areas_a = [detection_area(da) for da, db, _ in matches]
    matched_areas_b = [detection_area(db) for da, db, _ in matches]
    all_areas_a = [detection_area(d) for dets in runs_a.values() for d in dets]
    all_areas_b = [detection_area(d) for dets in runs_b.values() for d in dets]

    if matches:
        dice_vals = [2 * v / (1 + v) for _, _, v in matches]
        rel_err_vals = [abs(ba - aa) / aa if aa > 0 else 0.0 for aa, ba in zip(matched_areas_a, matched_areas_b)]
        area_stats = paired_stats(matched_areas_a, matched_areas_b)
        area_stats["dice_mean"] = float(np.mean(dice_vals))
        area_stats["relative_error_mean"] = float(np.mean(rel_err_vals))
        # KS test on area distribution should reflect the full detection sets, not just matches
        ks_stat, ks_p = ks_2samp(all_areas_a, all_areas_b) if all_areas_a and all_areas_b else (float("nan"), float("nan"))
        area_stats["ks_stat"], area_stats["ks_pvalue"] = float(ks_stat), float(ks_p)
    else:
        area_stats = {k: float("nan") for k in (
            "dice_mean", "relative_error_mean", "ba_bias", "ba_loa_low", "ba_loa_high",
            "ks_stat", "ks_pvalue", "ttest_pvalue", "wilcoxon_pvalue",
        )}

    print(f"\nArea agreement ({len(matches)} matched instances @ IoU>={args.match_iou}):")
    print(f"  Dice={sig(area_stats['dice_mean'])}  Mean relative error={sig(area_stats['relative_error_mean'])}")
    print(f"  Bland-Altman bias={sig(area_stats['ba_bias'])}"
          f" (LoA {sig(area_stats['ba_loa_low'])} to {sig(area_stats['ba_loa_high'])})")
    print(f"  KS stat={sig(area_stats['ks_stat'])} (p={sig(area_stats['ks_pvalue'])}, all detections)"
          f"  paired t-test p={sig(area_stats['ttest_pvalue'])}"
          f"  Wilcoxon p={sig(area_stats['wilcoxon_pvalue'])}")

    results = {
        "config": vars(args),
        "ap25": sig(ap25), "ap50": sig(ap50), "ap75": sig(ap75), "map_coco": sig(map_coco),
        "count_stats": {k: sig(v) for k, v in count_stats.items()},
        "area_stats": {k: sig(v) for k, v in area_stats.items()},
        "n_matched_instances": len(matches),
    }

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, default=str))
        print(f"\nWrote full results to {args.json}")

    if not args.no_csv:
        csv_path = Path(args.csv) if args.csv else Path(args.data_root) / "equiv-results.csv"
        write_csv(csv_path, args, ap25, ap50, ap75, map_coco, count_stats, area_stats, len(matches))
        print(f"Appended results to {csv_path}")

    if not args.no_plot:
        plot_path = Path(args.plot) if args.plot else Path(args.data_root) / f"equiv-plots_{args.dataset}.png"
        make_figure(plot_path, args.dataset, args.model_a, args.model_b, counts_a, counts_b,
                    matched_areas_a, matched_areas_b, all_areas_a, all_areas_b, count_stats, area_stats)
        print(f"Wrote summary figure to {plot_path}")


if __name__ == "__main__":
    main()
