"""Statistical analysis plots and tables for evaluation.

Generates two types of charts (all splits overlaid with different colors):
1. Metrics by GT distance: SLAB, rotation error (deg), translation error (%)
2. Metrics by min inliers threshold: SLAB, rotation error (deg),
   translation error (%), percentage of dropped samples
"""

import numpy as np
from pathlib import Path


SPLIT_COLORS = {
    "val": "#1f77b4",
    "lightbox": "#ff7f0e",
    "sunlamp": "#2ca02c",
}

BIN_EDGES = [3, 4, 5, 6, 7, 8, 9, 10]

MIN_INLIERS_THRESHOLDS = [4, 5, 6, 7, 8, 9, 10, 11]


def _get_color(split_name):
    return SPLIT_COLORS.get(split_name, "#d62728")


def _compute_distance_bins(per_sample):
    """Compute per-bin metrics for a single split."""
    distances = per_sample["gt_distances"]
    bin_indices = np.digitize(distances, BIN_EDGES) - 1
    bin_indices = np.clip(bin_indices, 0, len(BIN_EDGES) - 2)
    num_bins = len(BIN_EDGES) - 1
    bin_labels = [f"{BIN_EDGES[i]}-{BIN_EDGES[i+1]}m" for i in range(num_bins)]

    results = {}
    for bi in range(num_bins):
        mask = bin_indices == bi
        label = bin_labels[bi]
        n = int(mask.sum())
        r = {"count": n}

        if n > 0:
            valid_mask = mask & per_sample["epnp_valid"]
            n_valid = int(valid_mask.sum())
            if n_valid > 0:
                r["rot_err"] = float(per_sample["epnp_rot_errors"][valid_mask].mean())
                r["trans_rel"] = float(per_sample["epnp_trans_rel_errors"][valid_mask].mean() * 100)
                r["slab"] = float(per_sample["epnp_slab_scores"][valid_mask].mean())

        results[label] = r

    return bin_labels, results


def _compute_inliers_sweep(per_sample):
    """Compute metrics at varying min-inliers thresholds for a single split."""
    n_inliers = per_sample["epnp_n_inliers"]
    has_pose = per_sample["has_pose"]
    rot_errors = per_sample["epnp_rot_errors"]
    trans_rel = per_sample["epnp_trans_rel_errors"]
    slab_scores = per_sample["epnp_slab_scores"]
    epnp_valid = per_sample["epnp_valid"]

    n_total = int(has_pose.sum())
    results = {}
    for thresh in MIN_INLIERS_THRESHOLDS:
        # A sample is "accepted" if PnP succeeded AND n_inliers >= threshold
        accepted = epnp_valid & (n_inliers >= thresh) & has_pose
        n_accepted = int(accepted.sum())
        dropped_pct = (1.0 - n_accepted / max(n_total, 1)) * 100

        r = {"dropped_pct": dropped_pct, "n_accepted": n_accepted}
        if n_accepted > 0:
            r["rot_err"] = float(rot_errors[accepted].mean())
            r["trans_rel"] = float(trans_rel[accepted].mean() * 100)
            r["slab"] = float(slab_scores[accepted].mean())

        results[thresh] = r

    return results


def plot_metrics_by_distance(all_per_sample, output_dir):
    """Plot SLAB, rot error, trans error by GT distance (all splits overlaid)."""
    import matplotlib.pyplot as plt

    num_bins = len(BIN_EDGES) - 1
    bin_labels = [f"{BIN_EDGES[i]}-{BIN_EDGES[i+1]}m" for i in range(num_bins)]
    x_pos = np.arange(num_bins)
    n_splits = len(all_per_sample)
    bar_width = 0.8 / max(n_splits, 1)

    # Compute per-split data
    split_data = {}
    for split_name, ps in all_per_sample.items():
        _, bins = _compute_distance_bins(ps)
        split_data[split_name] = bins

    # Print table
    print(f"\n{'=' * 70}")
    print("Metrics by GT Distance (all splits)")
    print(f"{'=' * 70}")
    for split_name, bins in split_data.items():
        print(f"\n  {split_name}:")
        header = f"  {'Bin':>10} {'Count':>7} {'RotErr':>10} {'TransErr%':>10} {'SLAB':>10}"
        print(header)
        print(f"  {'-' * len(header.strip())}")
        for label in bin_labels:
            r = bins[label]
            count = r["count"]
            rot = f"{r['rot_err']:.2f}" if "rot_err" in r else "--"
            tr = f"{r['trans_rel']:.2f}" if "trans_rel" in r else "--"
            slab = f"{r['slab']:.4f}" if "slab" in r else "--"
            print(f"  {label:>10} {count:>7} {rot:>10} {tr:>10} {slab:>10}")

    # Plot: 3 subplots (SLAB, rot error, trans error)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = [
        ("slab", "SLAB Score", axes[0]),
        ("rot_err", "Rotation Error (deg)", axes[1]),
        ("trans_rel", "Translation Error (%)", axes[2]),
    ]

    for key, ylabel, ax in metrics:
        for i, (split_name, bins) in enumerate(split_data.items()):
            values = [bins[label].get(key, float("nan")) for label in bin_labels]
            offset = (i - (n_splits - 1) / 2) * bar_width
            ax.bar(x_pos + offset, values, bar_width,
                   label=split_name, color=_get_color(split_name), alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bin_labels, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by Distance")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("EPnP Metrics by GT Distance", fontsize=14)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "metrics_by_distance.png", dpi=150)
    plt.close(fig)


def plot_metrics_by_min_inliers(all_per_sample, output_dir):
    """Plot SLAB, rot error, trans error, % dropped by min-inliers threshold."""
    import matplotlib.pyplot as plt

    # Compute per-split data
    split_data = {}
    for split_name, ps in all_per_sample.items():
        split_data[split_name] = _compute_inliers_sweep(ps)

    # Print table
    print(f"\n{'=' * 70}")
    print("Metrics by Min Inliers Threshold (all splits)")
    print(f"{'=' * 70}")
    for split_name, sweep in split_data.items():
        print(f"\n  {split_name}:")
        header = f"  {'MinInl':>8} {'Accept':>8} {'Drop%':>8} {'RotErr':>10} {'TransErr%':>10} {'SLAB':>10}"
        print(header)
        print(f"  {'-' * len(header.strip())}")
        for thresh in MIN_INLIERS_THRESHOLDS:
            r = sweep[thresh]
            n_acc = r["n_accepted"]
            drop = f"{r['dropped_pct']:.1f}"
            rot = f"{r['rot_err']:.2f}" if "rot_err" in r else "--"
            tr = f"{r['trans_rel']:.2f}" if "trans_rel" in r else "--"
            slab = f"{r['slab']:.4f}" if "slab" in r else "--"
            print(f"  {thresh:>8} {n_acc:>8} {drop:>8} {rot:>10} {tr:>10} {slab:>10}")

    # Plot: 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    metrics = [
        ("slab", "SLAB Score", axes[0, 0]),
        ("rot_err", "Rotation Error (deg)", axes[0, 1]),
        ("trans_rel", "Translation Error (%)", axes[1, 0]),
        ("dropped_pct", "Dropped Samples (%)", axes[1, 1]),
    ]

    for key, ylabel, ax in metrics:
        for split_name, sweep in split_data.items():
            values = [sweep[t].get(key, float("nan")) for t in MIN_INLIERS_THRESHOLDS]
            ax.plot(MIN_INLIERS_THRESHOLDS, values, "o-", linewidth=2, markersize=6,
                    label=split_name, color=_get_color(split_name))
        ax.set_xlabel("Min Inliers Threshold")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by Min Inliers")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(MIN_INLIERS_THRESHOLDS)

    fig.suptitle("EPnP Metrics by Min Inliers Threshold", fontsize=14)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "metrics_by_min_inliers.png", dpi=150)
    plt.close(fig)


def run_stats_analysis(all_per_sample, output_dir):
    """Run all statistical analyses across all splits."""
    output_dir = Path(output_dir)

    plot_metrics_by_distance(all_per_sample, output_dir)
    plot_metrics_by_min_inliers(all_per_sample, output_dir)

    print(f"\n  Plots saved to: {output_dir}")
