"""Benchmark kNN discovery across neighbor counts and time-series lengths.

Run from the repository root:
    python -m benchmarks.run_knn_sensitivity --profile smoke

The benchmark uses the shared dataset generator and edge-scoring functions.
Each seed defines a time series whose prefixes are evaluated at every
sample length and neighbor count.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy

from benchmarks.adapters import generate_case
from benchmarks.metrics import normalize_ce_edges, score_edges
from causationentropy.core.discovery import discover_network

PROFILES = {
    "smoke": {
        "n": 3,
        "lengths": [80],
        "ks": [3],
        "seeds": [0],
        "shuffles": 5,
    },
    "pilot": {
        "n": 5,
        "lengths": [200, 500],
        "ks": [3, 5, 10],
        "seeds": [0, 1, 2],
        "shuffles": 30,
    },
    "full": {
        "n": 5,
        "lengths": [200, 500, 1000, 2000],
        "ks": [3, 5, 10],
        "seeds": list(range(30)),
        "shuffles": 200,
    },
}

RESULT_FIELDS = [
    "n",
    "T",
    "k",
    "seed",
    "max_lag",
    "n_shuffles",
    "metric",
    "runtime_s",
    "TP",
    "FP",
    "FN",
    "precision",
    "recall",
    "F1",
    "cross_TP",
    "cross_FP",
    "cross_FN",
    "cross_TN",
    "cross_precision",
    "cross_recall",
    "cross_F1",
    "cross_FPR",
    "self_TP",
    "self_FP",
    "self_FN",
    "n_truth",
    "n_predicted",
]


def _git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return precision, recall, f1


def aggregate(rows: list[dict], bootstrap_draws: int) -> list[dict]:
    """Micro-average cross-link recovery and bootstrap over independent seeds."""
    groups: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        key = (int(row["n"]), int(row["T"]), int(row["k"]))
        groups[key].append(row)

    summaries = []
    for (n, length, k), group in sorted(groups.items()):
        group = sorted(group, key=lambda row: int(row["seed"]))
        tp = sum(int(row["cross_TP"]) for row in group)
        fp = sum(int(row["cross_FP"]) for row in group)
        fn = sum(int(row["cross_FN"]) for row in group)
        tn = sum(int(row["cross_TN"]) for row in group)
        precision, recall, f1 = _precision_recall_f1(tp, fp, fn)

        ci_low = None
        ci_high = None
        if len(group) > 1 and bootstrap_draws > 0:
            counts = np.array(
                [
                    [
                        int(row["cross_TP"]),
                        int(row["cross_FP"]),
                        int(row["cross_FN"]),
                    ]
                    for row in group
                ],
                dtype=np.int64,
            )
            # Reuse the same resampled positions for each balanced k/T group.
            rng = np.random.default_rng(2026)
            sampled = rng.integers(0, len(group), size=(bootstrap_draws, len(group)))
            totals = counts[sampled].sum(axis=1)
            denominators = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
            f1_samples = np.divide(
                2 * totals[:, 0],
                denominators,
                out=np.zeros(bootstrap_draws, dtype=float),
                where=denominators != 0,
            )
            ci_low, ci_high = (
                float(value) for value in np.quantile(f1_samples, [0.025, 0.975])
            )

        runtimes = [float(row["runtime_s"]) for row in group]
        summaries.append(
            {
                "n": n,
                "T": length,
                "k": k,
                "n_seeds": len(group),
                "cross_TP": tp,
                "cross_FP": fp,
                "cross_FN": fn,
                "cross_TN": tn,
                "cross_precision": precision,
                "cross_recall": recall,
                "cross_F1": f1,
                "cross_F1_ci95_low": ci_low,
                "cross_F1_ci95_high": ci_high,
                "cross_FPR": fp / (fp + tn) if fp + tn else 0.0,
                "runtime_median_s": statistics.median(runtimes),
            }
        )
    return summaries


def summary_markdown(summary: list[dict], config: dict) -> str:
    """Render cross-link recovery metrics using exact source/target/lag matches."""
    lines = [
        "# kNN discovery sensitivity (issue #55)",
        "",
        f"Profile: `{config['profile']}`; "
        f"metric: `{config['metric']}`; "
        f"max lag: {config['max_lag']}; "
        f"permutation shuffles: {config['n_shuffles']}.",
        "",
        "Cross-links are scored by exact (source, target, lag) matches. "
        "Counts are pooled across seeds before computing precision, recall, "
        "F1, and false-positive rate. Runtime is the median across seeds. "
        "Confidence intervals use a percentile bootstrap over seeds.",
        "",
        "| n | T | k | seeds | TP | FP | FN | precision | recall | F1 | "
        "95% CI (F1) | FPR | median runtime (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in summary:
        ci = (
            f"[{row['cross_F1_ci95_low']:.3f}, " f"{row['cross_F1_ci95_high']:.3f}]"
            if row["cross_F1_ci95_low"] is not None
            else "N/A"
        )
        lines.append(
            f"| {row['n']} | {row['T']} | {row['k']} | {row['n_seeds']} | "
            f"{row['cross_TP']} | {row['cross_FP']} | {row['cross_FN']} | "
            f"{row['cross_precision']:.3f} | {row['cross_recall']:.3f} | "
            f"{row['cross_F1']:.3f} | {ci} | {row['cross_FPR']:.3f} | "
            f"{row['runtime_median_s']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark-results/issue55")
    )
    parser.add_argument("--n-bootstraps", type=int, default=1000)
    parser.add_argument("--allow-large-run", action="store_true")
    args = parser.parse_args(argv)

    if args.profile == "full" and not args.allow_large_run:
        parser.error("the full profile requires --allow-large-run")
    if args.n_bootstraps < 0:
        parser.error("--n-bootstraps must be nonnegative")
    if args.output.exists():
        if not args.output.is_dir() or any(args.output.iterdir()):
            parser.error(f"output directory must be empty: {args.output}")

    profile = PROFILES[args.profile]
    max_lag = 2
    alpha = 0.05
    metric = "chebyshev"
    signature = inspect.signature(discover_network)
    required = {"k_means", "metric"}
    if not required.issubset(signature.parameters):
        parser.error("discover_network is missing a required benchmark parameter")

    for length in profile["lengths"]:
        if any(k >= length - max_lag for k in profile["ks"]):
            parser.error("neighbor count must be below the effective sample size")

    git_status = _git_output("status", "--porcelain")
    config = {
        "issue": 55,
        "profile": args.profile,
        "n": profile["n"],
        "lengths": profile["lengths"],
        "ks": profile["ks"],
        "seeds": profile["seeds"],
        "n_shuffles": profile["shuffles"],
        "method": "standard",
        "information": "knn",
        "metric": metric,
        "max_lag": max_lag,
        "alpha_forward": alpha,
        "alpha_backward": alpha,
        "n_jobs": 1,
        "sample_design": "nested time-series prefixes per seed",
        "bootstrap_draws": args.n_bootstraps,
        "git_head": _git_output("rev-parse", "HEAD"),
        "git_dirty": bool(git_status) if git_status is not None else None,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    rows = []
    with (args.output / "run_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()

        for seed in profile["seeds"]:
            series, truth, node_order = generate_case(
                profile["n"], max(profile["lengths"]), seed
            )
            for length in profile["lengths"]:
                data = series[:length].copy()
                for k in profile["ks"]:
                    start = time.perf_counter()
                    graph = discover_network(
                        data,
                        method="standard",
                        information="knn",
                        max_lag=max_lag,
                        metric=metric,
                        k_means=k,
                        alpha_forward=alpha,
                        alpha_backward=alpha,
                        n_shuffles=profile["shuffles"],
                        n_jobs=1,
                        random_state=seed,
                    )
                    runtime = time.perf_counter() - start

                    predicted = normalize_ce_edges(graph, node_order)
                    scores = score_edges(truth, predicted)
                    total_cross = profile["n"] * (profile["n"] - 1) * max_lag
                    cross_tn = total_cross - (
                        scores["cross_TP"] + scores["cross_FP"] + scores["cross_FN"]
                    )
                    if cross_tn < 0:
                        raise ValueError("discovered edges exceed the scoring universe")
                    negatives = cross_tn + scores["cross_FP"]
                    row = {
                        "n": profile["n"],
                        "T": length,
                        "k": k,
                        "seed": seed,
                        "max_lag": max_lag,
                        "n_shuffles": profile["shuffles"],
                        "metric": metric,
                        "runtime_s": runtime,
                        **scores,
                        "cross_TN": cross_tn,
                        "cross_FPR": (
                            scores["cross_FP"] / negatives if negatives else 0.0
                        ),
                    }
                    writer.writerow({field: row[field] for field in RESULT_FIELDS})
                    file.flush()
                    rows.append(row)
                    print(
                        f"seed={seed} T={length} k={k} "
                        f"TP={scores['cross_TP']} "
                        f"FP={scores['cross_FP']} "
                        f"FN={scores['cross_FN']} "
                        f"F1={scores['cross_F1']:.3f} "
                        f"runtime={runtime:.1f}s",
                        flush=True,
                    )

    summary = aggregate(rows, args.n_bootstraps)
    (args.output / "summary.json").write_text(
        json.dumps({"config": config, "summary": summary}, indent=2) + "\n",
        encoding="utf-8",
    )
    report = summary_markdown(summary, config)
    (args.output / "SUMMARY.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Saved results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
