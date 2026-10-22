"""Tests for the kNN discovery sensitivity benchmark."""

import csv
import json

import networkx as nx
import numpy as np
import pytest

pytest.importorskip("benchmarks")

from benchmarks import run_knn_sensitivity as sensitivity


def test_aggregate_pools_seed_counts():
    rows = [
        {
            "n": 3,
            "T": 80,
            "k": 3,
            "seed": 0,
            "cross_TP": 1,
            "cross_FP": 1,
            "cross_FN": 1,
            "cross_TN": 9,
            "runtime_s": 2.0,
        },
        {
            "n": 3,
            "T": 80,
            "k": 3,
            "seed": 1,
            "cross_TP": 2,
            "cross_FP": 2,
            "cross_FN": 0,
            "cross_TN": 8,
            "runtime_s": 4.0,
        },
    ]
    summary = sensitivity.aggregate(rows, bootstrap_draws=100)
    assert len(summary) == 1
    result = summary[0]
    assert (result["cross_TP"], result["cross_FP"], result["cross_FN"]) == (
        3,
        3,
        1,
    )
    assert result["cross_F1"] == pytest.approx(6 / 10)
    assert result["cross_FPR"] == pytest.approx(3 / 20)
    assert result["runtime_median_s"] == pytest.approx(3)
    assert 0 <= result["cross_F1_ci95_low"] <= result["cross_F1_ci95_high"] <= 1


def test_smoke_writes_scored_results(monkeypatch, tmp_path):
    calls = []

    def fake_generate_case(n, length, seed):
        assert (n, length, seed) == (3, 80, 1)
        return np.zeros((length, n)), {(0, 1, 1)}, ["X0", "X1", "X2"]

    def fake_discover_network(
        data,
        method,
        information,
        max_lag,
        metric,
        k_means,
        alpha_forward,
        alpha_backward,
        n_shuffles,
        n_jobs,
        random_state,
    ):
        calls.append((data.shape, method, information, metric, k_means))
        graph = nx.MultiDiGraph()
        graph.add_edge("X0", "X1", lag=1)
        graph.add_edge("X2", "X1", lag=2)
        return graph

    monkeypatch.setattr(sensitivity, "generate_case", fake_generate_case)
    monkeypatch.setattr(sensitivity, "discover_network", fake_discover_network)
    output = tmp_path / "benchmark"
    assert (
        sensitivity.main(
            ["--profile", "smoke", "--output", str(output), "--n-bootstraps", "0"]
        )
        == 0
    )
    assert calls == [((80, 3), "standard", "knn", "chebyshev", 3)]
    with (output / "run_results.csv").open(newline="", encoding="utf-8") as file:
        results = list(csv.DictReader(file))
    assert len(results) == 1
    assert (results[0]["cross_TP"], results[0]["cross_FP"]) == ("1", "1")
    assert results[0]["cross_TN"] == "10"
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["summary"][0]["cross_recall"] == pytest.approx(1.0)
    assert "95% CI" in (output / "SUMMARY.md").read_text(encoding="utf-8")

def test_paired_sweep_reuses_seed_series_across_k_and_lengths(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sensitivity.PROFILES,
        "paired_test",
        {
            "n": 3,
            "lengths": [40, 80],
            "ks": [3, 5],
            "seeds": [0, 1],
            "shuffles": 5,
        },
    )
    generated = []
    observed = []

    def fake_generate_case(n, length, seed):
        generated.append((n, length, seed))
        series = np.tile(np.arange(length)[:, None] + seed * 100, (1, n))
        return series, {(0, 1, 1)}, ["X0", "X1", "X2"]

    def fake_discover_network(data, metric, k_means, **kwargs):
        observed.append(
            (kwargs["random_state"], len(data), k_means, data.copy())
        )
        return nx.MultiDiGraph()

    monkeypatch.setattr(sensitivity, "generate_case", fake_generate_case)
    monkeypatch.setattr(sensitivity, "discover_network", fake_discover_network)

    output = tmp_path / "paired_sweep"
    assert (
        sensitivity.main(
            [
                "--profile",
                "paired_test",
                "--output",
                str(output),
                "--n-bootstraps",
                "0",
            ]
        )
        == 0
    )

    assert generated == [(3, 80, 0), (3, 80, 1)]
    assert len(observed) == 8
    for seed in (0, 1):
        cases = {
            (length, k): data
            for run_seed, length, k, data in observed
            if run_seed == seed
        }
        assert set(cases) == {(40, 3), (40, 5), (80, 3), (80, 5)}
        for k in (3, 5):
            assert np.array_equal(cases[(40, k)], cases[(80, k)][:40])
        assert np.array_equal(cases[(80, 3)], cases[(80, 5)])

    with (output / "run_results.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 8
    assert all(int(row["cross_FN"]) == 1 for row in rows)
    assert all(int(row["cross_TN"]) == 11 for row in rows)


def test_smoke_profile_has_ground_truth_cross_link():
    profile = sensitivity.PROFILES["smoke"]
    data, truth, node_order = sensitivity.generate_case(
        profile["n"], max(profile["lengths"]), profile["seeds"][0]
    )
    assert data.shape == (profile["lengths"][0], profile["n"])
    assert node_order == ["X0", "X1", "X2"]
    assert any(source != target for source, target, _ in truth)
