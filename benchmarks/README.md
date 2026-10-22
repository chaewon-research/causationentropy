# Benchmark harness for issue #18

Reproducible runtime/accuracy comparison: CausationEntropy Gaussian oCSE
vs Tigramite PCMCI + ParCorr on identical synthetic data.

## Quick start

```bash
python -m benchmarks.run --profile quick --output benchmark-results
python -m benchmarks.run --profile full --output benchmark-results
```

## Conventions (answers to the open questions on #18)

* **Generator:** `linear_stochastic_gaussian_process(rho=0.7, p=0.2)`,
  explicit `erdos_renyi_graph(n, p, seed, directed=True)` ground truth.
  With `p=0.2`, `n=20` yields ~77 true edges, matching the issue report.
* **Edge matching:** exact `(source, target, lag)`; all ground-truth
  edges are lag 1, so lag-2 discoveries count as false positives.
* **Self-lags:** reported separately from cross-links. Standard oCSE
  conditions on the target's own lags, so it cannot discover self-lags
  by construction; counting them in the main score would penalise it on
  generators with autoregression. Cross-links are the main metric.
* **CE config:** `method="standard"`, Gaussian, serial (`n_jobs=1`),
  `alpha_forward=alpha_backward=0.05`, profile-dependent `n_shuffles`.
* **PCMCI config:** `ParCorr`, `tau_max=max_lag`, `pc_alpha=0.05`.
  Only directed links at lag >= 1 count; undirected/conflicting and
  contemporaneous links are ignored.
* **Significance comparability:** end-to-end defaults (permutation vs
  analytical). An analytical Gaussian backend is out of scope here.
* **Repetitions:** fixed seeds; runtime = median, accuracy =
  micro-averaged (TP/FP/FN summed across seeds).
* **Execution:** serial comparison is primary. Parallel (`n_jobs`) results
  from #28 can be added as a separate table later.
* **Profiles:** `quick` (smoke), `small` (accuracy signal),
  `full` (the issue table: `n=5,10,20`, `T=500`, 5 seeds).

## Outputs

`--output` receives `run_results.csv` (per-seed rows), `summary.json`
(config + aggregates), and `SUMMARY.md` (the comparison table).
Raw result directories are local artifacts (see `.gitignore`); copy the
final table to `benchmarks/RESULTS_issue18.md` when publishing.

## CI

The full benchmark does not run in CI. CI covers the generators,
adapters, and metrics on a tiny dataset via
`causationentropy/tests/test_benchmark_harness.py`.

## kNN discovery sensitivity (issue #55)

This separate entry point sweeps the Chebyshev kNN discovery estimator over
neighbor counts `k`, time-series lengths `T`, and random seeds. It reuses
the generator and exact-edge scoring rules described above.

Run from the repository root, using a fresh output directory for each run:

```bash
python -m benchmarks.run_knn_sensitivity --profile smoke --output benchmark-results/issue55-smoke
python -m benchmarks.run_knn_sensitivity --profile pilot --output benchmark-results/issue55-pilot
python -m benchmarks.run_knn_sensitivity --profile full --allow-large-run --output benchmark-results/issue55-full
```

All profiles use `method="standard"`, `information="knn"`,
`metric="chebyshev"`, `max_lag=2`, both significance levels at
0.05, and `n_jobs=1`. Each seed generates one longest time series;
the shorter conditions use its prefixes, and every `k` is evaluated
on the same observations at a given `T`.

* `smoke`: one seed, `T=80`, `k=3`, five shuffles. Checks that the
  pipeline and outputs work; not an accuracy estimate.
* `pilot`: three seeds, `T=200,500`, `k=3,5,10`, 30 shuffles.
  Exploratory only.
* `full`: 30 seeds, `T=200,500,1000,2000`, `k=3,5,10`,
  200 shuffles. This is computationally expensive and requires
  `--allow-large-run`.

The output directory contains `config.json`, per-run
`run_results.csv`, `summary.json`, and `SUMMARY.md`. The primary
scores are cross-link TP/FP/FN, precision, recall, F1, and
false-positive rate, with exact source/target/lag matching. Results
are micro-averaged across seeds; runtime is the median. The reported
95% F1 intervals resample independent seeds, not individual edges
or overlapping time-series windows.

For final results, run against the finalized kNN estimator from #53
and retain `config.json` so the evaluated commit and parameter
settings are identifiable.
