---
name: New Round Introduction
about: Checklist for introducing a new MLPerf Training round
---

## New Round: v[X.Y]

Round chair: @<!-- mention -->
Prior round: v[<!-- e.g. 6.0 -->]
Submission deadline: <!-- date -->
Tracking issue: #<!-- issue number -->

Use this checklist when the Training WG adopts a new round. For **brand-new benchmarks** added in the same round, also complete the [New Benchmark](new_benchmark.md) checklist.

Key references:

- [Submission rules & roadmap](https://github.com/mlcommons/policies/blob/master/submission_rules.adoc#43-benchmark-roadmap-schedule)
- [Training rules](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc)
- [Compatibility table (Training)](https://github.com/mlcommons/policies/blob/master/MLPerf_Compatibility_Table.adoc#training)
- [MLPerf Training benchmarks page](https://mlcommons.org/benchmarks/training/)

---

### 1. Confirm round scope with the Training WG

- [ ] Finalize benchmark list for v[X.Y]: carrying forward, new, and retired benchmarks
- [ ] Confirm submission deadline and ruleset version (e.g. `6.1.0`)
- [ ] Confirm any benchmark-specific rule changes that carry into the new round

---

### 2. [mlcommons/training](https://github.com/mlcommons/training) — reference implementations

- [ ] Move retired benchmark folders under [`retired_benchmarks/`](https://github.com/mlcommons/training/tree/master/retired_benchmarks)
- [ ] Add a new round section to [README.md](https://github.com/mlcommons/training/blob/master/README.md) with:
  - [ ] Submission deadline
  - [ ] Benchmark table (model, reference path, framework, dataset, parameter count)
  - [ ] Links to each active reference implementation
- [ ] For **new** benchmarks: merge reference code and follow the [New Benchmark](new_benchmark.md) checklist (dataset upload, RCP logs, README, etc.)
- [ ] For **carrying-forward** benchmarks: confirm reference code still matches the closed-division rules for the new round

---

### 3. [mlcommons/training_policies](https://github.com/mlcommons/training_policies) — rules document

Update [training_rules.adoc](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc):

- [ ] Update document date at the top of the file
- [ ] For **carrying-forward** benchmarks: set `Latest version available` to `v[X.Y]` in all active tables:
  - [ ] [Benchmarks / datasets](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc#3-benchmarks)
  - [ ] [Closed division models & targets](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc#41-closed-division)
  - [ ] [Hyperparameters](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc#91-hyperparameters)
  - [ ] [Quality / evaluation frequency](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc#94-quality-measure)
  - [ ] [Minimum number of runs](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc#12-benchmark-results)
- [ ] For **new** benchmarks: add rows to each table above plus any benchmark-specific appendix rules
- [ ] For **retired** benchmarks:
  - [ ] Remove from active tables
  - [ ] Move rules to [Appendix: Deprecated benchmarks](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc#appendix-deprecated-benchmarks)
  - [ ] Add a `v[X.Y]` appendix section if the round introduced round-specific exceptions (see [v6.0 Specific Rules](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc#appendix-v60-specific-rules) for an example)

---

### 4. [mlcommons/logging](https://github.com/mlcommons/logging) — checkers & summarizer

- [ ] Add `training_[X.Y].0/` compliance-checker configs under [`mlperf_logging/compliance_checker/`](https://github.com/mlcommons/logging/tree/master/mlperf_logging/compliance_checker)
  - [ ] Copy forward `common.yaml` and per-benchmark closed/open YAMLs from the prior round; update for retired/new benchmarks
- [ ] Add or update RCP JSON files under [`mlperf_logging/rcp_checker/training_[X.Y].0/`](https://github.com/mlcommons/logging/tree/master/mlperf_logging/rcp_checker)
- [ ] Update [`mlperf_logging/benchmark_meta.py`](https://github.com/mlcommons/logging/blob/master/mlperf_logging/benchmark_meta.py):
  - [ ] Allowed benchmark list for the new ruleset
  - [ ] Required result-file counts per benchmark
- [ ] Add [`scripts/verify_for_v[X.Y]_training.sh`](https://github.com/mlcommons/logging/tree/master/scripts) (copy from prior round and update ruleset version)
- [ ] Update [`mlperf_logging/result_summarizer/config.yaml`](https://github.com/mlcommons/logging/blob/master/mlperf_logging/result_summarizer/config.yaml) with the new ruleset column/benchmark list
- [ ] Tag a new logging release once changes are merged (submitters pin to this tag)

---

### 5. [mlcommons/policies](https://github.com/mlcommons/policies) — compatibility table

Update [MLPerf_Compatibility_Table.adoc](https://github.com/mlcommons/policies/blob/master/MLPerf_Compatibility_Table.adoc#training):

- [ ] Add a column for Training v[X.Y]
- [ ] Mark supported benchmarks for the new round
- [ ] Mark all retired benchmarks as unsupported in the new column 

---



---

### 6. Website & communications

Email [marketing@mlcommons.org](mailto:marketing@mlcommons.org) and ask them to update the [Training benchmarks page](https://mlcommons.org/benchmarks/training/):

- [ ] **Carrying-forward benchmarks:** update "Latest available version" to v[X.Y]
- [ ] **Retired benchmarks:** leave "Latest available version" unchanged
- [ ] **New benchmarks:** add a row for each new benchmark
- [ ] Sort rows by "Latest available version" (newest round additions at the top)

Optional but recommended before the first submission deadline:

- [ ] Publish benchmark blog posts on [mlcommons.org](https://mlcommons.org/benchmarks/training/) for new or materially changed benchmarks
- [ ] Present round changes at a Training WG meeting
