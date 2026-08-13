---
name: New Benchmark
about: Checklist for introducing a new benchmark to MLPerf Training
---

## New Benchmark: [Benchmark Name]

Task force lead: @<!-- mention -->
Tracking issue: #<!-- issue number -->

### Initial reference code ([roadmap](https://github.com/mlcommons/policies/blob/master/submission_rules.adoc#43-benchmark-roadmap-schedule))

- [ ] [Review guidelines on how to create a good MLPerf Training reference](https://github.com/mlcommons/training_policies/blob/master/CONTRIBUTING.md#requirements-for-an-mlperf-training-reference)
- [ ] Finalize dataset
- [ ] Finalize model architecture
- [ ] Finalize reference framework
- [ ] Finalize platform/hardware that the reference would be implemented in, add it to the [approved list](https://github.com/mlcommons/training_policies/blob/master/CONTRIBUTING.md#general)
- [ ] Finalize reference precision and generate initial loss curves to understand training behavior. Decide the rough benchmarking region and figure out whether the benchmark should start from randomly initialized weights or from a previously trained checkpoint.
- [ ] Finalize batch sizes for RCPs and get good hyperparameters at these batch sizes (need at least 3 batch sizes covering small, medium, and large ranges)
- [ ] Finalize evaluation metric and dataset
- [ ] Finalize which hyperparameters are unconstrained for submitters to modify

### Finalize code

- [ ] Upload processed dataset to MLC bucket (reach out to support@mlcommons.org for write access and download instructions)
- [ ] Create PR with initial codebase to [mlcommons/training](https://github.com/mlcommons/training). Must include run instructions and a README following the [benchmark readme template](https://github.com/mlcommons/training/blob/master/benchmark_readme_template.md). Ensure your [GitHub handle](https://github.com/mlcommons/training/blob/master/CONTRIBUTING.md) is set up to contribute to MLC repositories.
- [ ] Add checkpoint + resume capability to the reference codebase

### Generate RCPs

- [ ] [Choose target accuracy](https://github.com/mlcommons/training_policies/blob/master/CONTRIBUTING.md#how-to-choose-the-target-accuracy)
- [ ] [Choose eval frequency](https://github.com/mlcommons/training_policies/blob/master/CONTRIBUTING.md#how-to-choose-the-evaluation-frequency)
- [ ] [Choose number of submission runs](https://github.com/mlcommons/training_policies/blob/master/CONTRIBUTING.md#how-to-choose-the-number-of-submission-runs-n-needed)
- [ ] [Generate RCPs](https://github.com/mlcommons/training_policies/blob/master/CONTRIBUTING.md#some-things-to-note-while-generating-reference-convergence-points-rcps)
- [ ] [Add new RCPs to the logging repo](https://github.com/mlcommons/logging/tree/master/mlperf_logging/rcp_checker)
- [ ] Commit RCP log files to the reference implementation folder in training/

### Add new benchmark details

- [ ] Update logging repo to include compliance checks for the new benchmark (example [PR359](https://github.com/mlcommons/logging/pull/359))
- [ ] Update [training rules](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc) by adding the new benchmark in all relevant tables:
  - [ ] [Benchmarks](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc#3-benchmarks)
  - [ ] [Division](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc#41-closed-division)
  - [ ] [Hyperparameters](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc#91-hyperparameters)
  - [ ] [Quality](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc#94-quality-measure)
  - [ ] [Results](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc#12-benchmark-results)
  - [ ] [RCP rules](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc#131-rcp-rules-and-guidelines)
  - [ ] Add any benchmark-specific rules to the [appendix](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc#131-rcp-rules-and-guidelines)
- [ ] Update the [compatibility table](https://github.com/mlcommons/policies/blob/master/MLPerf_Compatibility_Table.adoc#training) with the new benchmark
- [ ] Create a new benchmark presentation deck and present to the Training WG
- [ ] Create an initial blog write-up with technical details. Once ready, MLC will bring in a technical writer to polish and publish before the first round. Sample blogs: [flux.1](https://mlcommons.org/2025/10/training-flux1/), [llama31_405b](https://mlcommons.org/2025/05/training-llama31405b/)
