# Reference Convergence Points (RCP)

**Status: placeholder — RCPs not yet generated. Intentionally left blank.**

This directory will hold the Reference Convergence Points for the
recommendation_v4 (HSTU / yambda-5b) benchmark once convergence runs are
complete.

Per the MLPerf Training
[CONTRIBUTING guidance](https://github.com/mlcommons/training_policies/blob/master/CONTRIBUTING.md)
("Some things to note while generating reference convergence points"):

- Use FP32 or BF16 precision and record the exact precision used in the RCP JSON.
- Generate RCPs for at least **3 reasonable batch sizes**.
- Run RCPs with an eval frequency **higher** than the chosen benchmark eval
  frequency (more data points for picking the target accuracy).
- Run at least **2N seeds**, where N = number of submission runs.

The convergence target for this benchmark is **eval AUC >= 0.80275** (see
[../README.MD](../README.MD) §9). The RCP JSON files and convergence-curve plots
(samples-to-converge vs. batch size / seed) will be committed here.

## TODO

- [ ] Run >= 2N-seed convergence sweeps at >= 3 batch sizes.
- [ ] Record precision (FP32/BF16) per the rules.
- [ ] Add `rcp_<batchsize>.json` files in the mlperf_logging RCP format.
- [ ] Add convergence-curve plots and the chosen target-accuracy justification.
