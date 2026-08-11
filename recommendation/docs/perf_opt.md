# Performance Optimizations — MI350X HSTU / OneTrans (yambda-5b, bs=1024, TRITON)

Performance work for the 8× MI350X HSTU ranker on `yambda-5b` at `batch_size=1024`
with the **TRITON** HSTU kernel and bf16 training. Companion to
[`training_recipe.md`](./training_recipe.md) (environment + reproduction).

Throughput numbers are global samples/sec across 8 GPUs (`global_sps`), measured
at steady state (instantaneous, computed from consecutive logged steps).

---

## LN-dropout: multi-row, separated-RNG path on MI350

### What

`_ln_mul_dropout_*` has two kernel variants:

- **legacy** — single program per row, RNG fused inline (`_ln_mul_dropout_fwd`).
- **separated-RNG** — multiple rows per program, dropout mask precomputed once
  and reused by the backward (`_ln_mul_dropout_fwd_rng` /
  `_ln_mul_dropout_bwd_dx_du_rng`).

The separated path was previously gated to Blackwell only (`is_sm100_plus()`).
MI350X (`gfx950`) benefits from the same structure, so the gate now also enables
it on MI350.

### Where

| file | change |
|---|---|
| `ops/utils.py` | `is_amd_mi350()` (gfx950 detect) + `use_separated_rng_ln_mul_dropout()` gate |
| `ops/triton/triton_hstu_linear.py` | dispatch LN-dropout fwd to the separated-RNG path when the gate is true |

```python
# ops/utils.py
def use_separated_rng_ln_mul_dropout() -> bool:
    return is_sm100_plus() or is_amd_mi350()
```

### Perf

**+5.6% end-to-end → 14,222 global sps** (separated-RNG vs legacy fused, identical
config, full boost clocks — see the caveat below).

---

## Caveat — GPU clock lock can mask all perf changes

A node-level GPU clock lock will silently invalidate any benchmark on this
machine, so check it before trusting numbers.

During this work all 8 GPUs were stuck in **`perf_determinism`** performance
level at **sclk 1093 MHz** (DPM level 1) while the real max is **2200 MHz**
(level 2) — despite 100% utilization, ~370 W of power headroom (629 / 1000 W),
and low temps (~50 °C). This was **not** thermal/power throttling; it was
leftover node state from a prior job.

Effect: a **uniform ~1.87× slowdown of every Triton compute kernel**
(`2200 / 1093 ≈ 2.0×`), including kernels unrelated to any code change. It made
the LN-dropout fix above look like a regression until the clock state was found.

### Detect + fix

```bash
rocm-smi --showperflevel          # expect "auto", not perf_determinism/manual/low
rocm-smi -d 0 --showclocks        # expect sclk ~2000+ MHz under load
rocm-smi --setperflevel auto      # restore boost
```

`scripts/launch_slurm.sh` (worker phase) now logs the perf level + a live `sclk` sample on
every launch, auto-restores `auto` if it finds a `perf_determinism`/`manual`/`low`
lock, and warns (to reset from the host) if it lacks permission inside the
container. **Always sanity-check `sclk ≈ 2000+ MHz` before trusting a benchmark.**
