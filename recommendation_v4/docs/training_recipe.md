# Training Recipe

Reproducible environment + configuration for training HSTU / DLRM-v3 on the
`yambda-5b` dataset.

---

## MI350X

Single-node, 8× AMD **Instinct MI350X** (`gfx950`, ~288 GiB HBM3e each), HSTU
ranker on `yambda-5b` with the **TRITON** HSTU kernel and **bf16**
mixed-precision training.

### Hardware / host

| item | value |
|---|---|
| GPUs | 8× AMD Instinct MI350X (`gfx950`, ROCm 7.2.1) |
| Host CPU | AMD EPYC 9655 96-Core (192 cores × 2 threads) |

### Container image

```
rocm/primus:v26.3
```

The image's native PyTorch is kept as-is and must not be reinstalled — it is
the ROCm-matched build used by triton/fbgemm.

### Dependency versions

Aligned with the B200 path: same torch major.minor, same torchrec commit,
same fbgemm SHA. The image's native torch / torchvision / torchaudio /
torchrec / fbgemm_gpu are all replaced; only the image's triton stays.

| package | version | install |
|---|---|---|
| **torch** | `2.12.0+rocm7.2` | `pip install --upgrade --no-deps --index-url https://download.pytorch.org/whl/rocm7.2 torch==2.12.0+rocm7.2` |
| **torchvision** | `0.27.0+rocm7.2` | `pip install --upgrade --no-deps --index-url https://download.pytorch.org/whl/rocm7.2 torchvision` — ABI must match torch 2.12 |
| **torchaudio** | `2.11.0+rocm7.2` | `pip install --upgrade --no-deps --index-url https://download.pytorch.org/whl/rocm7.2 torchaudio` — ABI must match torch 2.12 |
| **triton** | `3.6.0` | image native, unchanged |
| **fbgemm_gpu** | `fbgemm_gpu_nightly_rocm-2026.6.2` (built from FBGEMM commit `10b77573`, same SHA as the B200 path) for `gfx950` | rebuild from source against the replaced torch. Build command: `python setup.py -j 32 bdist_wheel --build-target=default --build-variant=rocm -DHIP_ROOT_DIR=/opt/rocm -DAMDGPU_TARGETS=gfx950` |
| **torchrec** | `1.7.0a0+bf55480` (git tag `v2026.06.01.00`) | `pip install --force-reinstall --no-deps "git+https://github.com/pytorch/torchrec.git@v2026.06.01.00"` |
| **polars-u64-idx** | `1.33.1` | 64-bit row index — `yambda-5b` has > 4.29 B rows. Installed from a pre-staged local tarball by `scripts/launch_smoke_8gpu.sh` |

### Training configuration

From `generative_recommenders/dlrm_v3/train/gin/yambda_5b.gin`:

| parameter | value | gin binding |
|---|---|---|
| num_workers (dataloader) | 4 | `make_train_test_dataloaders.num_workers` |
| prefetch_factor | 8 | `make_train_test_dataloaders.prefetch_factor` |
| num_blocks | 1 | `make_train_test_dataloaders.num_blocks` |
| train_split_percentage | 0.90 | `make_train_test_dataloaders.train_split_percentage` |
| history_length (per-sample UIH budget) | 2039 | `get_dataset.history_length` |
| max_seq_len (attention budget) | 2048 | `get_hstu_configs.max_seq_len` |
| bf16 training | True | `make_model.bf16_training` |
| HBM cap (per GPU) | 260 GiB | `make_optimizer_and_shard.hbm_cap_gb` (env `HBM_CAP_GB`) |
| **triton autotune pinning** | **False (pinned)** | `apply_env_bootstrap.TRITON_FULL_AUTOTUNE` |
| dense optimizer | Adam, lr 1e-3, betas (0.95, 0.999), eps 1e-8 | `dense_optimizer_factory_and_class.*` |
| sparse optimizer | RowWiseAdagrad, lr 1e-3, betas (0.95, 0.999), eps 1e-8 | `sparse_optimizer_factory_and_class.*` |
| world_size | 8 | `MetricsLogger.world_size` |

Effective global batch = `batch_size × world_size = 32 × 8 = 256` samples/step.

### Environment variables

| var | value | purpose |
|---|---|---|
| `HSTU_HAMMER_KERNEL` | `TRITON` | fast HSTU kernel (vs `PYTORCH` fallback) |
| `DLRM_DATA_PATH` | dataset root | overrides gin default `/apps/chcai/dlrm_data` |
| `HBM_CAP_GB` | (optional) | embedding planner HBM budget per GPU |
| `RUN_NAME` | run id | results dir → `results/<RUN_NAME>/` |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | allocator headroom |
| `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | rank visibility |

`TRITON_FULL_AUTOTUNE` is set automatically by the gin-driven bootstrap
(`generative_recommenders.dlrm_v3.train._env_bootstrap.apply_env_bootstrap`),
which runs in `train_ranker._main_func` BEFORE the triton kernel modules
import — so the gin file is the source of truth.

### Measured performance

| variant | steady-state ms/step | global sps | epoch ETA (3.23B anchors) |
|---|---|---|---|
| nightly + fp32 + PYTORCH attn (baseline) | ~190 | ~1340 | ~28 d |
| nightly + bf16 + TRITON attn | ~93 | ~2787 | ~13.4 d |
| primus + bf16 + TRITON attn | ~67.5 | ~3793 | ~9.9 d |
| primus + fbgemm HEAD + bf16 + TRITON, autotune drift | ~53 fast / ~70 slow | 3700–4860 | 7.7–10.2 d |
| **primus + fbgemm HEAD + bf16 + TRITON + pinning (default)** | **~52** | **~4970** | **~7.6 d** |

The "pinning" line is the deterministic per-cold-start equilibrium —
three layer-norm / jagged triton kernels have two stable autotune winners
and the pin forces the fast one every run.

### Known pitfalls

- The image ships `fbgemm_gpu==2026.5.14`. The wheel built from FBGEMM HEAD
  (`2026.6.1`) is required for the 70 → 52 ms step. Build inside the
  container so the wheel links against the image's native torch.
- Stock `polars` silently overflows on `yambda-5b` (> 4.29 B rows); always
  use `polars-u64-idx`.
- When changing shape (batch size, history length), GPU, or triton/torch
  version, flip `apply_env_bootstrap.TRITON_FULL_AUTOTUNE = True` and run
  with `TRITON_PRINT_AUTOTUNING=1` to re-capture winners, then update the
  pinned configs at the `pinned_or_full(...)` call sites in
  `generative_recommenders/ops/triton/`.
- Do not run with bf16 on the `PYTORCH` HSTU attention backend at our
  sequence length — `pt_hstu_attention`'s QK einsum backward overflows in
  bf16 at N > 1k and produces NaN at step 1. bf16 is only safe with TRITON.

---

## B200

Single-node, 8× NVIDIA **B200** (Blackwell, `sm_100`, ~183 GiB HBM each), HSTU
ranker on `yambda-5b` with the **TRITON** HSTU kernel and **bf16** mixed-precision
training.

### Hardware / host

| item | value |
|---|---|
| GPUs | 8× NVIDIA B200 (`sm_100`, compute capability 10.0) |
| Host driver | 580.159.03 (reports CUDA 13.0) |
| Forward-compat userspace driver | `libcuda.so.595.58.03` (CUDA 13.2.1; engaged automatically by the NGC image) |

### Container image

```
nvcr.io/nvidia/pytorch:26.04-py3
```

Digest: `sha256:192d749b4d773610ec9e01c0443a9df545d196c412b7b8fd33bfa3da362a49e7`

The image's native PyTorch is kept as-is and must not be reinstalled (so CUPTI
stays matched to the driver and `sm_100` support is preserved).

`nvcr.io/nvidia/pytorch:26.01-py3` (torch `2.10.0a0` / CUDA 13.1, digest
`sha256:38ed2ecb2c16d10677006d73fb0a150855d6ec81db8fc66e800b5ae92741007e`) is
also validated and performance-equivalent — rebuild `fbgemm_gpu` against
whichever image's torch you run.

### Dependency versions

| package | version | notes |
|---|---|---|
| **torch** | `2.12.0a0+0291f960b6.nv26.04.48445190` (CUDA 13.2) | native to the image; not reinstalled |
| **triton** | `3.6.0` | native to the image; provides `triton.language.make_tensor_descriptor` (required by the TRITON HSTU path) |
| **fbgemm_gpu** | FBGEMM commit `10b775730212923f65f7b78f79b6a01d80cf3c29` (2026-06-01 `main`, CUDA 13.2, `sm_100`) | built from source against the native torch; public wheels are ABI-incompatible with the NGC torch. The built wheel is named `fbgemm_gpu_nightly-2026.6.1` — that version is the build date, not the source date, so always identify the build by the commit above. Build command: `TORCH_CUDA_ARCH_LIST=10.0 python setup.py bdist_wheel --build-target default --build-variant cuda --package_channel nightly --nvml_lib_path /usr/lib/x86_64-linux-gnu/libnvidia-ml.so` (~55 min — the `sm_100` TBE-forward kernels dominate via `ptxas`) |
| **torchrec** | `1.7.0.dev20260601+cu130` (nightly, tested) | installed `--no-deps` from `https://download.pytorch.org/whl/nightly/cu130`. Perf-neutral vs stable `1.4.0`; use `1.4.0` (latest stable) if you prefer a non-pre-release |
| **polars-u64-idx** | `1.33.1` | 64-bit row index — `yambda-5b` has > 4.29 B rows (overflows stock polars' 32-bit index) |
| CUPTI (for `torch.profiler`) | 13.2 (native) | matches the driver; the `+cu128` stack's CUPTI 12.8 fails on B200 (`CUPTI_ERROR_INVALID_DEVICE`) |

Additional Python deps:
`xxhash`, `gin-config`, `absl-py`, `pandas`, `tensorboard`, `pyarrow`, `pyyaml`,
`tqdm`, `psutil`, `torchmetrics==1.0.3`, `tensordict`, `pyre-extensions`,
`iopath`, `typing-inspect`.

### Training configuration

From `generative_recommenders/dlrm_v3/train/gin/yambda_5b.gin`:

| parameter | value | gin binding |
|---|---|---|
| batch_size (train) | 32 | `make_train_test_dataloaders.batch_size` |
| eval_batch_size | 32 | `make_train_test_dataloaders.eval_batch_size` |
| num_workers (dataloader) | 4 | `make_train_test_dataloaders.num_workers` |
| prefetch_factor | 8 | `make_train_test_dataloaders.prefetch_factor` |
| num_blocks | 1 | `make_train_test_dataloaders.num_blocks` |
| train_split_percentage | 0.90 | `make_train_test_dataloaders.train_split_percentage` |
| history_length (per-sample UIH budget) | 2039 | `get_dataset.history_length` |
| max_seq_len (attention budget) | 2048 | `get_hstu_configs.max_seq_len` |
| bf16 training | True | `make_model.bf16_training` |
| HBM cap (per GPU) | 150 GiB | `make_optimizer_and_shard.hbm_cap_gb` (env `HBM_CAP_GB`) |
| **triton autotune pinning** | **True (full autotune)** | `apply_env_bootstrap.TRITON_FULL_AUTOTUNE` — the pinned configs are MI350X-specific, so B200 runs full autotune to find its own `sm_100` winners |
| dense optimizer | Adam, lr 1e-3, betas (0.95, 0.999), eps 1e-8 | `dense_optimizer_factory_and_class.*` |
| sparse optimizer | RowWiseAdagrad, lr 1e-3, betas (0.95, 0.999), eps 1e-8 | `sparse_optimizer_factory_and_class.*` |
| world_size | 8 | `MetricsLogger.world_size` |

Effective global batch = `batch_size × world_size = 32 × 8 = 256` samples/step.

### Environment variables

| var | value | purpose |
|---|---|---|
| `HSTU_HAMMER_KERNEL` | `TRITON` | fast HSTU kernel (vs `PYTORCH` fallback) |
| `TORCH_CUDA_ARCH_LIST` | `10.0` | target `sm_100` for JIT / Triton compilation |
| `DLRM_DATA_PATH` | dataset root | overrides gin default `/apps/chcai/dlrm_data` |
| `HBM_CAP_GB` | `150` | embedding planner HBM budget per GPU |
| `RUN_NAME` | run id | results dir → `results/<RUN_NAME>/` |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | allocator headroom |
| `TRITON_CACHE_DIR` | cache path | persist compiled Triton kernels across runs |
| `WORLD_SIZE` / `LOCAL_WORLD_SIZE` | `8` | mp.spawn rank count |

### Known pitfalls

- Never reinstall torch in this image — a cu12x wheel breaks CUPTI and may drop
  `sm_100`.
- The `+cu128` stack (`torch==2.7.1+cu128` + `fbgemm-gpu==1.2.0+cu128` +
  `torchrec==1.2.0+cu128`) runs on B200 but cannot profile GPU activity (CUPTI
  12.8 vs the 13.2 driver).
- Stock `polars` silently overflows on `yambda-5b` (> 4.29 B rows); always use
  `polars-u64-idx`.
- `EmbeddingBoundsCheck ... Setting idx to zero` warnings are benign data clamps.
