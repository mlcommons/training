# Training Recipe

Reproducible environment + configuration for training HSTU / DLRM-v3 on the
`yambda-5b` dataset.

---

## B200

Single-node, 8× NVIDIA **B200** (Blackwell, `sm_100`, ~183 GiB HBM each), HSTU
ranker on `yambda-5b` with the **TRITON** HSTU kernel and **bf16** mixed-precision
training.

### Hardware / host

| item | value |
|---|---|
| GPUs | 8× NVIDIA B200 (`sm_100`, compute capability 10.0) |
| Host driver | 580.159.03 (reports CUDA 13.2) |
| Forward-compat userspace driver | `libcuda.so.595.45.04` (engaged automatically by the NGC image) |

### Container image

```
nvcr.io/nvidia/pytorch:26.01-py3
```

Digest: `sha256:38ed2ecb2c16d10677006d73fb0a150855d6ec81db8fc66e800b5ae92741007e`

The image's native PyTorch is kept as-is and must not be reinstalled (so CUPTI
stays matched to the driver and `sm_100` support is preserved).

### Dependency versions

| package | version | notes |
|---|---|---|
| **torch** | `2.10.0a0+a36e1d39eb.nv26.01.42222806` (CUDA 13.1) | native to the image; not reinstalled |
| **triton** | `3.6.0` | native to the image; provides `triton.language.make_tensor_descriptor` (required by the TRITON HSTU path) |
| **fbgemm_gpu** | `fbgemm_gpu_nightly-2026.6.1` (CUDA 13.1, `sm_100`) | built from source against the native torch, from FBGEMM commit `939f2da156b05d2f1bcba8c037d613c1098d0db5` (2026-04-29); public wheels are ABI-incompatible with the NGC torch |
| **torchrec** | `1.4.0` | installed with `--no-deps` |
| **polars-u64-idx** | `1.33.1` | 64-bit row index — `yambda-5b` has > 4.29 B rows (overflows stock polars' 32-bit index) |
| CUPTI (for `torch.profiler`) | 13.1 (native) | matches the driver; the `+cu128` stack's CUPTI 12.8 fails on B200 (`CUPTI_ERROR_INVALID_DEVICE`) |

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
