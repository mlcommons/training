# Recommendation v4 — HSTU sequential recommendation (Yambda-5b)

MLPerf Training reference benchmark. This is a fork of
[meta-recsys/generative-recommenders](https://github.com/meta-recsys/generative-recommenders)
extended to train an HSTU (Hierarchical Sequential Transduction Units) ranking
model on the [Yambda-5b](https://huggingface.co/datasets/yandex/yambda)
music-recommendation dataset, sized as an MLPerf-style training benchmark inside
the `mlcommons/training` tree.

## 1. Summary

This benchmark trains a model that predicts what a person will listen to next.
Given the history of songs a user has played, liked, or skipped, the model
learns to rank which song the user is most likely to genuinely listen to (rather
than skip) next. This is the same kind of "what should we recommend next?"
problem that powers music and video streaming feeds. The model is trained on a
large public dataset of anonymized music-listening events and is scored on how
well it predicts future listens it has never seen.

## 2. Benchmark overview (technical)

The model is a **sequential recommender**: instead of treating each interaction
independently (as classic click-through-rate models like DLRM-DCNv2 do), it
consumes a user's chronologically ordered interaction history as a sequence and
applies a Transformer-style attention stack (HSTU) over it. Each training
example is one "anchor" listen event together with that user's prior history
(user interaction history, or UIH) and a set of contextual/cross features. The
supervised target is a binary `listen_plus` label (a real listen: played for at
least 50% of the track) versus a skip.

Training is **streaming / temporal-order**: the timeline is sliced into
fixed-duration windows and the model trains on window `T` then evaluates on the
strictly-future window `T+1`, so every reported metric is genuine
next-period generalization with no future leakage. The quality metric is
**AUC** on the held-out future window, and the convergence target is
**AUC >= 0.80275** (matching the DLRM-DCNv2-style target).

The reference runs on 8 GPUs (validated on AMD Instinct MI350X / MI355X and
NVIDIA B200; see [docs/training_recipe.md](docs/training_recipe.md)) and scales
to multi-node via SLURM.

## 3. Directions — steps to run

The benchmark follows the standard MLPerf reference script flow:

```bash
# 0. build/enter the container (canonical frozen environment)
docker build -t recommendation_v4 .
docker run --rm -it --device=/dev/kfd --device=/dev/dri \
    -v /path/to/dlrm_data:/data/mlperf_dlrm_v4 recommendation_v4

# 1. download + preprocess the dataset
DLRM_DATA_PATH=/data/mlperf_dlrm_v4 ./download_dataset.sh

# 2. verify the preprocessed dataset
DLRM_DATA_PATH=/data/mlperf_dlrm_v4 ./verify_dataset.sh

# 3. run the benchmark to the quality target and report wall-clock time
DLRM_DATA_PATH=/data/mlperf_dlrm_v4 ./run_and_time.sh
```

- [`download_dataset.sh`](download_dataset.sh) wraps the preprocessing pipeline
  in `generative_recommenders.dlrm_v3.preprocess_public_data` (HuggingFace
  download + temporal split + session segmentation + item-popularity counts).
- [`verify_dataset.sh`](verify_dataset.sh) checks the preprocessed files against
  [`md5sums_yambda_5b_processed.txt`](md5sums_yambda_5b_processed.txt) (falls
  back to a layout check until the canonical checksums are pinned).
- [`run_and_time.sh`](run_and_time.sh) runs the full-reference streaming
  train+eval sweep on a single 8-GPU host with `AUC_THRESHOLD=0.80275` and MLPerf
  compliance logging, printing the elapsed time of the timed region.

### 3.1 Multi-node (SLURM)

For N >= 1 nodes use [`scripts/launch_slurm.sh`](scripts/launch_slurm.sh), which
provisions the container on each node and launches the same trainer. A bare
submit runs a small functional smoke run; set the run-shape knobs for the full
sweep:

```bash
# smoke (fast functional check)
sbatch --nodes=1 scripts/launch_slurm.sh

# full reference sweep
START_TS=0 NUM_TRAIN_TS=299 \
NUM_TRAIN_BATCHES=0 NUM_EVAL_BATCHES=0 \
EVAL_EVERY_N_WINDOWS=0 EVAL_EVERY_DATA_PCT=0.005 \
AUC_THRESHOLD=0.80275 \
sbatch --nodes=2 scripts/launch_slurm.sh
```

Multi-node uses real RDMA (RoCEv2); the fabric/NCCL setup is documented in
[docs/multi_node_config.md](docs/multi_node_config.md). Keep all run outputs
(log, checkpoints, mllog, TensorBoard) under a writable scratch path you own —
the dataset mount is read-only.

## 4. Model

The model is **HSTU** (Hierarchical Sequential Transduction Units), the
generative-recommender architecture from Meta's ICML'24 paper *Actions Speak
Louder than Words: Trillion-Parameter Sequential Transducers for Generative
Recommendations* ([arXiv:2402.17152](https://arxiv.org/abs/2402.17152)).

HSTU replaces the feature-interaction stack of a classic DLRM with a stack of
pointwise-attention "transducer" layers operating over the user's interaction
sequence. In this benchmark (the `dlrm_v3` path):

- **Embeddings**: sparse tables for `item_id`, `artist_id`, `album_id`, `uid`,
  and 7 cross-feature hashes (e.g. `user_x_artist`, `item_x_hour`), sharded
  across GPUs with TorchRec `DistributedModelParallel`.
- **Sequence model**: an HSTU attention stack (`HSTU_NUM_LAYERS`, default 3)
  over the interleaved UIH, computed with a fused jagged-attention Triton kernel
  in bf16.
- **Supervision**: a single `listen_plus` binary task. The candidate event's
  `action_weight` carries the supervision bit, and BCE loss is masked to
  listen_plus candidates.

See `generative_recommenders/dlrm_v3/configs.py`
(`get_hstu_configs`, `get_embedding_table_config`) for the exact architecture
and table specs, and the upstream README for the original modeling code.

## 5. Dataset

[Yambda-5b](https://huggingface.co/datasets/yandex/yambda) is a public
anonymized music-recommendation dataset from Yandex. The `5b` variant is used
for the reference. Statistics after preprocessing:

| | |
|---|---|
| Total interaction events | **4.76 B** |
| Unique users | **1.00 M** |
| Max events per user | 27,738 |
| Median events per user | 2,695 |
| Mean events per user | 4,763 |
| Train events (300d) | 4.76 B |
| Test events (1d) | 22.4 M |
| Item catalog size | 9.39 M |

### 5.1 Per-event-type distribution (across the full 4.76 B corpus)

| Pool | Definition | Count | Share |
|---|---|---|---|
| **listen_plus (lp)** | `is_listen AND played_ratio >= 50%` | 2.92 B | **61.3%** |
| **skip** | `is_listen AND played_ratio < 50%` | 1.71 B | **35.9%** |
| **like** | explicit thumbs-up action | 89 M | **1.9%** |
| other | dislike / unlike / undislike | 47 M | 1.0% |

The `like` pool is roughly **30x rarer** than `lp` — important context for the
gather strategy in §6.

### 5.2 Preprocessing & download

`./download_dataset.sh` (which calls
`python3 -m generative_recommenders.dlrm_v3.preprocess_public_data --dataset
yambda-5b --data-path <root>`) downloads the 5b variant from HuggingFace, then:

1. **Encodes** the raw `event_type` string into a uint8 lookup (listen=0,
   like=1, dislike=2, unlike=3, undislike=4).
2. **Splits** events temporally — 300 train days, 30-min gap, 1 test day — by
   Global Temporal Split (GTS).
3. **Segments** per-user event timelines into sessions on a 30-min inactivity
   gap.
4. **Computes** per-item popularity for downstream metric weighting.
5. **Writes** the layout `DLRMv3YambdaDataset` expects:

```
<data-path>/
├── raw/5b/multi_event.parquet                    50 GB (downloaded)
├── shared_metadata/
│   ├── artist_item_mapping.parquet               60 MB
│   ├── album_item_mapping.parquet                76 MB
│   └── embeddings.parquet                         18 GB (unused by HSTU training)
└── processed_5b/
    ├── train_sessions.parquet                     47 GB  ← main training input
    ├── test_events.parquet                       152 MB
    ├── session_index.parquet                     600 MB
    ├── item_popularity.npy                        75 MB
    └── split_meta.json                            anchor + boundary stats
```

For smaller variants (`yambda-50m` / `yambda-500m`) substitute the dataset name
(`DATASET=yambda-50m ./download_dataset.sh`). Preprocessing takes ~2 min for 50m
and ~53 min for 5b end-to-end.

Integrity is verified with `./verify_dataset.sh` against
[`md5sums_yambda_5b_processed.txt`](md5sums_yambda_5b_processed.txt).

## 6. How data is fed to HSTU

For every training anchor (a LISTEN event with >= `min_history` prior events),
the dataset builds a `(uih_kjt, candidate_kjt)` pair:

```
UIH (User Interaction History):
  ┌─ Sequence features (chronologically interleaved across 3 pools)
  │    item_id, artist_id, album_id            ← per-position
  │    action_weight                            ← per-position (LP_BIT/LIKE_BIT/SKIP_BIT)
  │    action_timestamp, dummy_watch_time      ← per-position
  └─ Contextual features (length 1 each)
       uid + 7 cross-feature hashes (user_x_artist, item_x_hour, …)
                                                = 8 contextual entries

CANDIDATE (the LISTEN event at the anchor):
  item_id, artist_id, album_id, item_query_time,
  item_action_weight (LP_BIT if listen_plus, else 0),
  item_dummy_watchtime
```

The candidate's `action_weight` is **the supervision label**: HSTU's
`_get_supervision_labels_and_weights` masks BCE training to
`(supervision_bitmask & task_weight) > 0`, with `task_weight = 1` (LP bit) for
the single `listen_plus` task — so only listen_plus candidates supervise.

### 6.1 Per-pool gather (the cap = L // 3 strategy)

The UIH is built by `DLRMv3YambdaDataset._gather_interleaved_history`. For each
anchor it:

1. Scans the most recent `scan_window` (default 20,000) events of any type
   before the anchor, **clipped to user_start**.
2. From those, takes **the last `L // 3` events** from each of the three pools
   (lp, like, skip) independently.
3. Concatenates and **re-sorts chronologically** to produce an interleaved
   sequence.
4. Tags each event's pool identity into `action_weight` via OR'd bitmask
   (LP=1, LIKE=2, SKIP=4).

With `history_length = 4086` and `max_seq_len = 4096`: per-pool cap = `4086 //
3 = 1362`, and `3 × 1362 + 8 contextual + 1 candidate = 4095 <= 4096` (no
truncation). Because the `like` pool is rare (1.9%) it under-fills (~105 events
per anchor on average); the Triton jagged-attention backend skips unfilled
slots, so the under-fill costs sequence budget but not GPU compute.

## 7. Optimizer

Two optimizers, configured in
[`yambda_5b.gin`](generative_recommenders/dlrm_v3/train/gin/yambda_5b.gin):

| component | optimizer | gin binding | key settings |
|---|---|---|---|
| Dense params (HSTU blocks, MLPs) | **Adam** | `dense_optimizer_factory_and_class.*` | lr `DENSE_LR`, betas (0.95, 0.999), eps 1e-8, weight_decay 0 |
| Sparse embedding tables | **RowWiseAdagrad** (fused FBGEMM TBE) | `sparse_optimizer_factory_and_class.*` | lr `SPARSE_LR`, eps 1e-8, weight_decay 0 |

Gradient clipping (`GRAD_CLIP_NORM`, default `max_norm=1.0`) is applied to the
dense parameters on the streaming path; the fused sparse optimizer is
unaffected. Training is bf16 mixed precision (`make_model.bf16_training=True`).

## 8. Hyperparameters

All tunable hyperparameters live in
[`yambda_5b.gin`](generative_recommenders/dlrm_v3/train/gin/yambda_5b.gin)
(the config-file source of truth) and are **overridable via environment
variables** (the env value takes precedence over the gin default, per MLPerf
CONTRIBUTING rule 4d). The gin macros (`@env_int`, `@env_float`, `@env_str`)
enforce the correct type for each parameter.

| hyperparameter | env var | gin binding | type | default | tuning rule |
|---|---|---|---|---|---|
| Per-rank batch size | `BATCH_SIZE` | `batch_size` | int | 1024 | positive integer (global batch = `BATCH_SIZE × world_size`) |
| Dense learning rate | `DENSE_LR` | `dense_optimizer_factory_and_class.learning_rate` | float | 1e-7 | positive float |
| Sparse learning rate | `SPARSE_LR` | `sparse_optimizer_factory_and_class.learning_rate` | float | 1e-7 | positive float |
| Grad clip max-norm | `GRAD_CLIP_NORM` | `streaming_train_eval_loop.grad_clip_norm` | float | 1.0 | float >= 0 (0 disables) |
| RNG seed | `SEED` | `seed_everything.seed` | int | 1 | any integer (-1 = random per run) |
| HSTU attention layers | `HSTU_NUM_LAYERS` | `get_hstu_configs.hstu_attn_num_layers` | int | 3 | positive integer |
| UIH history length | `HISTORY_LENGTH` | `get_dataset.history_length` | int | 4086 | positive integer (per-pool cap = L//3) |
| Max sequence length | `MAX_SEQ_LEN` | `get_hstu_configs.max_seq_len` | int | 4096 | positive integer (>= `history_length + 9`) |
| History strategy | `HISTORY_STRATEGY` | `get_dataset.history_strategy` | str | `interleaved` | one of `interleaved` \| `last_n` |
| Min history (anchor floor) | `MIN_HISTORY` | `get_dataset.min_history` | int | 4086 | integer >= 0 |
| Train user split | `TRAIN_SPLIT_PERCENTAGE` | `*.train_split_percentage` | float | 1.0 | float in (0, 1] |
| Streaming shuffle fraction | `STREAMING_SHUFFLE_FRACTION` | `get_dataset.streaming_shuffle_fraction` | float | 0.0 | float in [0, 1] |
| Streaming shuffle seed | `STREAMING_SHUFFLE_SEED` | `get_dataset.streaming_shuffle_seed` | int | 0 | any integer |
| Split salt | `SPLIT_SALT` | `get_dataset.split_salt` | int | 0 | any integer |
| Start window | `START_TS` | `streaming_train_eval_loop.start_ts` | int | 150 | integer >= 0 |
| Number of train windows | `NUM_TRAIN_TS` | `streaming_train_eval_loop.num_train_ts` | int | 149 | positive integer (clamped to available) |
| Sparse A2A fwd precision | `SPARSE_A2A_FWD` | `make_optimizer_and_shard.sparse_a2a_forward_precision` | str | `fp32` | one of `fp32` \| `bf16` \| `fp16` |
| Sparse A2A bwd precision | `SPARSE_A2A_BWD` | `make_optimizer_and_shard.sparse_a2a_backward_precision` | str | `fp32` | one of `fp32` \| `bf16` \| `fp16` |

Non-tunable / fixed reference values (optimizer betas (0.95, 0.999), eps 1e-8,
weight_decay 0, bf16 training, streaming window = 86400 s) are pinned in the gin
file. Submitters tuning hyperparameters must follow the allowed values above and
the
[MLPerf training rules](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc#hyperparameters).

## 9. Quality target & evaluation

- **Metric**: AUC on the held-out future evaluation window (`window_auc` for the
  `listen_plus` task), computed by `MetricsLogger` in
  `generative_recommenders/dlrm_v3/utils.py`.
- **Target**: **eval AUC >= 0.80275**. Set via `AUC_THRESHOLD=0.80275`
  (`MetricsLogger.auc_threshold`); the run logs `RUN_STOP` with `SUCCESS` and
  stops once the target is reached. The gin default of `1.0` is unreachable
  (trains all windows with no early stop) and is overridden by the reference
  scripts.
- **Evaluation frequency**: the full-reference run uses
  `EVAL_EVERY_DATA_PCT=0.005` — evaluate every 0.5% of the training stream
  (~200 evenly-data-spaced eval points), independent of node count. The
  alternative per-window cadence (`EVAL_EVERY_N_WINDOWS`) is mutually exclusive.
- **Evaluation set**: a fixed held-out future window (`eval_holdout_ts`, default
  `start_ts + num_train_ts`); with `TRAIN_SPLIT_PERCENTAGE < 1.0` the held-out
  users' anchors over that window form the eval set. The temporal one-window
  lead guarantees no future leakage (see §11).

Evaluation is always one window ahead of training, so reported AUC is genuine
next-period generalization.

## 10. Reference Convergence Points (RCP)

*Placeholder — to be generated.*

RCPs have **not yet been generated** for this benchmark. Per the MLPerf
[CONTRIBUTING guidance](https://github.com/mlcommons/training_policies/blob/master/CONTRIBUTING.md),
RCPs must be generated for at least 3 reasonable batch sizes using at least 2N
seeds (N = number of submission runs), in FP32 or BF16, with the exact precision
recorded in the RCP JSON. The convergence curves (steps/samples to reach
AUC >= 0.80275) will be added under [`rcp/`](rcp/) once the convergence runs are
complete. This section is intentionally left blank for now.

## 11. Streaming (temporal-order) training

`scripts/launch_slurm.sh` and `run_and_time.sh` default to
`--mode streaming-train-eval`, which trains Yambda in strict wall-clock order
instead of shuffling the whole corpus. The timeline is sliced into
fixed-duration **windows** (default 1 day,
`get_dataset.streaming_window_seconds = 86400`), and the loop walks them forward:

```
window T:    train          window T+1:  eval (then train)   window T+2: eval (then train) ...
   └─ train window T ─┐
                      └─ eval window T+1 ─┐
                                          └─ train window T+1 ─┐
                                                               └─ eval window T+2 ...
```

i.e. for each step it **trains window T, then evaluates window T+1** before
advancing — always predicting the immediate future from the past.

### 11.1 Temporal guarantee

The streaming path enforces **no future leakage** at two levels:

1. **Across windows** — a window is the set of anchors whose target/candidate
   timestamp falls in `[t_min + T·W, t_min + (T+1)·W)`. Training only ever sees
   windows `<= T`; the evaluation window `T+1` is strictly in the future of every
   training anchor it is scored against.
2. **Within an anchor** — history is gathered **causally**: the UIH scan is
   `scan_start:flat_pos` (events strictly before the anchor), so no event at or
   after the anchor's timestamp can enter its features.

This is a *temporal* split on the training stream — distinct from the
preprocessing GTS split (§5) that carves off the final test day. Windows are
indexed off the per-anchor target timestamp via a lazily-built, mmap'd
`anchor_ts` cache keyed by `(history_length, min_history)`.

### 11.2 Streaming knobs

All configurable via
[`yambda_5b.gin`](generative_recommenders/dlrm_v3/train/gin/yambda_5b.gin) with
env overrides:

| env | gin | default | meaning |
|---|---|---|---|
| `START_TS` | `streaming_train_eval_loop.start_ts` | 150 | first window (early windows are near-empty warm-up) |
| `NUM_TRAIN_TS` | `streaming_train_eval_loop.num_train_ts` | 149 | number of train windows (clamped to available) |
| `PERSISTENT_LOADER` | `streaming_train_eval_loop.persistent_loader` | 1 | reuse one worker pool across windows |
| `DOUBLE_BUFFER` | `streaming_train_eval_loop.double_buffer` | 1 | prepare the next window in a background thread |
| `EVAL_EVERY_N_WINDOWS` | `streaming_train_eval_loop.eval_every_n_windows` | 1 | eval cadence by window count (0 to use data-pct) |
| `EVAL_EVERY_DATA_PCT` | `streaming_train_eval_loop.eval_every_data_pct` | 0.0 | eval cadence by fraction of train data (full ref: 0.005) |
| `MIN_HISTORY` | `get_dataset.min_history` | 4086 | anchor-eligibility floor (0 = ~all users incl. cold-start) |

### 11.3 Checkpointing & resume

The streaming loop is resume-aware: set `CKPT_PATH` to enable DMP checkpoint
save/load (auto-resolves to the highest-numbered subdir), with retention via
`KEEP_LAST_N` and cadences `IN_WINDOW_CKPT_FREQ` / `CKPT_STEP_FREQ` /
`CKPT_TIME_INTERVAL_S`. The MLPerf run state (run-started flag, global sample
count) is persisted across resume so compliance logging is continuous. See
`generative_recommenders/dlrm_v3/checkpoint.py`.

## 12. License

Apache 2.0 (inherited from upstream).
