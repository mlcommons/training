# Recommendation v4 — HSTU + Yambda-5b

This is a fork of [meta-recsys/generative-recommenders](https://github.com/meta-recsys/generative-recommenders) extended to train HSTU (Hierarchical Sequential Transducer Units) on the [Yambda-5b](https://huggingface.co/datasets/yandex/yambda) music-recommendation dataset, sized as an MLPerf-style training benchmark inside the `mlcommons/training` tree.

For the original repository and the underlying ICML'24 paper (*Actions Speak Louder than Words*), see the upstream README at the link above. This README focuses on what this fork adds: the Yambda data pipeline, the per-pool gather strategy, and how the data feeds into the HSTU `modules/` (dlrm_v3) path.

## 1. Quick start (Yambda, N×8-GPU)

`scripts/launch_slurm.sh` is the single entry point for **N ≥ 1 nodes**. It
auto-detects its context: run inside the container it takes the single-node
worker path; submitted via `sbatch` it orchestrates the multi-node run
(provision + per-node launch). N=1 is byte-for-byte the legacy single-node path.

A bare submit reproduces the **frozen reference run** (full 299-window sweep +
data-fraction eval cadence) — all run-shape/cadence defaults are baked in, so no
env knobs are required:

**Single node (8-GPU):**

```bash
sbatch --nodes=1 scripts/launch_slurm.sh
```

**Multi-node (N×8-GPU):**

```bash
sbatch --nodes=2 scripts/launch_slurm.sh
```

For a fast functional check instead of a full run, prepend `SMOKE=1` (short
window, capped batches, per-window eval):

```bash
SMOKE=1 sbatch --nodes=1 scripts/launch_slurm.sh
```

Multi-node uses real RDMA (RoCEv2). The fabric/NCCL setup and every
cluster-specific knob (interfaces, HCAs, GID/TC, RDMA overlay) are documented in
[docs/multi_node_config.md](docs/multi_node_config.md) — read it before running on
a different cluster.

Override the data path or run name without editing the gin:

```bash
DLRM_DATA_PATH=/apps/chcai/dlrm_data \
RUN_NAME=my_experiment \
bash scripts/launch_slurm.sh
```

Data path resolves at runtime via `env_path` gin macros (see [yambda_5b.gin](generative_recommenders/dlrm_v3/train/gin/yambda_5b.gin)). Traces and any per-run outputs land in `results/<RUN_NAME>/`.

### 1.1 The frozen reference shape

The reference run-shape and eval cadence are the built-in defaults (set in the
orchestrate phase of `scripts/launch_slurm.sh`), so the bare `sbatch` commands
above ARE the reference run. The single- and multi-node launchers are identical
except for `--nodes`; the trainer auto-derives `NNODES`/`NODE_RANK`/`MASTER_ADDR`/
`WORLD_SIZE` from SLURM. The baked-in shape:

| knob | reference default |
|---|---|
| `START_TS` / `NUM_TRAIN_TS` | 0 / 299 (full sweep) |
| eval cadence | `EVAL_EVERY_DATA_PCT=0.005` (every 0.5% of the training stream — a fixed number of samples between evals, independent of node count), per-window cadence off |
| `NUM_TRAIN_BATCHES` / `NUM_EVAL_BATCHES` | 0 / 0 (consume full windows) |

To customize, override any knob via env (e.g. `RUN_NAME=...`, `LOG=...`,
`AUC_THRESHOLD=...`). Selecting the per-window eval cadence
(`EVAL_EVERY_N_WINDOWS>0`) automatically disables the data-fraction one (they are
mutually exclusive). Keep all run outputs (`LOG`, checkpoints, mllog,
TensorBoard) under a writable scratch path you own — the dataset mount is
read-only.

## 2. Data preparation

```bash
python3 -m generative_recommenders.dlrm_v3.preprocess_public_data \
  --dataset yambda-5b --data-path /apps/chcai/dlrm_data
```

This downloads the 5b variant of [yandex/yambda](https://huggingface.co/datasets/yandex/yambda) from HuggingFace, then:

1. **Encodes** the raw `event_type` string column into a uint8 lookup (listen=0, like=1, dislike=2, unlike=3, undislike=4).
2. **Splits** events temporally — 300 train days, 30-min gap, 1 test day — by `Global Temporal Split` (GTS).
3. **Segments** per-user event timelines into sessions on a 30-min inactivity gap.
4. **Computes** per-item popularity for downstream metric weighting.
5. **Writes** the layout `DLRMv3YambdaDataset` expects:

```
<data-path>/
├── raw/5b/multi_event.parquet                    50 GB (downloaded)
├── shared_metadata/
│   ├── artist_item_mapping.parquet               60 MB
│   ├── album_item_mapping.parquet                76 MB
│   └── embeddings.parquet                        18 GB (unused by HSTU training)
└── processed_5b/
    ├── train_sessions.parquet                    47 GB  ← main training input
    ├── test_events.parquet                       152 MB
    ├── session_index.parquet                     600 MB
    ├── item_popularity.npy                       75 MB
    └── split_meta.json                           anchor + boundary stats
```

For smaller variants (yambda-50m / yambda-500m) substitute the dataset name. The preprocessor takes ~2 min for 50m and ~53 min for 5b end-to-end.

## 3. Yambda dataset statistics

Numbers from the 5b variant, after preprocessing:

| | |
|---|---|
| Total interaction events | **4.76 B** |
| Unique users | **1.00 M** |
| Max events per user | 27,738 |
| Median events per user | 2,695 |
| Mean events per user | 4,763 |
| Train events (300d) | 4.76 B |
| Test events (1d) | 22.4 M |
| Training positions (≥2039 prior events filter) | **3.23 B** |
| Item catalog size | 9.39 M |

### 3.1 Per-event-type distribution (across the full 4.76 B corpus)

| Pool | Definition | Count | Share |
|---|---|---|---|
| **listen_plus (lp)** | `is_listen AND played_ratio ≥ 50%` | 2.92 B | **61.3%** |
| **skip** | `is_listen AND played_ratio < 50%` | 1.71 B | **35.9%** |
| **like** | explicit thumbs-up action | 89 M | **1.9%** |
| other | dislike / unlike / undislike | 47 M | 1.0% |

The `like` pool is roughly **30× rarer** than `lp` — important context for the gather strategy below.

## 4. How data is fed to HSTU

For every training anchor (a LISTEN event with ≥ `min_history` prior events — frozen default `4086`, the "full `history_length` of context required" power-users filter; set `$MIN_HISTORY=0` to include ~all users plus their cold-start first event), the dataset builds a `(uih_kjt, candidate_kjt)` pair:

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

The candidate's `action_weight` is **the supervision label**: HSTU's `_get_supervision_labels_and_weights` masks BCE training to `(supervision_bitmask & task_weight) > 0`, with `task_weight = 1` (LP bit) for the single `listen_plus` task — so only listen_plus candidates supervise.

### 4.1 Per-pool gather (the cap = L // 3 strategy)

The UIH is built by `DLRMv3YambdaDataset._gather_interleaved_history`. For each anchor, it:

1. Scans the most recent `scan_window` (default 20,000) events of any type before the anchor, **clipped to user_start** so users with shorter history get a smaller window.
2. From those, takes **the last `L // 3` events** from each of the three pools (lp, like, skip) independently.
3. Concatenates the three streams and **re-sorts chronologically** to produce an interleaved sequence.
4. Tags each event's pool identity into `action_weight` via OR'd bitmask (LP=1, LIKE=2, SKIP=4).

With `L = 2039` and `max_seq_len = 2048`:
- Per-pool cap = `L // 3 = 679`
- Maximum total UIH = `3 × 679 = 2037` events
- Plus `8 contextual + 1 candidate = 9` overhead → 2046 ≤ 2048 model budget (no truncation)

### 4.2 Effective per-anchor fill on real data

Because the `like` pool is rare (1.9% of events) and the average user has only ~4,763 lifetime events:

| Pool | per-pool cap (L//3) | actual avg fill per anchor | fill rate |
|---|---|---|---|
| lp | 679 | ~673 | **99%** |
| like | 679 | ~105 | **15%** (data-bounded, not cap-bounded) |
| skip | 679 | ~624 | 92% |
| **total UIH** | 2037 max | **~1402** | 69% |

The `like` cap of 679 is unreachable for yambda data — at the 1.9% global like rate, filling 679 likes would require a user to have ~36k prior events, but the **longest user in the dataset has only 27,738 events total** (and the median user has 2,695). So under-fill on `like` is fundamental to the data.

This means the model sees on average ~1,402 UIH events per sample, not the theoretical 2,037. With the TRITON jagged-attention backend the GPU only does work for the actual events, so the under-fill costs **sequence budget but not GPU compute** — no wasted attention work, just less context per sample than the budget suggests.

## 5. Streaming (temporal-order) training

`scripts/launch_slurm.sh` defaults to `--mode streaming-train-eval`, which
trains Yambda in strict wall-clock order instead of shuffling the whole corpus.
The timeline is sliced into fixed-duration **windows** (default 1 day,
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

### 5.1 Temporal guarantee

The streaming path enforces **no future leakage** at two levels:

1. **Across windows** — a window is the set of anchors whose *target/candidate*
   timestamp falls in `[t_min + T·W, t_min + (T+1)·W)`. Training only ever sees
   windows `≤ T`; the evaluation window `T+1` is strictly in the future of every
   training anchor it is scored against. Eval always leads train by exactly one
   window, so reported eval NE/AUC is genuine next-period generalization, never
   an in-sample measurement.
2. **Within an anchor** — history is still gathered **causally**: the UIH scan
   is `scan_start:flat_pos` (events strictly before the anchor), so even though a
   long user history may reach back across earlier windows, no event at or after
   the anchor's timestamp can enter its features. Forward-time windowing and
   causal history are independent guarantees, and both hold simultaneously.

Note this is a *temporal* split on the training stream — distinct from the
preprocessing GTS split (§2) that carves off the final test day. Windows are
indexed off the per-anchor target timestamp via a lazily-built, mmap'd
`anchor_ts_L{H}[_m{MIN_HISTORY}].npy` cache (built once on first use; the
default non-streaming path never touches it). The anchor `positions` and
`anchor_ts` arrays are keyed by `(history_length, min_history)` so different
floors don't collide and the expensive flat store is shared across them.

### 5.2 Knobs

All configurable via gin ([yambda_5b.gin](generative_recommenders/dlrm_v3/train/gin/yambda_5b.gin))
with env overrides:

| env | gin | default | meaning |
|---|---|---|---|
| `START_TS` | `streaming_train_eval_loop.start_ts` | 150 | first window (early windows are near-empty warm-up; start dense) |
| `NUM_TRAIN_TS` | `streaming_train_eval_loop.num_train_ts` | 30 | number of train windows (clamped to available) |
| `PERSISTENT_LOADER` | `streaming_train_eval_loop.persistent_loader` | 1 | reuse one worker pool across windows (no per-window respawn) |
| `DOUBLE_BUFFER` | `streaming_train_eval_loop.double_buffer` | 1 | prepare the next window in a background thread during compute |
| `EVAL_EACH_WINDOW` | `streaming_train_eval_loop.eval_each_window` | 1 | eval window T+1 after training window T |
| `MIN_HISTORY` | `get_dataset.min_history` | 4086 | anchor-eligibility floor: min prior events for a LISTEN to be a sample (frozen default 4086 = full-history power-users filter; 0 = ~all users incl. cold-start) |
| — | `streaming_train_eval_loop.num_train_batches` / `num_eval_batches` | unset | cap per-window steps (unset = consume full window) |

### 5.3 Hiding the window-reset overhead

Advancing to a new window has a fixed cost — selecting the window's anchor
indices and warming the dataloader's first batch — that, done naively, stalls
training at every window boundary. Three layers drive it to ~0:

1. **Persistent loader** (`persistent_loader=1`). The naive path recreates a
   `DataLoader` per window, re-forking workers and paying first-batch warmup
   each time (~11 s/window). Instead we build **one** `DataLoader` backed by a
   stateful `StreamingWindowSampler` whose index set is swapped per window
   (`set_window`), so workers fork once and persist. This removes the respawn
   but still pays the index-mask + first-batch stall (~3.6 s/window).
2. **Double buffering** (`double_buffer=1`). Two pre-forked worker pools
   ping-pong: while the current window trains on pool A, the *next* window's
   index mask (`window_indices`, a GIL-releasing NumPy `np.where`) and
   first-batch prefetch are prepared on pool B in a **background thread**, so
   that work overlaps GPU compute. The boundary train batch then arrives warm —
   measured train first-batch data-wait drops to **~1–3 ms**. Pools are forked
   up front on the main thread (never inside the background thread), so a forking
   worker can never race a thread holding a lock.
3. **Eval prefetch one window ahead.** With `eval_each_window=1` the eval window
   (`T+1`) is prepared *before* training window `T` runs, so the idle eval pool
   prefetches its first batches concurrently with train compute. This hides the
   eval-side first-batch stall (**~0.55 s → ~2 ms**). It is safe because a
   sample's content depends only on the sampler's window indices, not on any
   train/eval flag.

Net effect: steady-state throughput matches the non-streaming baseline and the
per-window reset is effectively free; the only remaining one-time cost is the
process cold start (CUDA-graph capture + the first lazy `anchor_ts` mmap).

## 6. License

Apache 2.0 (inherited from upstream).
