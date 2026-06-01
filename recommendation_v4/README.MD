# Recommendation v4 — HSTU + Yambda-5b

This is a fork of [meta-recsys/generative-recommenders](https://github.com/meta-recsys/generative-recommenders) extended to train HSTU (Hierarchical Sequential Transducer Units) on the [Yambda-5b](https://huggingface.co/datasets/yandex/yambda) music-recommendation dataset, sized as an MLPerf-style training benchmark inside the `mlcommons/training` tree.

For the original repository and the underlying ICML'24 paper (*Actions Speak Louder than Words*), see the upstream README at the link above. This README focuses on what this fork adds: the Yambda data pipeline, the per-pool gather strategy, and how the data feeds into the HSTU `modules/` (dlrm_v3) path.

## 1. Quick start (8-GPU Yambda)

```bash
docker exec yambda_8gpu bash -c \
  'cd /workspace/recommendation_v4 && bash scripts/launch_smoke_8gpu.sh'
```

Override the data path or run name without editing the gin:

```bash
DLRM_DATA_PATH=/apps/chcai/dlrm_data \
RUN_NAME=my_experiment \
bash scripts/launch_smoke_8gpu.sh
```

Data path resolves at runtime via `env_path` gin macros (see [yambda_5b.gin](generative_recommenders/dlrm_v3/train/gin/yambda_5b.gin)). Traces and any per-run outputs land in `results/<RUN_NAME>/`.

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

For every training anchor (a LISTEN event with ≥ `history_length` prior events), the dataset builds a `(uih_kjt, candidate_kjt)` pair:

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

## 5. License

Apache 2.0 (inherited from upstream).
