# recommendation_v4 (HSTU + Yambda-5b) — reference

A walkthrough of what the proposed `recommendation_v4` MLPerf-training benchmark
is, how it differs from `recommendation_v2`, what the HSTU model is composed of,
and how to download the dataset and run training as-is.

All claims below are grounded in code/config paths inside this tree. Every
numeric constant cites a `file:line` source. Where doc and source disagree, the
source wins and the discrepancy is called out.

---

## 0. Sources of truth

The following files were read to assemble this document. If you change any of
them, audit this doc against the change.

- `training/recommendation_v2/torchrec_dlrm/README.MD` (v2 reference)
- `training/recommendation_v4/README.MD` (v4 fork overview)
- `training/recommendation_v4/docs/training_recipe.md` (v4 stacks/configs)
- `training/recommendation_v4/generative_recommenders/modules/stu.py` (HSTU layer)
- `training/recommendation_v4/generative_recommenders/modules/dlrm_hstu.py` (top-level `DlrmHSTU` + config dataclass)
- `training/recommendation_v4/generative_recommenders/modules/hstu_transducer.py` (preprocessor → STU stack → postprocessor)
- `training/recommendation_v4/generative_recommenders/ops/pytorch/pt_hstu_attention.py` (reference HSTU attention math in plain PyTorch)
- `training/recommendation_v4/generative_recommenders/ops/hstu_attention.py` (kernel dispatcher: PYTORCH / TRITON / TRITON_CC)
- `training/recommendation_v4/generative_recommenders/dlrm_v3/configs.py` (per-dataset HSTU config + embedding tables)
- `training/recommendation_v4/generative_recommenders/dlrm_v3/train/gin/yambda_5b.gin` (run config)
- `training/recommendation_v4/generative_recommenders/dlrm_v3/preprocess_public_data.py` (Yambda HuggingFace downloader/preprocessor)
- `training/recommendation_v4/generative_recommenders/dlrm_v3/datasets/yambda.py` (dataset feeding HSTU)
- `training/recommendation_v4/generative_recommenders/dlrm_v3/utils.py` (metrics logger / `auc_threshold` consumer)
- `training/recommendation_v4/scripts/launch_smoke_8gpu.sh` (run wrapper)

---

## 1. `recommendation_v4` vs `recommendation_v2`

v4 is **not** an evolution of v2 — it replaces a tabular CTR benchmark
(DLRMv2 + DCN on Criteo 1 TB) with a **sequential generative-recommender
benchmark** (HSTU on Yandex Yambda-5b). Codebase, dataset, task, loss
labeling, hyperparameters, and software stack are all different. They share
basically nothing except the "recommendation" label.

### 1.1 Upstream codebase / repo origin

| | v2 | v4 |
|---|---|---|
| Upstream repo | `pytorch/torchrec` examples (DLRM) | fork of `meta-recsys/generative-recommenders` (`README.MD:3`) |
| Layout | single dir: `torchrec_dlrm/` with `dlrm_main.py` | full repo tree: `generative_recommenders/`, `configs/`, `scripts/`, `main.py`, `setup.py`, gin-driven |
| Config style | argparse CLI flags | gin-config files under `generative_recommenders/dlrm_v3/train/gin/` (e.g. `yambda_5b.gin`) |

### 1.2 Model architecture

| | v2 | v4 |
|---|---|---|
| Model | **DLRM v2** — dense MLP + sparse embeddings + feature interaction (paper: Naumov et al. 1906.00091) | **HSTU** — Hierarchical Sequential Transducer Units (ICML'24 *Actions Speak Louder than Words*) (`README.MD:3`) |
| Interaction arch | DCN v2: `--interaction_type=dcn --dcn_num_layers=3 --dcn_low_rank_dim=512` (`recommendation_v2/torchrec_dlrm/README.MD:167-169`) | Transformer-style sequential self-attention over a User Interaction History (UIH) of length 2048, jagged-attention TRITON kernel (`README.MD:114, 132`; `training_recipe.md:57, 71`) |
| Embedding dim | 128 (`recommendation_v2/torchrec_dlrm/README.MD:157`) | 512 (`dlrm_v3/configs.py:33, 353`) |
| Pipeline | TorchRec model-parallel embeddings + data-parallel MLP, overlapped (`recommendation_v2/torchrec_dlrm/README.MD:3`) | TorchRec sharded embeddings + HSTU ranker; per-GPU HBM cap 260 GiB MI350X / 150 GiB B200 (`training_recipe.md:59, 176`) |

### 1.3 Dataset

| | v2 | v4 |
|---|---|---|
| Dataset | **Criteo 1 TB click logs** → multi-hot preprocessed variant (~3.8 TB materialized) (`recommendation_v2/torchrec_dlrm/README.MD:142-146`) | **Yambda-5b** (Yandex music, HuggingFace `yandex/yambda`, 5b variant) (`README.MD:3, 28`) |
| Domain | CTR prediction on tabular ads features (26 categorical + 13 dense) | Sequential music-recommendation events (listen / like / skip / dislike / unlike / undislike) per-user timelines |
| Size | `TOTAL_TRAINING_SAMPLES=4,195,197,692` rows (`recommendation_v2/torchrec_dlrm/README.MD:153`) | 4.76 B events, 1.00 M users, 9.39 M items; 3.23 B usable training anchors (`README.MD:62-69`) |
| Storage layout | numpy contiguous shuffled `.npy` (or preprocessed multi-hot bin) | parquet: `train_sessions.parquet` 47 GB, `test_events.parquet` 152 MB, etc. (`README.MD:40-52`) |
| Preprocessing | `process_Criteo_1TB_Click_Logs_dataset.sh` — 700 GB RAM, 1–2 days, then `materialize_synthetic_multihot_dataset.py` | `generative_recommenders.dlrm_v3.preprocess_public_data --dataset yambda-5b` — ~53 min end-to-end for 5b (`README.MD:54`) |
| Embedding cardinalities | `num_embeddings_per_feature` 26-vec, top entries 40 M (`recommendation_v2/torchrec_dlrm/README.MD:161`) | item 9.39 M, artist 1.29 M, album 3.37 M, uid 1.00 M, + 7 cross-features up to 100 M (`dlrm_v3/configs.py:40-48, 686-722`) |
| Required pre-processor pkg | none unusual | **`polars-u64-idx`** because yambda-5b exceeds polars' 32-bit row index (`training_recipe.md:44, 102-103`) |

### 1.4 Task formulation / supervision

| | v2 | v4 |
|---|---|---|
| Task | binary CTR (click / no-click) | sequential next-action ranking: given UIH, predict whether the candidate LISTEN event will be a "listen_plus" (`played_ratio ≥ 50%`) (`README.MD:103`) |
| Label | Criteo click label | `action_weight` bitmask on the candidate; supervision masked to `(supervision_bitmask & task_weight) > 0` with `task_weight = 1` (LP bit) → only `listen_plus` candidates supervise (`README.MD:103`) |
| Loss | BCE | BCE on `listen_plus` task |

### 1.5 Target metric

| | v2 | v4 |
|---|---|---|
| Target | **AUROC ≥ 0.80275** within 1 epoch on Criteo (`recommendation_v2/torchrec_dlrm/README.MD:173`) | `MetricsLogger.auc_threshold = 0.80275` (`yambda_5b.gin:107`). Same numeric value as v2 — likely inherited from the upstream DLRM-DCNv2 reporting convention rather than independently chosen for HSTU. Consumed in `dlrm_v3/utils.py:587-608` to log `time_to_auc_0.80275_sec` as soon as the `listen_plus` task's AUC crosses the threshold. Confirm with the proposing team whether this is the intended final benchmark target or a placeholder. |

### 1.6 Training hyperparameters

| | v2 (MLPerf example, 8 GPU) | v4 (`yambda_5b.gin`, 8 GPU) |
|---|---|---|
| Global batch | 65,536 (`recommendation_v2/torchrec_dlrm/README.MD:154`) | **8,192** (`batch_size=1024 × world_size=8`) (`yambda_5b.gin:1, 44`). Note `docs/training_recipe.md:65, 182` shows `32 × 8 = 256` — that doc has drifted; the gin file is the launch source of truth. |
| Epochs | 1 (`recommendation_v2/torchrec_dlrm/README.MD:163`) | 1 (`yambda_5b.gin:81`) |
| Dense optimizer | Adagrad, lr 0.005 (`recommendation_v2/torchrec_dlrm/README.MD:170-171`) | **Adam**, lr 1e-3, betas (0.95, 0.999), eps 1e-8 (`yambda_5b.gin:19-24`) |
| Sparse optimizer | (Adagrad on embeddings via TorchRec) | **RowWiseAdagrad**, lr 1e-3, betas (0.95, 0.999), eps 1e-8 (`yambda_5b.gin:27-32`) |
| Precision | fp32 (no bf16 flag in v2 example) | **bf16** mixed precision, gated on the TRITON HSTU kernel (`yambda_5b.gin:8`; `training_recipe.md:58, 109-111`) |
| Sequence length | n/a (non-sequential model) | `history_length=2039`, `max_seq_len=2048` (`yambda_5b.gin:74, 78`) |

### 1.7 Software stack

| | v2 | v4 (MI350X) | v4 (B200) |
|---|---|---|---|
| Container | none specified (bare AWS p4d, CUDA 11.0, NCCL 2.10.3) (`recommendation_v2/torchrec_dlrm/README.MD:37`) | `rocm/primus:v26.3` (`training_recipe.md:24`) | `nvcr.io/nvidia/pytorch:26.04-py3` (`training_recipe.md:132`) |
| GPU target | A100 40 GB | **MI350X** (`gfx950`, ROCm 7.2.1, 288 GiB HBM3e) | **B200** (`sm_100`, ~183 GiB HBM) |
| torch | TorchRec example era; CUDA 11.0 | `2.12.0+rocm7.2` (`training_recipe.md:38`) | `2.12.0a0` native NGC (CUDA 13.2) (`training_recipe.md:149`) |
| triton | not central | `3.6.0` (image native; required for HSTU TRITON backend) (`training_recipe.md:41`) | `3.6.0` (`training_recipe.md:150`) |
| fbgemm_gpu | TorchRec default | `fbgemm_gpu_nightly_rocm-2026.6.2` built from FBGEMM `10b77573` for `gfx950` (`training_recipe.md:42`) | same SHA, built for `sm_100` (`training_recipe.md:151`) |
| torchrec | (whatever TorchRec was current) | `1.7.0a0+bf55480` (`v2026.06.01.00`) (`training_recipe.md:43`) | `1.7.0.dev20260601+cu130` (`training_recipe.md:152`) |
| Launcher | `torchx … dist.ddp` | `scripts/launch_smoke_8gpu.sh` | `scripts/launch_smoke_8gpu.sh` |
| Key kernel | TorchRec EmbeddingBag + DCN | **HSTU TRITON jagged-attention** (`HSTU_HAMMER_KERNEL=TRITON`) (`training_recipe.md:71`) | same (`training_recipe.md:188`) |

---

## 2. HSTU model walkthrough

### 2.1 What HSTU is, in one paragraph

**HSTU = Hierarchical Sequential Transducer Units**, from the Meta paper
*Actions Speak Louder than Words* (ICML'24). It is a **decoder-only Transformer
variant**, redesigned for *recommendation* sequences (very long, very ragged,
heavy on categorical features). The block looks like a standard transformer
block superficially — attention + MLP — but two things are different from
GPT/SASRec attention:

1. **Pointwise SiLU instead of softmax** in the attention non-linearity (no
   log-sum-exp normalization).
2. **Gated output**: an extra projected stream `U` multiplies the attention
   output before the residual.

Everything else (residual connections, layer-norm, multi-head, positional
encoding, causal masking, KV-cache) is conventional transformer. The "S" in
STU = "Sequential Transducer Unit" = one HSTU block.

### 2.2 The composition: top-level model (DLRM-v3 / `DlrmHSTU`)

The full thing in `dlrm_hstu.py` is a small pipeline. Top-down:

```
KeyedJaggedTensor of raw ids
         │
         ▼
[1] TorchRec EmbeddingCollection           (≈150 G sparse params, sharded across GPUs)
         │  emits per-feature jagged embedding lookups
         ▼
[2] ContextualPreprocessor                 (interleaves UIH + appends candidate, adds
                                            positional / action / timestamp encodings)
         │  output: jagged sequence of length L per user, dim = transducer_embedding_dim
         ▼
[3] HSTUTransducer  ── STUStack of N HSTULayers (the "HSTU" attention blocks)
         │  output: contextualized per-position embedding
         ▼
[4] DefaultMultitaskModule                 (linear → BCE on listen_plus bit)
         │
         ▼
Per-anchor logit  →  BCE loss
```

For yambda-5b the per-dataset overrides in `dlrm_v3/configs.py:78-90, 346-425`
give:

| component | value | source |
|---|---|---|
| embedding tables | `item_id` 9.39 M × 512, `artist_id` 1.29 M × 512, `album_id` 3.37 M × 512, `uid` 1.00 M × 512, + 7 cross-features (e.g. `user_x_artist` 100 M × 512) | `dlrm_v3/configs.py:686-722` |
| embedding dim | 512 (`HSTU_EMBEDDING_DIM`) | `dlrm_v3/configs.py:33, 353` |
| HSTU layers | **5** (`hstu_attn_num_layers=5`) | `dlrm_v3/configs.py:82` |
| attention heads | 4 | `dlrm_v3/configs.py:79` |
| Q/K dim per head | 128 | `dlrm_v3/configs.py:81` |
| V/U (linear) dim per head | 128 | `dlrm_v3/configs.py:80` |
| transducer embedding dim | 512 | `dlrm_v3/configs.py:85, 354` |
| dropout | input 0.2, linear 0.1 | `dlrm_v3/configs.py:87-88` |
| max attention budget (model) | 8192 (yambda default; gin further caps to 2048 via `get_hstu_configs.max_seq_len = 2048` in `yambda_5b.gin:78`) | `dlrm_v3/configs.py:355` |
| task | `listen_plus`, BINARY_CLASSIFICATION, BCE | `dlrm_v3/configs.py:419-424` |

**Sparse-side parameter count, by table** (just the explicit ones; cross-features
add 282 M more rows × 512 dim ≈ 144 G params, which dominate):

```
item_id   :  9_390_624 × 512 ≈   4.81 B
artist_id :  1_293_395 × 512 ≈     662 M
album_id  :  3_367_692 × 512 ≈    1.72 B
uid       :  1_000_001 × 512 ≈     512 M
crosses   :       ~282 M × 512 ≈ 144.4 B   ← dominant
```

This is overwhelmingly an embedding-bound model — the dense HSTU stack (5
layers × ~1 M parameters each) is a rounding error next to the embedding
tables, which is why `make_optimizer_and_shard.hbm_cap_gb = 260` and why
TorchRec sharding is central.

### 2.3 Inside one STU (HSTU) layer

From `modules/stu.py:182-246, 292-355`. A single STU layer holds **four**
weight matrices, not the usual two (QKV + out):

```
_uvqk_weight   : (E, (hidden_dim·2 + attn_dim·2) · num_heads)
_uvqk_beta     : (...,)                bias for the above
_input_norm    : LayerNorm(E)
_output_weight : (hidden_dim · num_heads · 3, E)
_output_norm   : LayerNorm
```

Forward pass on input `x` of shape `[L, E]` (jagged):

#### 2.3.1 Fused U/V/Q/K projection

```
normed = LayerNorm(x)
[U | V | Q | K] = normed @ _uvqk_weight + _uvqk_beta     # one GEMM, then split
                                                          # U, V ∈ R^{H·hidden_dim}
                                                          # Q, K ∈ R^{H·attn_dim}
```

Compared to a regular transformer, you get an **extra projected stream `U`**.
`U` will gate the attention output later.

#### 2.3.2 HSTU attention (the core difference vs softmax attention)

Reference math, exactly as written in
`ops/pytorch/pt_hstu_attention.py:151, 167, 179, 182`:

```python
qk_attn = einsum("bhxa,bhya->bhxy", Q, K) * alpha   # alpha = 1 / sqrt(attn_dim)
qk_attn = F.silu(qk_attn) / max_seq_len             # ← pointwise SiLU, scalar divide
qk_attn = qk_attn * valid_attn_mask                 # mask (see 2.3.3)
attn    = einsum("bhxd,bhdv->bhxv", qk_attn, V)
```

Contrast with a vanilla transformer:

```python
qk   = (Q @ K.T) / sqrt(d)
qk   = softmax(qk + mask, dim=-1)                   # ← softmax normalises rows
attn = qk @ V
```

Two consequences of dropping softmax:

- **No row-wise normalization** → the per-key contribution is decoupled across
  positions. The paper argues this is *better* for recommendation, because a
  5-year-old "like" event shouldn't have its weight diluted just because the
  user has a longer history (which softmax would do).
- **Numerically more delicate**: the recipe warns *"`pt_hstu_attention`'s QK
  einsum backward overflows in bf16 at N > 1k and produces NaN at step 1; bf16
  is only safe with TRITON"* (`docs/training_recipe.md:109-111`). The TRITON
  kernel handles bf16 accumulation carefully; the reference PyTorch path
  doesn't.

#### 2.3.3 Custom attention mask (`_get_valid_attn_mask`, `pt_hstu_attention.py:32-84`)

HSTU supports four mask-combination knobs simultaneously:

- **causal**: lower triangle only (standard).
- **target-aware** (`num_targets`): the last `num_targets` positions are the
  candidate targets; their "row index" is clamped so all targets see the same
  prefix (the user's UIH) but cannot peek at each other.
- **max_attn_len** (sliding window): each position attends only to the previous
  `max_attn_len` events — bounds the receptive field for very long histories.
- **contextual_seq_len**: the first `contextual_seq_len` positions are
  *contextual* tokens (uid + cross-features). They are allowed to attend to
  everything (and everything attends back to them), regardless of causal order.
  This is how `uid` / `user_x_artist` etc. get full visibility despite living
  at the head of the sequence.

#### 2.3.4 Output: gated MLP

From `stu.py:336-354` → `hstu_compute_output`:

```
y = SwishLayerNorm(attn)             # SiLU(x · sigmoid(x)) then LN
y = concat([y, U]) @ _output_weight  # gating with the U stream
y = y · x + dropout                  # residual back to original x
```

The `U · y` gating is the second non-standard piece. It is reminiscent of
GLU / SwiGLU but applied to the *attention output*, not just an MLP.

#### 2.3.5 Stack

`STUStack` (`stu.py:426`) is just `nn.ModuleList` of N `STULayer`s applied
sequentially with the same jagged-tensor convention. No cross-layer fanciness.

### 2.4 "Transformer-style sequential attention over a UIH" — what the inputs actually look like

UIH = **User Interaction History**. For yambda, the input to one training
sample is one **anchor LISTEN event** plus that user's history. From
`README.MD:88-101` and `dlrm_v3/configs.py:399-418`:

```
sequence position:    0 .. 7   |   8 .. (L-2)                |  L-1
                      ─────────┼─────────────────────────────┼──────────
content:              contextual│  UIH (interleaved 3 pools)  │  candidate
                                │                              │
features per position:           uid, 7 cross-features (length-1 each)
                                  item_id, artist_id, album_id,
                                  action_weight (LP/LIKE/SKIP bitmask),
                                  action_timestamp, dummy_watch_time
                                                              candidate's:
                                                                item_candidate_id,
                                                                item_candidate_artist_id,
                                                                item_candidate_album_id,
                                                                item_query_time,
                                                                item_action_weight,
                                                                item_dummy_watchtime
```

The HSTU stack runs causal attention over this `L = 2048` sequence. The label
is the candidate's `listen_plus` bit (1 if `played_ratio ≥ 50%`, else 0), and
BCE is taken on the logit emitted at position `L-1`. So "transformer-style
sequential attention over UIH" literally means: the user's last ~2 k actions
are tokens, the candidate song is the last token, and a 5-layer HSTU
transformer predicts whether that candidate will be a `listen_plus`.

This is the conceptual jump from DLRMv2:

| | DLRMv2 (Criteo, v2) | HSTU (Yambda, v4) |
|---|---|---|
| Input shape | flat: 26 categorical + 13 dense features per ad impression | sequence of ~2 k past events per user, each a structured tuple |
| Mixing op | DCN: cross-products of feature vectors, then MLP | self-attention across positions (SiLU-gated, multi-head, causal) |
| Temporal modelling | none (each ad impression is i.i.d.) | central — masks, timestamps, action types are first-class |
| Depth | 1-shot (interaction arch + over-arch MLP) | 5 stacked HSTU blocks |
| "Why is the candidate good?" | low-rank cross of user/ad embeddings | attention over user's relevant past songs/artists/albums |

DLRMv2 is *wide-and-shallow* over tabular features. HSTU is *narrow-and-deep*
over a temporal sequence. Different paradigm.

### 2.5 Jagged attention — what it is and why it's used

A user's history length varies — yambda median is 2,695 events, max is 27,738
(`README.MD:65`). For a single training step you have a batch of B users with
very different sequence lengths. Two ways to lay this out on the GPU:

**Padded layout (standard transformer):**

```
input shape: [B, N_max, D]           e.g. [1024, 2048, 512]
```

This wastes compute proportional to `(N_max − N_user) / N_max` per row. On
yambda the average fill is ~1402/2037 ≈ 69%, so ~30% of every kernel is
multiplying zeros.

**Jagged layout (what HSTU uses):**

```
flat values  : [L_total, D]          L_total = Σ user_lengths   (≤ B · N_max)
offsets      : [B + 1]               cumulative starts, so user i occupies
                                     values[offsets[i] : offsets[i+1]]
```

`pt_hstu_attention.py:148, 183` shows the round-trip:

- `torch.ops.fbgemm.jagged_to_padded_dense(...)` only when calling into a dense
  einsum
- `torch.ops.fbgemm.dense_to_jagged(...)` on the way out

That's the reference path. The **TRITON jagged-attention kernel**
(`ops/triton/triton_hstu_attention.py`, dispatched in `ops/hstu_attention.py:27,
71`) skips the padded intermediate entirely: each Triton program handles one
user's `[N_user, N_user]` attention block directly, so:

- **No wasted FLOPs.** Empty positions never enter a GEMM.
- **No wasted memory.** No padded `[B, H, N_max, N_max]` attention scores buffer
  — that buffer alone would be `1024 · 4 · 2048 · 2048 · 2 bytes ≈ 34 GB` per
  step (at global batch 1024 × bf16).
- **Variable-length backward is correct without masking tricks.** The kernel
  iterates `[offsets[i], offsets[i+1])` per program; the gradient never touches
  non-existent positions.

This is *the* enabling optimization for the under-filled `like` pool to be
cheap. The README notes (`README.MD:132`): *"With the TRITON jagged-attention
backend the GPU only does work for the actual events, so the under-fill costs
sequence budget but not GPU compute"*. With a padded kernel, the unused 31% of
every sequence would cost real FLOPs.

Practically: jagged attention is a generic technique (it shows up in
FlashAttention's varlen variants too); HSTU's TRITON kernel is its
specialization with SiLU + gated output + the four-way mask.

---

## 3. Yambda-5b — size, contents, download, run

### 3.1 What's in it

[`yandex/yambda`](https://huggingface.co/datasets/yandex/yambda) on HuggingFace.
From `dlrm_v3/preprocess_public_data.py:233-245` + `README.MD:56-81`:

| field | value |
|---|---|
| Provider | Yandex Music recommendation logs |
| Sizes | yambda-50m, yambda-500m, **yambda-5b** (v4 uses 5b) |
| Events | 4.76 B interactions across 300 days |
| Users | 1.00 M unique |
| Items | 9.39 M songs (+ 1.29 M artists, 3.37 M albums) |
| Event types | `listen` / `like` / `dislike` / `unlike` / `undislike` (encoded as uint8 0–4) |
| Listen events also carry | `played_ratio` (used to derive the `listen_plus` label at 50% threshold) |
| Train / test split | Global Temporal Split: 300 days train, 30-min gap, 1 day test |

### 3.2 On-disk footprint after preprocessing (`README.MD:39-52`)

```
<DLRM_DATA_PATH>/
├── raw/5b/multi_event.parquet                    50 GB   (downloaded)
├── shared_metadata/
│   ├── artist_item_mapping.parquet               60 MB
│   ├── album_item_mapping.parquet                76 MB
│   └── embeddings.parquet                        18 GB   (unused by HSTU training)
└── processed_5b/
    ├── train_sessions.parquet                    47 GB   ← main training input
    ├── test_events.parquet                       152 MB
    ├── session_index.parquet                     600 MB
    ├── item_popularity.npy                        75 MB
    └── split_meta.json                            anchor + boundary stats
```

Plan for **~115 GB free disk** to do everything end-to-end (raw + shared +
processed). If you skip the unused `embeddings.parquet` (which the script
downloads anyway), you still need ~97 GB.

### 3.3 Download + preprocess

Both happen in one command. Download is via the `datasets` library
(HuggingFace), so you need internet and `pip install datasets`. From
`dlrm_v3/preprocess_public_data.py:276-317`:

```bash
pip install datasets polars-u64-idx pyarrow xxhash gin-config absl-py pandas

export DLRM_DATA_PATH=/your/big/disk/dlrm_data
mkdir -p "$DLRM_DATA_PATH"

cd /home/suachong/training/recommendation_v4
python3 -m generative_recommenders.dlrm_v3.preprocess_public_data \
  --dataset yambda-5b \
  --data-path "$DLRM_DATA_PATH"
```

Per `README.MD:54`: **~53 minutes end-to-end** for the 5b variant on a
reasonable box. For a quick smoke test substitute `--dataset yambda-50m`
(~2 min, ~1 GB on disk).

Critical: **install `polars-u64-idx`, not stock `polars`.** yambda-5b has
>4.29 B rows and overflows polars' default 32-bit row index silently
(`training_recipe.md:102-103`, `scripts/launch_smoke_8gpu.sh:13-20`).

### 3.4 Run training (8-GPU smoke)

From `scripts/launch_smoke_8gpu.sh` and `README.MD:9-22`.

**Inside the validated container** (recommended; everything's pre-staged):

```bash
docker exec yambda_8gpu bash -c \
  'cd /workspace/recommendation_v4 && bash scripts/launch_smoke_8gpu.sh'
```

Override data path / run name without editing the gin:

```bash
DLRM_DATA_PATH=/your/big/disk/dlrm_data \
RUN_NAME=my_experiment \
bash scripts/launch_smoke_8gpu.sh
```

**From scratch on a bare host**, you need to assemble the stack per
`docs/training_recipe.md`. The hard requirements are:

- **ROCm path**: `rocm/primus:v26.3`, torch `2.12.0+rocm7.2`, triton `3.6.0`,
  fbgemm_gpu built from commit `10b77573` for `gfx950`, torchrec
  `1.7.0a0+bf55480`. See `training_recipe.md:30-45`.
- **CUDA path**: `nvcr.io/nvidia/pytorch:26.04-py3`, native torch (do NOT
  reinstall), fbgemm_gpu built from same commit for `sm_100`, torchrec
  `1.7.0.dev20260601+cu130`. See `training_recipe.md:147-155`.

In both cases the actual launch is just:

```bash
python -m generative_recommenders.dlrm_v3.train.train_ranker \
    --dataset yambda-5b --mode train-eval
```

(plus `HSTU_HAMMER_KERNEL=TRITON` for CUDA; `=PYTORCH` is forced on ROCm in
the smoke script because the Triton kernel hits PassManager errors on some
shapes there — see `scripts/launch_smoke_8gpu.sh:31-33`. The PYTORCH fallback
gives ~190 ms/step baseline, not the ~52 ms primus-pinned number.)

### 3.5 What you'll see

Per `training_recipe.md:84-91`, on 8× MI350X in the optimal config:

- ~52 ms/step at global batch 256 (per the doc; gin says 8,192 — see §1.6 note)
- ~4,970 samples/sec
- ~7.6 days for one epoch over 3.23 B training anchors

If `auc_threshold = 0.80275` is the real benchmark target (still TBD),
`time_to_auc_0.80275_sec` will be logged as soon as the eval AUC on
`listen_plus` crosses it (`dlrm_v3/utils.py:587-608`).

---

## 4. TL;DR

- **HSTU ≈ decoder-only Transformer** with two tweaks: SiLU/N replaces softmax
  in attention, and a `U`-gated output replaces the standard MLP block.
- **DLRMv3 (yambda) = TorchRec embeddings → contextual preprocessor → 5
  stacked HSTU layers → BCE head on `listen_plus`.** Sparse tables (≈150 G
  params) dominate the model; the dense HSTU stack is tiny by comparison.
- **UIH = user interaction history.** Each sample is one anchor LISTEN event
  plus that user's last ~2 k events (LISTEN_PLUS / LIKE / SKIP, interleaved
  chronologically, gathered with a `L//3`-per-pool cap), and HSTU does causal
  self-attention across them.
- **Jagged attention** packs variable-length per-user sequences as
  `(flat_values, offsets)` instead of padding to `N_max`, so the Triton kernel
  never spends FLOPs on empty positions — essential because the average
  sequence is only 69% full on yambda.
- **Yambda-5b** is a 4.76 B-event / 1 M-user Yandex Music dataset on
  HuggingFace (`yandex/yambda`); downloading + preprocessing takes ~53 min and
  ~115 GB disk; run via
  `python -m generative_recommenders.dlrm_v3.train.train_ranker --dataset yambda-5b --mode train-eval`
  (or `scripts/launch_smoke_8gpu.sh`).

---

## 5. Open questions to bring back to the proposing team

1. **Target metric.** `yambda_5b.gin:107` reuses DLRMv2's `0.80275` AUC
   threshold. Is this the intended final v4 benchmark target, or a placeholder
   inherited from upstream? An HSTU model on a different dataset would
   normally need its own threshold chosen from a reference run.
2. **Batch size canonicalization.** `yambda_5b.gin:1` = `1024` per rank
   (global 8,192); `docs/training_recipe.md:65, 182` says `32` per rank
   (global 256). Which is the submission config?
3. **Convergence reference runs.** No `reference_results.md`-style table exists
   yet under `training/recommendation_v4`. Submission-quality v4 will need
   reference epochs-to-target numbers per dataset variant.
