# Multi-Node Training Enablement (yambda-5b, MI350X / Broadcom bnxt_re RoCE)

How N-node (N×8-GPU) distributed training was brought up for the yambda-5b HSTU
ranker on the `meta64` cv350 cluster, the hard problems solved, and **exactly
which settings are cluster/fabric-specific** so this can be reused or re-tuned
when the underlying network changes.

Companion to [`perf_opt.md`](./perf_opt.md) and [`training_recipe.md`](./training_recipe.md).
The single entry point is [`scripts/launch_slurm.sh`](../scripts/launch_slurm.sh);
the Python side is `generative_recommenders/dlrm_v3/train/{train_ranker,utils}.py`.

---

## TL;DR

- Multi-node works over **real RDMA** (RoCEv2 on 8× Broadcom bnxt_re HCAs).
  2-node = `world_size=16`, clean `rc=0`, ~7.7–8.0k `global_sps` (≈1.28× of
  1-node 6.2k; weak scaling, per-rank batch fixed).
- The one non-obvious blocker was a **userspace RDMA provider ABI mismatch**
  inside the container, fixed with an `LD_PRELOAD`/`LD_LIBRARY_PATH` **overlay**
  of the host's matched `rdma-core` (no container lib surgery).
- Everything is one script with three auto-detected phases
  (`orchestrate` → `provision` → `worker`) plus small Python changes for global
  ranks. All cluster-specific knobs are env-overridable and tagged
  `[CLUSTER-SPECIFIC]` in the script.

---

## Architecture: one script, three phases

`launch_slurm.sh` self-dispatches by context (`LAUNCH_SLURM_PHASE`, else
auto-detected via `/.dockerenv`):

| Phase | Runs on | Does |
|---|---|---|
| `orchestrate` | SLURM batch host | Resolve rendezvous (`MASTER_ADDR/PORT`), ensure container on every node (calls `provision`), then `docker exec` the `worker` phase on every node (one srun task per node). |
| `provision` | each compute node (host) | Ensure the `yambda_primus` container is up (baked image if present, else base image + pip), stage the host RDMA overlay on NFS. |
| `worker` | inside the container | Derive topology, set NCCL/RDMA env, apply the RDMA overlay, spawn this node's 8 GPU ranks via `train_ranker`. `NNODES==1` => legacy single-node path unchanged. |

Why one script: multi-node enablement is then a single committable file. The
worker phase is also what the streaming-e2e supervisor invokes directly
(single-node, already inside the container), so the production path is unchanged.

```
sbatch --nodes=N launch_slurm.sh
        │  (batch host: orchestrate)
        ├─ srun: provision  ──> docker container up + RDMA overlay staged   (×N nodes)
        └─ srun: docker exec launch_slurm.sh (worker)                       (×N nodes)
                   │ in container: topology + NCCL/RDMA env + LD overlay
                   └─ python train_ranker  ──> 8 local ranks  ──> RCCL rendezvous over RDMA
```

---

## The hard problems (lessons learned)

### 1. RDMA provider ABI mismatch — the core blocker

**Symptom:** multi-node RCCL died at init with
`ibv_create_qp ... Bad address`.

**Root cause:** the container image (`rocm/primus:v26.3`) ships an **older**
userspace `rdma-core` (v34, `libbnxt_re-rdmav34.so`) than the **host kernel**
bnxt_re driver's uapi (host `rdma-core` v61 / `libbnxt_re-rdmav59.so`). The v34
provider enumerates the HCAs and creates *shallow* QPs fine, but **faults when
creating a deep send queue** — RCCL uses `max_send_wr=256`. Verified with a
parameterized verbs probe: v34 `create_qp` is OK at depth ≤16 and faults at ≥64;
the host v59 provider works at **every** depth. So it is purely the **userspace
provider**, not the kernel or the fabric (a 2-node RoCEv2 RDMA-write test passes
on the stock stack, and bare-metal RCCL benchmarks run fine with the host libs).

**Fix (no container surgery):** the `provision` phase stages the host's matched
`rdma-core` on shared NFS (`$OVERLAY`):

```
$OVERLAY/lib/libibverbs.so.1          # host libibverbs v61
$OVERLAY/lib/libibverbs.so -> .so.1   # UNVERSIONED symlink (critical, see below)
$OVERLAY/lib/libnl-3.so.200, libnl-route-3.so.200
$OVERLAY/lib/libibverbs/<providers>.so   # incl. libbnxt_re-rdmav59.so
```

The `worker` phase makes RCCL load it at runtime:

```bash
export LD_LIBRARY_PATH="$OVERLAY/lib:$OVERLAY/lib/libibverbs:$LD_LIBRARY_PATH"
export LD_PRELOAD="$OVERLAY/lib/libibverbs.so.1:$LD_PRELOAD"
```

We do **not** modify the container's system libs — only this process tree's
`LD_*`. Single-node and other users keep the stock stack.

### 2. The UNVERSIONED `libibverbs.so` symlink is mandatory

An earlier overlay attempt set `LD_LIBRARY_PATH` but still failed with
`Bad address`. Reason: at `import torch` the ROCm stack pulls in the
**unversioned** soname `libibverbs.so` (not `libibverbs.so.1`). If the overlay
only has `libibverbs.so.1`, that unversioned lookup misses the overlay, falls
through to the **container's** old lib, which then occupies the `libibverbs.so.1`
slot — so RCCL's later `dlopen("libibverbs.so.1")` binds the v34 stack and
`create_qp(256)` faults again. The overlay **must** expose
`libibverbs.so -> libibverbs.so.1`. With it (verified via `/proc/<pid>/maps`),
the process maps **only** the host lib. `LD_PRELOAD` is belt-and-braces so the
host lib claims the soname slot first.

### 3. Two network planes — pin TCP bootstrap, RDMA for data

The container is `--network=host`, so RCCL sees **all** host interfaces and, left
to auto-detect, picks the wrong one. These nodes expose:
- `benic1p1..benic8p1` — per-GPU point-to-point RoCE links on `192.168.{1..8}.x/31`.
  These are **not node-routable** for plain TCP; the very first bring-up **hung**
  in `init_process_group` because RCCL tried the TCP bootstrap over a
  non-routable `192.168.x` backend addr.
- `fenic0` — the routable front-end (`10.190.x`).

So we split the planes explicitly:
- `NCCL_SOCKET_IFNAME=fenic0` → TCP bootstrap/rendezvous over the routable NIC.
- `NCCL_IB_HCA=bnxt_re0..7` → RDMA **data** over the 8 RoCE HCAs (the RoCEv2
  fabric *is* reachable rail-to-rail at the RDMA layer even though plain IP is not).

### 4. Minimal proven bnxt_re NCCL config

The minimal set proven on these nodes (matches cmcknigh's bare-metal RCCL
benchmarks): `NCCL_IB_GID_INDEX=3` (RoCEv2 IPv4 GID), `NCCL_IB_TC=104` (RoCE
lossless / PFC traffic class). **Do not** add the heavy
`QPS_PER_CONNECTION / ECE / DMABUF` block — that belongs to a different
(ionic AINIC) fabric and is counterproductive on bnxt_re. GPU-Direct RDMA
(`NCCL_NET_GDR_LEVEL`) is left **off**: it needs DMABUF/peermem, unavailable
in-container here, so RCCL stages through host memory (still real RDMA).

### 5. Rendezvous must be resolved on the host

The container image has **no SLURM client** (`scontrol` absent). So the
`orchestrate` phase resolves `MASTER_ADDR` (first host of the allocation) and a
deterministic `MASTER_PORT` (`20000 + job_id % 20000`, same on all nodes) **on
the host** and forwards them into the container via `docker exec -e`.

### 6. Global rank derivation (Python)

`mp.start_processes` hands out a node-local `local_rank` (0..7). Every downstream
consumer (data sharding, checkpoint I/O, metrics) needs the **global** rank:

```python
rank = node_rank * gpus_per_node + local_rank   # train_ranker._main_func
device = torch.device(f"cuda:{local_rank}")      # CUDA device stays node-local
```

Also: `make_optimizer_and_shard(local_world_size=gpus_per_node)` so the TorchRec
planner respects the intra-node GPU count, and `MetricsLogger(world_size=...)`
gets the live world size (the gin default of 8 would mis-normalize multi-node).
`NNODES==1` makes `rank == local_rank` — identical to the old single-node path.

### 7. `$0` is the staged `slurm_script`, not the repo path

For an sbatch batch script, `$0` =
`/var/spool/slurmd/job<ID>/slurm_script` (node-local), so deriving the script /
repo path from `$0` gives a path that **doesn't exist on other nodes** (`bash
$SELF` → "No such file", and the worker's `cd $REPO` → exit 127). The
`orchestrate` phase instead resolves the real shared-NFS path from SLURM:

```bash
SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | grep -oP 'Command=\K\S+')
# fallbacks: $SLURM_SUBMIT_DIR/scripts/launch_slurm.sh, then $SELF
REPO=$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)
```

### 8. `srun ... bash -c "…"` host-vs-remote expansion

Inside the double-quoted srun command string, **plain `$VAR` expands now on the
batch host** (values computed in orchestrate: `$MASTER_ADDR`, `$SCRIPT_PATH`, …)
while **`\$VAR` is deferred to each compute node** (`\$SLURM_NODEID`,
`\$(hostname)`) where the per-node SLURM env lives. Mixing these up sends every
rank the wrong node id.

### 9. `memlock` ulimit for QP registration

`docker run --ulimit memlock=-1:-1` is **required** — RDMA QP memory
registration needs unlimited locked memory. A container started with the default
8 MB memlock fails QP creation regardless of the overlay.

### 10. Provisioning & the image-bake caveat

Fresh nodes otherwise re-download a **6.1 GB** ROCm torch wheel + pip + build
torchrec-from-git every time. The script supports a pre-baked image
(`docker commit` → NFS tar → `docker load` offline). **Caveat:** the committed
image is **~127 GB** (ROCm base is huge), so the full-image NFS tar is impractical
(loading it can be slower than re-downloading 6 GB). For true download-avoidance
prefer a **local pip wheelhouse** (`pip install --no-index --find-links` from
~8 GB of NFS wheels) or a **local registry** (ships only the ~35 GB delta layer).
The bake hook is left in (`BAKE_IMAGE=1`) but defaults off; provisioning falls
back to base-image + pip.

### Debunked theory (do not re-introduce)

An earlier claim that the container's rdma-core was "too old → 0 devices /
Bad address" and needed an **in-place lib copy** was a red herring: the "0
devices" came from a *broken in-place copy* of the host EL9 libs (mixing v34
tooling that links `IBVERBS_PRIVATE_34` with host v61 libs breaks symbol-version
lookup). The stock container enumerates all 8 HCAs fine. The real issue is only
the deep-QP create path; the fix is the **LD overlay**, never in-place surgery.

---

## Cluster-specific settings — change these when the fabric/hardware changes

All are env-overridable and tagged `[CLUSTER-SPECIFIC]` in `launch_slurm.sh`
(`grep '\[CLUSTER-SPECIFIC\]' scripts/launch_slurm.sh`).

| Setting | Default (meta64) | What it is | How to find the right value |
|---|---|---|---|
| `#SBATCH --partition` | `meta64` | scheduler partition | `sinfo` |
| bind mounts + default paths | `/home/chcai`, `/apps/chcai` | repo + scratch, **must be shared/NFS on all nodes** | `df -h`, cluster docs |
| `IMAGE` | `rocm/primus:v26.3` | base container (GPU arch + ROCm version) | vendor image registry |
| docker `--device` | `/dev/kfd /dev/dri` (AMD) | GPU passthrough | NVIDIA: `--gpus all` / nvidia runtime |
| `--ulimit memlock` | `-1` | locked mem for RDMA QP | keep `-1` for any RDMA fabric |
| `TORCH_IDX` / torch,vision,audio | `rocm7.2`, `2.12.0+rocm7.2` … | ROCm-version'd wheels | `download.pytorch.org/whl/<rocmX>` |
| `FBGEMM_WHL` | gfx950 wheel on NFS | GPU-arch fbgemm | build/stage per arch |
| `NCCL_SOCKET_IFNAME` | `fenic0` | **routable** host NIC for TCP bootstrap | `ip -br addr` (pick the routable one; NOT the per-GPU RDMA NICs) |
| `NCCL_IB_HCA` | `bnxt_re0..7` | RDMA HCA device names | `ibv_devices` (vendor: `mlx5_*`, `ionic_*`, …) |
| `NCCL_IB_GID_INDEX` | `3` | RoCEv2 IPv4 GID index | `show_gids` (v1/v2 & IPv4/IPv6 differ per port) |
| `NCCL_IB_TC` | `104` | RoCE lossless / PFC traffic class | fabric/switch admin |
| `RDMA_OVERLAY` (+ provider .so) | `/apps/chcai/rdma_host_el9_new` | host rdma-core overlay | only needed if container rdma-core < host kernel uapi; else set `RDMA_OVERLAY=` to disable. Stage the host's matching `/usr/lib64/libibverbs/<provider>.so` |

**Different NIC vendor (e.g. Mellanox `mlx5`)** typically means: change
`NCCL_IB_HCA` names, re-check `NCCL_IB_GID_INDEX`/`NCCL_IB_TC`, and the RDMA
overlay is often **unnecessary** (Mellanox userspace in the image usually matches
the host) — set `RDMA_OVERLAY=` to skip it.

**Emergency fallback:** `NCCL_NET_TRANSPORT=socket` disables IB and runs
allreduce over TCP (`fenic0`). Functional but ~100–200× slower; use only to
isolate a fabric problem.
