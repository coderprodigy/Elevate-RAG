# 05 — Waterloo GPU servers (open models + databases)

**When this is needed:** day 4 of the compressed schedule, before the Phase 3 reference generator
(a mid-size open model) and Phase 4 graph work. Phases 0–2 don't need it. When we reach that point I'll
pause and ask you for the details in §2.

## 1. Target setup

```
 Mac (your laptop)                                   Waterloo GPU server
 ─────────────────                                   ───────────────────────────────────────────
 hgrag code, eval runs,          SSH tunnel          vLLM  :8000  → open generator / extractor (OpenAI-compatible API)
 embedders/rerankers (MPS)  ═══════════════════════► vLLM  :8001  → second model (optional)
 LLM gateway → localhost:8000                        Qdrant :6333 / :6334   (Docker, Podman or Apptainer)
 stores    → localhost:6333 / 7687                   Neo4j  :7474 / :7687
```

- The code always talks to `localhost`, so the same configuration works whether a service runs on the
  Mac or on the server.
- Services bind to `127.0.0.1` on the server (never `0.0.0.0`), so the SSH tunnel is the only way in.
- Model weights and database volumes go on the server's large scratch disk, not your home quota.

## 2. What I'll ask you for (gather these now if you can)

Run the probe script on the server and paste me its output. It only **reads** system information and
changes nothing.

```bash
# from the Mac, inside the project folder
scp scripts/cluster/probe_server.sh <you>@<server>:~/
ssh <you>@<server> 'bash ~/probe_server.sh' > server_probe.txt
```

Also tell me:

| Item | Example |
|---|---|
| Hostname(s) and how you log in | `ecetesla1.uwaterloo.ca` (example only), SSH key or password + 2FA |
| Off-campus access | Is the university VPN required? |
| Shared or dedicated GPUs? | Can we hold a GPU for several hours? Is there a usage policy? |
| Scheduler | Interactive use, or SLURM jobs only? |
| Containers | Is Docker, rootless Docker, Podman or Apptainer allowed? |
| Disk | Path to large scratch storage and the quota |

**Never paste passwords or private keys into the chat.** I'll write commands for you to run where a
credential is needed.

## 3. SSH setup on the Mac (one-off)

```bash
ssh-keygen -t ed25519 -C "ece750"            # skip if you already have a key
ssh-copy-id <you>@<server>                    # if the server allows key login
```

Add to `~/.ssh/config`:

```
Host uwgpu
    HostName <server>
    User <you>
    ServerAliveInterval 60
    # Forward the project ports; open the tunnel with: ssh -N uwgpu
    LocalForward 8000 127.0.0.1:8000
    LocalForward 8001 127.0.0.1:8001
    LocalForward 6333 127.0.0.1:6333
    LocalForward 7687 127.0.0.1:7687
    LocalForward 7474 127.0.0.1:7474
```

Then `ssh -N uwgpu` in a spare terminal keeps the tunnel open.

## 4. Model serving with vLLM (I'll script this in `scripts/cluster/` on day 4)

Rough GPU memory needs. Actual numbers depend on context length and KV cache:

| Model | Precision | Approx. weight memory | Fits on |
|---|---|---|---|
| Gemma 4 12B | bf16 | ~24 GB | 40–48 GB GPU (or 24 GB at fp8) |
| Qwen3.6-35B-A3B (MoE) | fp8 | ~35 GB | 48 GB GPU; bf16 needs 80 GB |
| Qwen3.6-35B-A3B | 4-bit AWQ/GPTQ | ~20 GB | 24 GB GPU (record the quantisation in the report) |
| Qwen3-Embedding-8B | bf16 | ~16 GB | 24 GB GPU |

The plan is a Python virtual environment on the server (`uv venv && uv pip install vllm`), started under
`tmux`, or as a SLURM job if that's the only option:

```bash
vllm serve <hf-model-id> --host 127.0.0.1 --port 8000 \
  --served-model-name generator-mid --max-model-len 16384 \
  --gpu-memory-utilization 0.85 --api-key "$VLLM_API_KEY"
```

On the Mac, `.env` gets `VLLM_BASE_URL=http://localhost:8000/v1` and `VLLM_API_KEY=...`. The
`waterloo_vllm` provider in `configs/models.yaml` already reads these.

## 5. Databases on the server

In order of preference:

1. **Docker or rootless Docker / Podman:** `docker compose -f docker/compose.core.yaml up -d`, using the
   same files as on a Mac.
2. **Apptainer** (common on shared clusters with no Docker), running the same images:

   ```bash
   apptainer run --bind $SCRATCH/qdrant:/qdrant/storage docker://qdrant/qdrant:<pinned-tag>
   ```

   Neo4j works the same way with `--bind` for `/data` and `/logs`.
3. **No containers:** Qdrant and Neo4j both ship standalone binaries or tarballs. Otherwise we fall back
   to FAISS + NetworkX, which needs no server at all.

Benchmark fairness (Phase 6): every engine gets the same CPU and memory limits. Under Docker that's
Compose `cpus`/`mem_limit`. Under Apptainer or SLURM it's `--cpus-per-task` and `--mem`, or `taskset`
on an interactive machine.

## 6. If server access is delayed

Use a hosted endpoint for the open models ([04_compute_for_open_models.md](04_compute_for_open_models.md))
and OrbStack on the Mac for the `core` stack. Nothing in the code changes, only `configs/models.yaml`
and `.env`.
