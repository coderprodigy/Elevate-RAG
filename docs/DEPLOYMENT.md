# Deploying on AWS EC2

For laptops, GPU workstations and university SLURM clusters see [RUNNING.md](RUNNING.md). **For
the scaled research runs on a GPU instance (vLLM, no API fees), follow the step-by-step guide in
[README §6](../README.md#6-run-the-scaled-version-on-aws-ec2-gpu).** This page covers EC2
security, the Docker image and CPU-only / API-model deployments.

Elevate-RAG is a batch research tool (a CLI), not a web service: on EC2 you run long experiment
jobs, not a public endpoint. Nothing needs inbound traffic except SSH.

```
EC2 instance (Ubuntu 24.04)
├── docker compose  →  Qdrant, Neo4j (+ Milvus, pgvector, Memgraph for E4)   ports bound to 127.0.0.1
└── elevate-rag image (CLI)  --network host  →  reaches the DBs on localhost
        └── /data  ←  EBS data volume: datasets, indexes, caches, runs, HF model cache
```

## 1. Choose an instance

These are starting points. Check current AWS prices and size from your own pilot run
(`elevate-rag index pilot` extrapolates cost and time from 5 % of a corpus).

| Workload | Suggested instance | Why |
|---|---|---|
| API models (Claude / OpenAI) + reference DBs (Qdrant + Neo4j) | `m7i.xlarge` (4 vCPU, 16 GB) | the CPU work is BM25, reranking batches and the DBs |
| E4 with every database at once | `m7i.2xlarge` (8 vCPU, 32 GB) | five engines with identical CPU/RAM limits (`ELEVATE_DB_*`) |
| Local open models (vLLM / Ollama) | a GPU instance sized to the model | serve the model separately; point `base_url` at it |

Storage: a **100 GB gp3 root volume** (Docker images and DB volumes live under `/var/lib/docker`)
plus a separate **gp3 data volume** for `/data`. That volume survives instance replacement.

## 2. Security

* **Security group:** inbound SSH (22) from your IP only. Nothing else. The database ports bind
  to `127.0.0.1` by default (`ELEVATE_BIND_HOST`), so they are unreachable from outside even if a
  rule is too broad. To open the Neo4j browser, use an SSH tunnel:
  `ssh -L 17474:localhost:17474 -L 17687:localhost:17687 ubuntu@<host>`.
* **Secrets:** keep them in `.env` on the instance (`chmod 600 .env`), or fetch them at start-up
  from AWS SSM Parameter Store / Secrets Manager via an instance IAM role. `.env` is git-ignored
  and excluded from the Docker build context, so keys never enter the image or the repository.
* **Database passwords:** replace every `change-me` value in `.env` before the first `make db-up`.
* **Spending:** `budget.max_usd` in the pipeline config is a hard cap checked before every paid
  call. It is `0` in the shipped configs; raise it deliberately.

## 3. Set up the host

```bash
git clone <your-repo-url> elevate-rag && cd elevate-rag
bash scripts/ec2_setup.sh            # Docker, Compose, make, git (Ubuntu 24.04); idempotent
newgrp docker                        # or log out and back in

# Data volume (CHECK THE DEVICE NAME with lsblk first; mkfs erases the device)
lsblk
sudo mkfs -t xfs /dev/nvme1n1        # only for a NEW, empty volume
sudo mkdir -p /mnt/elevate && sudo mount /dev/nvme1n1 /mnt/elevate
sudo chown 10001:10001 /mnt/elevate  # the image runs as uid 10001
# add the volume to /etc/fstab (by UUID, with nofail) so it mounts after a reboot

cp .env.example .env && chmod 600 .env && nano .env   # passwords, API keys
```

## 4. Start the databases and build the image

```bash
make db-up                    # Qdrant + Neo4j      (PROFILE=all for E4: + Milvus, pgvector, Memgraph)
make db-ps                    # wait until every service is "healthy"
make image                    # elevate-rag:latest  (EXTRAS="databases local-models" adds torch)
make image-smoke              # offline demo inside the image, no network: must finish OK
```

## 5. Run

Define a shell alias so every command runs in the image with the same mounts:

```bash
alias erag='docker run --rm --network host --env-file .env -v /mnt/elevate:/data -v "$PWD/configs:/app/configs:ro" elevate-rag:latest'

erag env                                                   # versions, DBs, credentials present
erag data prepare all                                      # → /mnt/elevate/data
erag experiment validate configs/experiments/E2_retrieval_modes.yaml
erag experiment estimate configs/experiments/E2_retrieval_modes.yaml
erag experiment run configs/experiments/E2_retrieval_modes.yaml --dry-run
```

For long jobs, run detached and follow the logs (JSON logs are on by default in the image):

```bash
docker run -d --name e2 --network host --env-file .env -v /mnt/elevate:/data \
  -v "$PWD/configs:/app/configs:ro" elevate-rag:latest \
  experiment run configs/experiments/E2_retrieval_modes.yaml
docker logs -f e2
```

Runs are resumable: if the instance stops or a job is interrupted, re-run the same command and
completed questions are skipped. Run ids do not depend on storage paths, so a run directory
copied between your laptop and EC2 still resumes.

### Without Docker (uv)

```bash
WITH_UV=1 bash scripts/ec2_setup.sh
uv sync --locked --extra databases
export ELEVATE_DATA_DIR=/mnt/elevate/data ELEVATE_CACHE_DIR=/mnt/elevate/cache \
       ELEVATE_INDEX_DIR=/mnt/elevate/indexes ELEVATE_RUNS_DIR=/mnt/elevate/runs
uv run elevate-rag env
```

## 6. Day-2 operations

| Task | Command |
|---|---|
| Update the code | `git pull && make image` |
| Stop the databases (data kept) | `make db-down` |
| Copy results to your laptop | `rsync -av ubuntu@<host>:/mnt/elevate/runs/ ./runs/` |
| Back up | snapshot the data EBS volume |
| Save money | stop the instance when idle; EBS volumes persist |

Record the instance type, `ELEVATE_DB_CPUS` / `ELEVATE_DB_MEMORY` and image tag in the report
appendix: database benchmarks are only comparable under identical resources.
