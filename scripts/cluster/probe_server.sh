#!/usr/bin/env bash
# Read-only probe of a GPU server: prints hardware, container runtimes and scheduler info.
# It changes nothing. Usage:  bash probe_server.sh > server_probe.txt
set -u

section() { printf '\n===== %s =====\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }

section "host"
hostname; uname -srm
[ -r /etc/os-release ] && grep -E '^(NAME|VERSION)=' /etc/os-release

section "cpu / memory"
nproc 2>/dev/null
free -g 2>/dev/null | head -2

section "gpu"
if have nvidia-smi; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
  nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
  have nvcc && nvcc --version | tail -1
else
  echo "nvidia-smi not found"
fi

section "disk (home and common scratch locations)"
df -h "$HOME" 2>/dev/null | tail -1
for d in "${SCRATCH:-}" /scratch /tmp "/scratch/$USER" "/work/$USER"; do
  [ -n "$d" ] && [ -d "$d" ] && df -h "$d" | tail -1 | sed "s|^|$d: |"
done
have quota && quota -s 2>/dev/null | tail -3

section "python"
for p in python3 python3.12 python3.11; do have "$p" && echo "$p: $($p --version 2>&1)"; done
have uv && uv --version
have conda && conda --version
have module && echo "environment modules available (module avail)"

section "containers"
for c in docker podman apptainer singularity; do
  if have "$c"; then echo "$c: $($c --version 2>&1 | head -1)"; else echo "$c: not found"; fi
done
have docker && (docker info >/dev/null 2>&1 && echo "docker daemon reachable by this user" || echo "docker present but daemon not reachable (no permission?)")
id -nG

section "scheduler"
if have sinfo; then sinfo -s 2>/dev/null | head -20; else echo "no SLURM (sinfo not found)"; fi

section "network"
for url in https://huggingface.co https://pypi.org https://registry-1.docker.io; do
  if have curl; then
    code=$(curl -s -o /dev/null -m 8 -w '%{http_code}' "$url")
    echo "$url -> HTTP $code"
  fi
done

section "who else is using the gpus"
have nvidia-smi && nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null | head -10
