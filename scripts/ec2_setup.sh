#!/usr/bin/env bash
# One-time host setup for an Ubuntu 24.04 EC2 instance (see docs/DEPLOYMENT.md).
#
#   bash scripts/ec2_setup.sh            # Docker + Compose + make/git
#   WITH_UV=1 bash scripts/ec2_setup.sh  # also uv, to run without the app image
#
# Idempotent: safe to re-run. It installs packages and adds the current user to the
# `docker` group. It never formats disks, deletes files or opens firewall ports.
set -euo pipefail

UV_VERSION="0.11.8"   # keep in step with .github/workflows/ci.yml

if [[ "$(. /etc/os-release && echo "${ID}-${VERSION_ID}")" != "ubuntu-24.04" ]]; then
  echo "warning: written for Ubuntu 24.04; continuing anyway" >&2
fi

sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  docker.io docker-compose-v2 make git ca-certificates curl
sudo systemctl enable --now docker

if ! id -nG "$USER" | grep -qw docker; then
  sudo usermod -aG docker "$USER"
  echo "added $USER to the docker group: log out and back in (or run 'newgrp docker')"
fi

if [[ "${WITH_UV:-0}" == "1" ]] && ! command -v uv >/dev/null 2>&1; then
  curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
fi

if [[ ! -f .env && -f .env.example ]]; then
  echo "next: cp .env.example .env, then set the database passwords and any API keys"
fi
docker --version
docker compose version
