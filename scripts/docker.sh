#!/usr/bin/env bash
# The Docker side of the platform: Keycloak (sign-in, realm ml-agents) and
# Langfuse (traces), with the Postgres/ClickHouse/Redis/MinIO they need.
#
#   scripts/docker.sh start   start them; returns once Keycloak serves the realm
#   scripts/docker.sh stop    stop them (data stays in the Docker volumes)
#
# `make docker-start` / `make docker-stop`; `make up` runs `start` itself when
# Docker is available.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }

compose() { if docker compose version >/dev/null 2>&1; then docker compose "$@"; else docker-compose "$@"; fi; }
SERVICES="postgres keycloak langfuse-web langfuse-worker clickhouse redis minio"
KC_URL="http://localhost:${KEYCLOAK_PORT:-8180}"

case "${1:-start}" in
  start)
    docker info >/dev/null 2>&1 || { echo "Docker is not running — start Docker first."; exit 1; }
    compose up -d $SERVICES
    printf "Waiting for Keycloak"
    for _ in $(seq 1 120); do
      curl -sf "$KC_URL/realms/ml-agents/.well-known/openid-configuration" >/dev/null && break
      printf "."; sleep 2
    done
    echo
    curl -sf "$KC_URL/realms/ml-agents/.well-known/openid-configuration" >/dev/null \
      || { echo "Keycloak did not come up — see: $(command -v docker-compose || echo docker compose) logs keycloak"; exit 1; }
    # A UI admin (ml-admin role) on a fresh realm; never touched once it exists.
    ONLY_IF_NEW=1 bash scripts/keycloak-user.sh "${ML_ADMIN_USER:-admin}" 1 "${ML_ADMIN_PASSWORD:-admin}"
    echo "Keycloak: $KC_URL/admin  (realm ml-agents)    Langfuse: http://localhost:3000"
    if [ -n "${ML_ADMIN_PASSWORD:-}" ]; then
      echo "UI admin sign-in: ${ML_ADMIN_USER:-admin} / (ML_ADMIN_PASSWORD from .env)"
    else
      echo "UI admin sign-in: ${ML_ADMIN_USER:-admin} / admin   — set ML_ADMIN_PASSWORD in .env on a server"
    fi
    ;;
  stop)
    compose stop
    ;;
  *)
    echo "usage: $0 start|stop"; exit 2
    ;;
esac
