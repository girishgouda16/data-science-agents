#!/usr/bin/env bash
# Add a sign-in account to the ml-agents realm. For many users, connect your
# directory instead (Keycloak admin → User federation / Identity providers).
#
#   make user u=alice                  one-time password; they choose their own at first sign-in
#   make user u=dana admin=1           plus ml-admin (may act on anyone's runs and models)
#   make user u=bob p=Secret123        a fixed password
#
# ONLY_IF_NEW=1 leaves an existing account untouched (make docker-start uses
# it for the default admin, so a changed password is never reset).
set -euo pipefail
cd "$(dirname "$0")/.."
U="${1:-}"; ADMIN="${2:-}"; PASS="${3:-}"
[ -n "$U" ] || { echo "usage: make user u=<username> [admin=1] [p=<password>]"; exit 2; }
TEMP=0
[ -n "$PASS" ] || { PASS="$(python3 -c 'import secrets; print(secrets.token_urlsafe(9))')"; TEMP=1; }

compose() { if docker compose version >/dev/null 2>&1; then docker compose "$@"; else docker-compose "$@"; fi; }

# kcadm runs inside the Keycloak container, as its bootstrap admin. A failure
# there stops this script (set -e) with Keycloak's own error message.
out="$(compose exec -T -e U="$U" -e P="$PASS" -e A="$ADMIN" -e T="$TEMP" -e NEW="${ONLY_IF_NEW:-0}" keycloak bash -c '
  set -e
  kc=/opt/keycloak/bin/kcadm.sh; cfg=--config=/tmp/kcadm.config
  $kc config credentials $cfg --server http://localhost:8080 --realm master \
      --user "$KC_BOOTSTRAP_ADMIN_USERNAME" --password "$KC_BOOTSTRAP_ADMIN_PASSWORD" >/dev/null 2>&1 \
      || { echo "cannot sign in to the Keycloak admin API (KEYCLOAK_ADMIN / KEYCLOAK_ADMIN_PASSWORD)" >&2; exit 1; }
  if $kc get users $cfg -r ml-agents -q username="$U" -q exact=true | grep -q "\"username\""; then
    [ "$NEW" = 1 ] && { echo "exists"; exit 0; }
  else
    $kc create users $cfg -r ml-agents -s username="$U" -s enabled=true \
        -s email="$U@ml-agents.local" -s emailVerified=true -s firstName="$U" -s lastName="Data Scientist Agents" >/dev/null
  fi
  if [ "$T" = 1 ]; then $kc set-password $cfg -r ml-agents --username "$U" --new-password "$P" --temporary
  else $kc set-password $cfg -r ml-agents --username "$U" --new-password "$P"; fi
  [ "$A" = 1 ] && $kc add-roles $cfg -r ml-agents --uusername "$U" --rolename ml-admin
  echo "saved"')"
[ "$out" = exists ] && { echo "'$U' already exists — left unchanged"; exit 0; }

echo "Account '$U'$( [ "$ADMIN" = 1 ] && echo ' (ml-admin)') — password: $PASS$( [ "$TEMP" = 1 ] && echo ' (one-time: they choose their own at first sign-in)')"
