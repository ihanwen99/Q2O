#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
cd "$ROOT"

ok()   { printf '  [ok]   %s\n' "$1"; }
fail() { printf '  [fail] %s\n' "$1"; exit 1; }

python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
  && ok "Python $(python -c 'import platform; print(platform.python_version())')" \
  || fail "Python 3.10 or newer is required"

python -c 'import flask, natsort, psycopg2, dimod, dwave.system, dwave.optimization' \
  && ok "Python dependencies import" || fail "Install requirements.txt first"

if [[ -n "${DWAVE_API_TOKEN:-}" || -s "${DWAVE_CONFIG_FILE:-$HOME/.config/dwave/dwave.conf}" ]]; then
  ok "D-Wave credentials configured"
else
  fail "Set DWAVE_API_TOKEN or create ~/.config/dwave/dwave.conf"
fi

python - <<'PY' && ok "PostgreSQL reachable with pg_hint_plan" || fail "Check the database settings in config.py"
from backend.postgres import open_postgres_connection
conn = open_postgres_connection()
assert conn is not None
with conn.cursor() as cur:
    cur.execute("LOAD 'pg_hint_plan'")
    cur.execute("SELECT count(*) FROM title")
    assert cur.fetchone()[0] > 0
conn.close()
PY

for q in benchmark/JOB/*/; do
  [[ -s "$q/cardinalities.json" && -s "$q/selectivities.json" ]] && compgen -G "$q*.sql" >/dev/null \
    || fail "Missing query inputs in $q"
done
ok "JOB query inputs present"
echo "Setup OK."
