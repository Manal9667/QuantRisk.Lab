#!/usr/bin/env bash
# QuantumRiskLab container entrypoint.
#   api         -> migrate, seed (idempotent), then serve the API (default)
#   experiments -> run the reproducible experiment sweep and exit
#   migrate     -> run migrations only
set -euo pipefail

CMD="${1:-api}"

wait_for_db() {
  echo "Waiting for PostgreSQL at ${QRL_DB_HOST:-postgres}:${QRL_DB_PORT:-5432}..."
  python - <<'PY'
import os, time
import psycopg
host = os.environ.get("QRL_DB_HOST", "postgres")
port = int(os.environ.get("QRL_DB_PORT", "5432"))
user = os.environ.get("QRL_DB_USER", "qrl")
pwd = os.environ.get("QRL_DB_PASSWORD", "qrl_password")
db = os.environ.get("QRL_DB_NAME", "quantumrisklab")
for attempt in range(60):
    try:
        with psycopg.connect(host=host, port=port, user=user, password=pwd, dbname=db, connect_timeout=2):
            print("PostgreSQL is ready.")
            break
    except Exception as exc:  # noqa: BLE001
        print(f"  ...not ready ({attempt+1}/60): {exc}")
        time.sleep(2)
else:
    raise SystemExit("PostgreSQL did not become ready in time.")
PY
}

case "$CMD" in
  api)
    wait_for_db
    python -m quantumrisklab.db.init_db --seed-data
    exec uvicorn quantumrisklab.api.main:app \
      --host "${QRL_API_HOST:-0.0.0.0}" --port "${QRL_API_PORT:-8000}"
    ;;
  experiments)
    wait_for_db
    python -m quantumrisklab.db.init_db --seed-data
    exec python scripts/run_experiments.py "${@:2}"
    ;;
  migrate)
    wait_for_db
    exec python -m quantumrisklab.db.init_db
    ;;
  *)
    exec "$@"
    ;;
esac
