#!/usr/bin/env bash
# Usage:
#   ./offboard.sh              # dry-run (default; prints what would transfer)
#   ./offboard.sh --execute    # perform the real transfer
#
# Stages the project (minus data/ and regeneratable caches) into the
# Gefion SFTP outbox at /dcai/projects/cu_0030/xfer/chache/outbox/.
# That outbox syncs to GenomeDK once daily at 7am.
set -euo pipefail

PROJECT_ROOT="/dcai/projects/cu_0030/smrt-foundation"
OUTBOX="/dcai/projects/cu_0030/xfer/chache/outbox/smrt-foundation"

if [[ "${1:-}" == "--execute" ]]; then
    DRY_RUN_FLAG=""
    echo "mode:   EXECUTE"
else
    DRY_RUN_FLAG="--dry-run"
    echo "mode:   DRY-RUN (use --execute to actually transfer)"
fi
echo "source: $PROJECT_ROOT/"
echo "dest:   $OUTBOX/"

rsync -ah --info=progress2 --stats --partial \
  $DRY_RUN_FLAG \
  --exclude='data/' \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='*.egg-info/' \
  --exclude='.pytest_cache/' \
  --exclude='.gwf/' \
  --exclude='.DS_Store' \
  "$PROJECT_ROOT/" \
  "$OUTBOX/"
