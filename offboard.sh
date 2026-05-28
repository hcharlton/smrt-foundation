#!/usr/bin/env bash
# Usage:
#   ./offboard.sh              # dry-run (default; prints what would transfer)
#   ./offboard.sh --execute    # transfer + verification scaffolding
#
# Stages the project (minus data/ and regeneratable caches) into the Gefion
# SFTP outbox at /dcai/projects/cu_0030/xfer/chache/outbox/, which is pulled
# to GenomeDK by rclone (once daily at 7am).
#
# In --execute mode:
#   1. Check the source for files unreadable to you. rsync would skip these
#      silently and they would never reach GenomeDK, so they are reported up
#      front rather than going missing.
#   2. rsync into the outbox with --chmod=Da+rX,Fa+r so the SFTP daemon (a
#      different user) can read every file. Applied on the transfer, not via a
#      pre-chmod of the source, so jobs writing after start are still covered,
#      the live tree is not mutated, and dry-run reflects reality.
#   3. chmod -R a+rX the outbox as a belt-and-suspenders guarantee.
#   4. Write SHA256SUMS into the outbox so the transfer can be verified on
#      GenomeDK with `sha256sum -c SHA256SUMS`.
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
echo "chmod:  Da+rX,Fa+r on destination (so SFTP daemon can read every file)"

# 1. Readability check (read-only, runs in both modes). Files listed here are
#    unreadable to you and so will be skipped by rsync and absent from the
#    manifest: a silent gap. find's own "Permission denied" lines (if any)
#    flag directories you cannot traverse, which block rsync the same way.
echo
echo "== source readability check =="
UNREADABLE=$(find "$PROJECT_ROOT" \
    \( -name data -o -name .venv -o -name __pycache__ -o -name '*.egg-info' \
       -o -name .pytest_cache -o -name .gwf \) -prune -o \
    -type f ! -readable -print || true)
if [[ -n "$UNREADABLE" ]]; then
    N_UNREADABLE=$(wc -l <<< "$UNREADABLE")
    echo "WARNING: $N_UNREADABLE source file(s) unreadable to you; they will NOT transfer:"
    head -20 <<< "$UNREADABLE"
    if (( N_UNREADABLE > 20 )); then
        echo "  ... ($((N_UNREADABLE - 20)) more)"
    fi
    echo "Fix perms/ownership on these before trusting the transfer."
else
    echo "ok: every transferable source file is readable"
fi

echo
echo "== rsync into outbox =="
rsync -ah --info=progress2 --stats --partial \
  --chmod=Da+rX,Fa+r \
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

if [[ -n "$DRY_RUN_FLAG" ]]; then
    echo
    echo "(dry-run) skipping outbox chmod + checksum manifest; re-run with --execute"
    exit 0
fi

# 3. Guarantee every staged file is daemon-readable, independent of rsync's
#    perm-update-on-skip behavior for files already present from earlier runs.
echo
echo "== chmod -R a+rX outbox =="
chmod -R a+rX "$OUTBOX"

# 4. Manifest, written after the chmod and then made readable itself, so it
#    travels with the data. Verify on GenomeDK: sha256sum -c SHA256SUMS
echo
echo "== checksum manifest (reads the full staged tree, may take a few minutes) =="
( cd "$OUTBOX" && find . -type f ! -name SHA256SUMS -exec sha256sum {} + | sort -k2 > SHA256SUMS )
chmod a+r "$OUTBOX/SHA256SUMS"
echo "wrote $OUTBOX/SHA256SUMS ($(wc -l < "$OUTBOX/SHA256SUMS") files)"
