#!/usr/bin/env bash
# Resets the pipeline's data, so that scripts/run_pipeline.sh can rebuild it
# from scratch: deletes every file under data/, keeping the folders, and
# every collection in the Qdrant database. Shows what would go, and asks
# before deleting anything.
#
#   scripts/clean_data.sh
#
# data/qdrant/ is skipped here. It is Qdrant's own storage, owned by root and
# in use by the running server, so it is cleared through Qdrant's API
# instead (scripts/clear_qdrant.py, which reads QDRANT_API_KEY like the CLI
# does).
#
# Everything else under data/ goes: the stores, the raw pages, the logs, and
# any saved snapshots in data/qdrant_snapshots/. Exits 1, deleting nothing,
# if Qdrant can't be listed, if the answer isn't yes, or if there is no
# terminal to ask on.
set -uo pipefail
cd "$(dirname "$0")/.."

# Every file under data/ except data/qdrant/, one per line.
data_files() {
  [ -d data ] || return 0
  find data -path data/qdrant -prune -o ! -type d -print
}

# Count and size of the files under one folder of data/ (or its top level).
summarize() {
  local label=$1
  shift
  local count bytes
  count=$(find "$@" ! -type d -print 2>/dev/null | wc -l)
  bytes=$(find "$@" ! -type d -printf '%s\n' 2>/dev/null | awk '{s += $1} END {print s + 0}')
  [ "$count" -gt 0 ] &&
    printf '  %-22s %7d file(s) %8s\n' "$label" "$count" "$(numfmt --to=iec-i --suffix=B "$bytes")"
}

echo "files under data/ (the folders are kept; data/qdrant/ is cleared through Qdrant):"
total=$(data_files | wc -l)
if [ "$total" -eq 0 ]; then
  echo "  none"
else
  for dir in data/*/; do
    [ "$dir" = "data/qdrant/" ] && continue
    summarize "${dir#data/}" "$dir"
  done
  summarize "(data/ itself)" data -mindepth 1 -maxdepth 1
fi
echo
uv run python scripts/clear_qdrant.py --dry-run || exit 1
echo

if [ ! -t 0 ]; then
  echo "not deleting anything: there is no terminal to ask on" >&2
  exit 1
fi
read -r -p "Delete these $total file(s) and every Qdrant collection? [y/N] " answer
case "$answer" in
  y | Y | yes | YES | Yes) ;;
  *)
    echo "nothing deleted"
    exit 1
    ;;
esac

uv run python scripts/clear_qdrant.py || exit 1
if [ -d data ]; then
  find data -path data/qdrant -prune -o ! -type d -exec rm -f -- {} +
fi
echo "deleted $total file(s) under data/;"
