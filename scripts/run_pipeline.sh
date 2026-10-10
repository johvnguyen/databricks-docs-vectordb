#!/usr/bin/env bash
# Runs the whole pipeline: scrapes the docs, parses and chunks them, embeds
# the chunks, then loads the Qdrant collection, verifies it and checks it.
#
# Works from an empty data/ directory, and with whatever is already in it:
# each stage creates the files and folders it needs, leaves the rest alone,
# and redoes only what is new or changed, so a run over a finished data/ is
# quick. data/qdrant/, Qdrant's own storage, is never touched here.
#
#   scripts/run_pipeline.sh
#   scripts/run_pipeline.sh --no-save-old   # unattended
#
# The steps are the CLI's own commands, run in order from the repo root, and
# the script stops at the first one that fails:
#
#   1. scrape enumerate          robots.txt and the sitemaps
#   2. scrape fetch              every page: about 50 minutes for some 6,000
#                                pages at the default 2 per second
#   3. parse run                 pages into chunks
#   4. embed run                 chunks into vectors: about $0.32 of OpenAI
#                                tokens from scratch
#   5. index run                 vectors into Qdrant
#   6. index verify --complete
#   7. index check
#
# Before step 1 it checks everything the steps will need, so that a missing
# key doesn't surface an hour in: DATABRICKS_DOCS_CONTACT, OPENAI_API_KEY,
# QDRANT_API_KEY, and that Qdrant answers with that key. Each is read as the
# CLI reads it: the environment wins, even when blank; otherwise .env. Blank
# counts as missing.
#
# `index run` asks whether to save a snapshot before it replaces anything;
# --save-old or --no-save-old answers in advance, and without a terminal one
# of them is needed. Exits with the code of the step that failed (130 for
# Ctrl-C), or 0. Every step is safe to interrupt, so running the script again
# picks up where it stopped.
set -uo pipefail
cd "$(dirname "$0")/.."

usage() {
  echo "usage: scripts/run_pipeline.sh [--save-old | --no-save-old]"
}

save_flag=""
for arg in "$@"; do
  case "$arg" in
    --save-old | --no-save-old)
      if [ -n "$save_flag" ] && [ "$save_flag" != "$arg" ]; then
        usage >&2
        echo "error: --save-old and --no-save-old can't be combined" >&2
        exit 2
      fi
      save_flag="$arg"
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      echo "error: unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

trim() {
  local value=$1
  value=${value#"${value%%[![:space:]]*}"}
  printf '%s' "${value%"${value##*[![:space:]]}"}"
}

setting() {
  local name=$1 value=""
  if [ -n "${!name+set}" ]; then
    value=${!name}
  elif [ -f .env ]; then
    value=$(sed -n -E "s/^[[:space:]]*(export[[:space:]]+)?$name[[:space:]]*=//p" .env | tail -n 1)
    value=$(trim "$value")
    value=${value#[\"\']}
    value=${value%[\"\']}
  fi
  trim "$value"
}

problems=0
problem() {
  echo "error: $1" >&2
  problems=$((problems + 1))
}

[ -n "$(setting DATABRICKS_DOCS_CONTACT)" ] ||
  problem "DATABRICKS_DOCS_CONTACT is not set (or blank); scraping sends it in the User-Agent. Add it to .env"
[ -n "$(setting OPENAI_API_KEY)" ] ||
  problem "OPENAI_API_KEY is not set (or blank); embed run needs it. Add it to .env"
qdrant_key=$(setting QDRANT_API_KEY)
qdrant_url=$(setting QDRANT_URL)
qdrant_url=${qdrant_url:-http://qdrant:6333}
if [ -z "$qdrant_key" ]; then
  problem "QDRANT_API_KEY is not set (or blank); every index step needs it. Add it to .env"
else
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 10 -H "api-key: $qdrant_key" \
    "${qdrant_url%/}/collections" || true)
  case "$code" in
    200) ;;
    401 | 403) problem "Qdrant at $qdrant_url rejected QDRANT_API_KEY (HTTP $code)" ;;
    000) problem "Qdrant at $qdrant_url did not answer" ;;
    *) problem "Qdrant at $qdrant_url answered HTTP $code" ;;
  esac
fi
[ "$problems" -eq 0 ] || exit 1

trap 'printf "\nstopped: interrupted; run this script again to pick up where it stopped\n" >&2; exit 130' INT

steps=(
  "scrape enumerate"
  "scrape fetch"
  "parse run"
  "embed run"
  "index run${save_flag:+ $save_flag}"
  "index verify --complete"
  "index check"
)
for i in "${!steps[@]}"; do
  step=${steps[$i]}
  printf '\n== %d/%d: databricks-docs-vectordb %s\n' "$((i + 1))" "${#steps[@]}" "$step"
  uv run databricks-docs-vectordb $step
  status=$?
  if [ "$status" -ne 0 ]; then
    printf '\nstopped: `%s` exited %d; run this script again to pick up where it stopped\n' \
      "$step" "$status" >&2
    exit "$status"
  fi
done
printf '\ndone: all %d steps finished\n' "${#steps[@]}"
