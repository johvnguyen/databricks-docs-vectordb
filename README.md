# databricks-docs-vectordb

A data pipeline that scrapes the public Databricks documentation and loads it
into a Qdrant vector database. It's meant as the base for a RAG study aid for
Databricks certifications, one that answers questions grounded in, and cited
to, the official docs.

## Status

The data pipeline is finished, and the project is wrapped up there for now.

| Phase | | |
|---|---|---|
| 1. Scraping | done | `scrape` |
| 2. Parsing and cleaning | done | `parse` |
| 3. Chunking and embedding | done | `embed` |
| 4. Vector database ingestion | done | `index` |
| 5. Retrieval layer and evaluation | deferred | |
| 6. RAG system | deferred | |
| 7. Maintenance and operations | deferred | |

A full run covers about 6,057 pages. They become 52,817 chunks and 52,903
vectors, with OpenAI's `text-embedding-3-small` at 1,536 dimensions, in one
Qdrant collection.

## How it works

Four stages, each a command group. Each stage reads the previous one's output
and writes its own:

| Stage | Reads | Writes |
|---|---|---|
| `scrape` | robots.txt and the sitemaps of the `aws/en` docs and the REST API reference | `data/manifest.sqlite`, `data/raw/` (one HTML file per page) |
| `parse` | `data/raw/` | `data/parsed.sqlite`: markdown chunks with page, heading and section metadata |
| `embed` | `data/parsed.sqlite` | `data/embeddings.sqlite`: one vector per chunk, or per slice of an oversized chunk |
| `index` | `data/embeddings.sqlite` | the Qdrant collection `databricks_docs_<provider>_<model>_<dimensions>` |

Every stage is safe to interrupt with Ctrl-C and to run again. A rerun
redoes only what changed, so running the whole pipeline again keeps it in
step with the docs:

- `scrape fetch --refresh` re-checks pages with conditional requests.
- `parse run` re-parses changed pages.
- `embed run` re-embeds changed chunks.
- `index run` replaces changed pages and removes pages that are gone.

Each Qdrant point holds only the lookup key back into `embeddings.sqlite`
(`chunk_id`, `sub_index`) and the two filter fields, `url` and `section`. A
search hit is read back from `embeddings.sqlite`.

## Setup

The project runs in a VS Code dev container. Docker on the host runs it next
to a Qdrant sidecar (`compose.yaml`).

1. Copy `.env.example` to `.env` and fill it in:

   | Variable | Needed by |
   |---|---|
   | `QDRANT_API_KEY` | the dev container: it won't open without it. Qdrant uses it as its API key, and every `index` command sends it |
   | `DATABRICKS_DOCS_CONTACT` | `scrape enumerate` and `scrape fetch`; it goes in the User-Agent |
   | `OPENAI_API_KEY` | `embed run`, and `index check`'s text queries |

   Optional: `QDRANT_URL` (default `http://qdrant:6333`) and
   `DATABRICKS_DOCS_DATA_DIR` (default `data/`).

2. Open the folder in the dev container. Qdrant 1.19.1 starts with it:
   - Its data lives in `data/qdrant/`.
   - Port 6333 is published on every host interface, so the API key matters.
   - The web dashboard is at `http://localhost:6333/dashboard`.

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (Python 3.12).

## Running the pipeline

```bash
uv run databricks-docs-vectordb scrape enumerate
uv run databricks-docs-vectordb scrape fetch
uv run databricks-docs-vectordb parse run
uv run databricks-docs-vectordb embed run
uv run databricks-docs-vectordb index run
uv run databricks-docs-vectordb index verify --complete
uv run databricks-docs-vectordb index check
```

`index run` checks first that `embed verify --complete` is clean. Before it
replaces or removes anything, it asks once whether to save a snapshot of the
collection to `data/qdrant_snapshots/`. Pass `--save-old` or `--no-save-old`
to answer in advance; without a terminal, one of them is required.

Runs log to `data/logs/`.

## Commands

Every stage has a `status` and a `verify` command:

- `status` shows where things stand.
- `verify` checks the stage's output. It exits 1 on any problem, and
  `--complete` also counts unfinished work as a problem.
- None of them needs the network, except `index status` and `index verify`,
  which reach Qdrant.

| Command | What it does |
|---|---|
| `scrape enumerate` | fetch robots.txt and the sitemaps, and update the manifest |
| `scrape fetch [--limit N \| --sample N] [--refresh] [--retry-failed] [--rate R] [--concurrency C]` | download pending pages |
| `scrape status`, `scrape verify [--complete] [--by-section]` | what was fetched; check the saved pages |
| `parse run [--limit N \| --sample N] [--refresh]` | parse new or changed pages into chunks |
| `parse status`, `parse verify [--complete] [--by-section]` | chunk counts and sizes; check the chunk store |
| `embed backfill-tokens [--model M]` | fill in token counts with the configured (or another) tokenizer |
| `embed run [--limit N \| --sample N] [--rate R] [--tpm T] [--concurrency C]` | embed every chunk that needs it |
| `embed status`, `embed verify [--complete]` | what's embedded, failed or left; check the embeddings store |
| `index run [--refresh] [--save-old \| --no-save-old]` | bring the Qdrant collection up to date |
| `index status [--by-section]` | the collection's point and page counts, and other models' collections |
| `index verify [--complete] [--by-section]` | check every point against the embeddings store, `parsed.sqlite` and the manifest |
| `index check` | sanity checks and a report: self-retrieval, recall@10, sample text queries, filtered queries |

Each command's `--help` explains it in full.

## Tests

```bash
uv run pytest            # everything that needs no network
uv run pytest -m network # the live docs site and the Qdrant sidecar
./run_tests.sh           # both, from inside the dev container
```

`run_tests.sh` waits for the sidecar, runs every test, then deletes any
leftover `test_` collections. Tests against the sidecar use throwaway
`test_<random>` collections, never the real one. Pre-commit runs ruff.

## Layout

- `databricks_docs_vectordb/`: the package, one subpackage per stage
  (`scraping`, `parsing`, `embedding`, `indexing`) and the CLI in `cli.py`
- `scripts/run_pipeline.sh`: runs every stage in order, from an empty `data/`
  or an existing one, then verifies and checks the collection
- `scripts/snapshot_qdrant.py`: saves a snapshot of the Qdrant collections to
  `data/qdrant_snapshots/` (`uv run python scripts/snapshot_qdrant.py`; its
  docstring shows how to restore one)
- `tests/`: one test directory per stage
- `data/`: everything the pipeline writes (not committed)
- `design-docs/`: the project plan and the per-phase design docs (kept out
  of git)

## Future work

These phases are deferred to a future day:

- **Retrieval layer and evaluation:** a retrieval interface, an evaluation
  set of questions with expected pages, metrics and a baseline. `index check`
  is a starting point.
- **RAG system:** an external LLM answering from retrieved docs, with
  citations, behind a simple chat UI.
- **Maintenance and operations:** scheduled re-runs and monitoring. Every
  stage already updates incrementally.
