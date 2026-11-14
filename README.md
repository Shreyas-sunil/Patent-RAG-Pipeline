# patent-rag: download → extract → parse → embed → index

**Full guide with architecture diagrams, the week plan and troubleshooting: `Patent_RAG_Pipeline_Guide.docx`.**

Builds the searchable index for your patent & standards RAG chatbot, following the
Patent AI 7-day plan (Qdrant + bge-m3 on a GPU server). One script per stage; run them in order.

| # | Script | What it does | Output |
|---|--------|--------------|--------|
| 1 | `01_inventory.py` | Counts the standards you already have: body, number, version, format, duplicates, scans | `inventory.csv` |
| 2a | `02_download_uspto.py` | US patents from the USPTO Open Data Portal weekly XML, filtered by your CPC codes | `corpus/patents/area-*/US….xml` |
| 2b | `03_download_bigquery.py` | Non-US patents (EP, WO, CN, JP, KR, IN…) from Google Patents on BigQuery | `corpus/patents/area-*/bigquery/*.jsonl` |
| 2c | `04_download_standards.py` | Missing 3GPP specs and chosen ETSI documents | `corpus/standards/3gpp/…`, `…/etsi/…` |
| 3 | `05_extract.py` | Unzips archives, converts `.doc` → `.docx`, optional OCR of scanned PDFs | files next to the originals |
| 4 | `06_parse.py` | Splits every document into chunks: one per claim, abstract, description windows, spec clauses | `chunks/<batch>.jsonl` + report |
| 5 | `07_build_index.py` | Embeds chunks with bge-m3 (dense + sparse) and stores them in Qdrant | Qdrant collection `corpus` |
| 6 | `08_search.py` | Asks the index a question and prints the best passages | — |
| all | `pipeline.py` | Runs every step in order in the background, with pause / continue / stop | `state/` |

Shared modules: `config.toml` (all settings), `common.py` (config, resumable downloader, CPC matching),
`parsers.py` (format parsers), `embedder.py` (bge-m3), `retrieve.py` (hybrid search to import from your chatbot).

---

## Run everything with one command: `pipeline.py`

After the one-time setup below, `pipeline.py` runs all the steps in order in the background. You can log out of SSH and it keeps going.

```bash
python pipeline.py run          # start, or resume where it stopped
python pipeline.py status       # ✔ done  ▶ running  ■ stopped  ✖ failed, plus the last log lines
python pipeline.py logs         # follow the live log (Ctrl+C only stops watching)
python pipeline.py pause        # freeze the current step
python pipeline.py continue     # unfreeze it (also: resume)
python pipeline.py stop         # stop cleanly; `run` later continues from the same step
python pipeline.py skip         # abandon the current step, go on to the next
python pipeline.py reset        # forget progress (files, chunks and index are kept)
python pipeline.py steps        # show the exact command each step runs
```

Useful `run` options:

```bash
python pipeline.py run --from parse            # start at a step (re-runs it and everything after)
python pipeline.py run --only parse index      # just these steps
python pipeline.py run --skip bigquery         # leave a step out
python pipeline.py run --keep-going            # don't halt when a step fails
python pipeline.py run --loop 168              # repeat every 168 h = weekly updates
python pipeline.py run --foreground            # stay in this terminal
```

Step order: `inventory → uspto → bigquery → standards → extract → parse → index → check`.
Choose which steps run in the `[pipeline]` section of `config.toml`. BigQuery defaults to `"dry-run"`, which only reports the cost; set it to `"run"` once you've seen the number.

How it behaves:
- **Pause** freezes the step's processes in place. A paused indexer still holds GPU memory, so use **stop** if you need the GPU back.
- **Stop**, a crash or a reboot never loses work. `run` restarts the interrupted step, and each script skips what it already finished (downloaded files, parsed files, the indexer's checkpoint).
- A failed step halts the pipeline (unless `--keep-going`). Fix the cause, then `run` retries that step.
- State and the log live in `state/` (`pipeline.json`, `pipeline.log`).

The individual scripts below still work on their own. Use them for testing each stage before the first full run.

---

## 0. One-time setup (GPU server, Ubuntu)

```bash
unzip patent-rag.zip && cd patent-rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Qdrant vector database
sudo apt install -y docker.io tmux libreoffice-writer      # libreoffice only for old .doc files
sudo docker run -d --name qdrant --restart unless-stopped -p 6333:6333 \
  -v $HOME/qdrant_data:/qdrant/storage qdrant/qdrant
curl http://localhost:6333        # prints Qdrant's version
```

Credentials:
- **USPTO:** free API key from data.uspto.gov (sign in → My ODP → API key), then `export USPTO_API_KEY=...` (put it in `~/.bashrc`).
- **BigQuery:** a Google Cloud project with billing, then `gcloud auth application-default login`.

Edit **`config.toml`**: your areas and their CPC prefixes and targets, USPTO date range, BigQuery project, 3GPP releases/series, ETSI documents.

Run every long step inside tmux (`tmux new -s jobs`, detach with Ctrl+B then D, `tmux attach -t jobs` to return).

## 1. Inventory your standards

Copy your ~32k standards documents into `corpus/standards/` (any sub-folders), then:

```bash
python 01_inventory.py
```

Read the summary: duplicates to delete, old versions to drop, scans that need OCR.

## 2. Download

```bash
python 02_download_uspto.py --list           # see the weekly files it will walk (newest first)
python 02_download_uspto.py --max-files 1    # test on one week; check corpus/patents/area-*/
python 02_download_uspto.py                  # full run, hours; stop/restart any time
python 02_download_uspto.py --status         # per-area counts vs. targets

python 03_download_bigquery.py               # DRY RUN: prints the TB it would scan, bills nothing
python 03_download_bigquery.py --run         # runs the query and exports .jsonl

python 04_download_standards.py --dry-run    # lists missing 3GPP/ETSI specs
python 04_download_standards.py
```

Notes
- Each patent goes into the first area whose CPC prefix matches. Each tool stops an area at its `target`.
- BigQuery bills by bytes scanned and the full-text columns are large. The dry run shows the exact size; `max_bytes_billed` in `config.toml` is a hard cap. One query covers all areas, so add areas *before* running it.
- BigQuery keeps one document per patent family (granted first) and, by default, skips families that have a US member, since those come from the USPTO tool.
- Indian (InPASS) patents have no bulk service. Drop PDFs you fetch manually into `corpus/patents/area-<name>/`, named by publication number (e.g. `IN202341012345A.pdf`).
- Don't point these tools at the Google Patents or Espacenet websites; their terms forbid bulk scraping.

## 3. Extract

```bash
python 05_extract.py            # unzip 3GPP zips, convert .doc → .docx
python 05_extract.py --ocr      # also OCR scanned PDFs (sudo apt install -y ocrmypdf tesseract-ocr)
```

## 4. Parse into chunks

Always test before the full run. Fixing a pattern after parsing 60,000 files means parsing them all again.

```bash
python 06_parse.py --test corpus/patents/area-wireless/US11234567B2.xml   # inspect one file
python 06_parse.py --sample 20                                            # 20 random files + stats
python 06_parse.py                                                        # everything new
```

The report must show **patents with claims ≥ 95%**. Re-run `06_parse.py` whenever new downloads land; it only parses new files and writes a new batch file.

## 5. Embed and index (overnight)

Stop Ollama/the chatbot first on a 24 GB card so the GPU is free.

```bash
python 07_build_index.py --limit 10000      # speed test: prints chunks/s and a full-run estimate
python 07_build_index.py                    # every chunks/*.jsonl not yet indexed
python 07_build_index.py --status
```

- Out of GPU memory → `--encode-batch 16` (or 32).
- Interrupted → just run it again; it resumes from its checkpoint (`chunks/indexed.sqlite`).
- Re-indexing a file overwrites the same points (ids come from document + section), never duplicates.
- `--recreate` deletes the collection and starts over.

**Done when** `--status` shows the Qdrant point count equal to the number of chunk lines (`wc -l chunks/*.jsonl`).

## 6. Check retrieval

```bash
python 08_search.py "beam failure recovery on contention-free random access"
python 08_search.py "RRC reconfiguration procedure" --kind standards
python 08_search.py "thermal runaway detection" --area area-batteries --rerank
```

## Using it from your chatbot

```python
from retrieve import Retriever
r = Retriever(rerank=True)                         # bge-m3 + bge-reranker-v2-m3
hits = r.search(question, k=8, kind=None, area=None)
# each hit: doc, section, text, title, kind, area, date, cpc, assignees, depends_on, score
context = "\n\n".join(f"[{i+1}] {h['doc']} — {h['section']}\n{h['text'][:2000]}" for i, h in enumerate(hits))
```

This plugs straight into Step 7 of the plan (`answer.py`): replace its `search` import with the `Retriever` above.

## Weekly updates

```bash
python 02_download_uspto.py     # picks up new weekly files (raise targets in config.toml to keep growing)
python 05_extract.py && python 06_parse.py && python 07_build_index.py
```

## Try the pipeline without a GPU

`--fake` uses hash vectors instead of bge-m3 and `--qdrant-path` uses an embedded Qdrant folder instead of Docker.
Good for checking parsing and wiring on a laptop; the search quality is meaningless.

```bash
python 06_parse.py
python 07_build_index.py --fake --qdrant-path index_test
python 08_search.py "your question" --fake --qdrant-path index_test
```

## Files written

```
corpus/      downloaded and extracted documents
downloads/   USPTO weekly zips (deleted after splitting unless keep_zips = true)
manifests/   uspto.sqlite, standards.sqlite: what's downloaded, failed, pending
chunks/      <batch>.jsonl, <batch>.report.txt, done.txt, errors.log, indexed.sqlite
inventory.csv
```

---

## Documentation

| Document | What it covers |
|----------|----------------|
| [`Patent_RAG_Pipeline_Guide.docx`](Patent_RAG_Pipeline_Guide.docx) | Full guide to this pipeline: architecture diagrams, the week plan, troubleshooting |
| [`docs/Patent_AI_Build_Plan.docx`](docs/Patent_AI_Build_Plan.docx) | Build plan for the full patent & standards AI (large corpus, Qdrant, GPU embedding, Unsloth fine-tuning) |
| [`docs/Patent_RAG_Chatbot_Build_Plan.docx`](docs/Patent_RAG_Chatbot_Build_Plan.docx) | Original 7-day build plan for a local RAG chatbot (Ollama, ChromaDB, Streamlit) |
