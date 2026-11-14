"""Step 5 - embed the chunks with bge-m3 and store them in Qdrant.

Each chunk gets a dense vector (meaning) and a sparse vector (keywords) from one
GPU pass. Both go into one Qdrant collection, with dense vectors compressed to
int8 so ~10M chunks fit in RAM. Dependent claims are embedded together with the
text of their parent claim, so "The method of claim 1, wherein..." still matches.

Point ids come from (doc, section), so re-running a batch overwrites instead of
duplicating. Progress is checkpointed: if the run stops, start it again and it
continues where it left off.

Start Qdrant first (once):
    docker run -d --name qdrant --restart unless-stopped -p 6333:6333 \
        -v $HOME/qdrant_data:/qdrant/storage qdrant/qdrant

    python 07_build_index.py --limit 10000            # speed test: note chunks/sec
    python 07_build_index.py                          # index every chunks/*.jsonl not yet indexed
    python 07_build_index.py chunks/batch-xyz.jsonl   # one file
    python 07_build_index.py --status
    python 07_build_index.py --fake --qdrant-path index_test   # pipeline test, no GPU/Docker needed
"""
import argparse
import itertools
import json
import pathlib
import sqlite3
import sys
import time
import uuid

from qdrant_client import QdrantClient, models as m

from common import load_config
from embedder import DIM, Encoder

PAYLOAD_FIELDS = ("doc", "section", "kind", "area", "text", "depends_on", "title", "date",
                  "cpc", "assignees", "source")


def point_id(doc, section):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc}::{section}"))


def connect(a):
    if a.qdrant_path:
        return QdrantClient(path=a.qdrant_path)
    return QdrantClient(url=a.qdrant_url, timeout=300)


def ensure_collection(q, name, recreate=False):
    if recreate and q.collection_exists(name):
        q.delete_collection(name)
    if q.collection_exists(name):
        return
    q.create_collection(
        name,
        vectors_config={"dense": m.VectorParams(size=DIM, distance=m.Distance.COSINE, on_disk=True)},
        sparse_vectors_config={"sparse": m.SparseVectorParams(modifier=None)},
        quantization_config=m.ScalarQuantization(scalar=m.ScalarQuantizationConfig(
            type=m.ScalarType.INT8, quantile=0.99, always_ram=True)),
        optimizers_config=m.OptimizersConfigDiff(indexing_threshold=20000),
    )
    for field in ("doc", "kind", "area", "cpc", "source", "date"):
        try:
            q.create_payload_index(name, field, m.PayloadSchemaType.KEYWORD)
        except Exception:
            pass                                  # local mode ignores payload indexes
    print(f"Created collection '{name}'")


def chunk_stream(path, start_line=0):
    """Yield (doc_start_line, chunk) with embed_text set, grouped by document."""
    def lines():
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= start_line and line.strip():
                    try:
                        yield i, json.loads(line)
                    except json.JSONDecodeError:      # half-written line from a stopped parse
                        print(f"  skipping unreadable line {i + 1}", file=sys.stderr)

    for _, grp in itertools.groupby(lines(), key=lambda x: x[1]["doc"]):
        grp = list(grp)
        doc_start = grp[0][0]
        claims = {c["section"]: c["text"] for _, c in grp if c["section"].startswith("claim")}
        for _, c in grp:
            parent = claims.get(f"claim {c.get('depends_on')}", "") if c.get("depends_on") else ""
            head = f"{c['doc']} | {c.get('title', '')} | {c['section']}".replace(" |  |", " |")
            c["embed_text"] = f"{head}\n{('Parent claim: ' + parent[:600] + chr(10)) if parent else ''}{c['text']}"
            yield doc_start, c


def count_lines(path):
    with open(path, "rb") as f:
        return sum(buf.count(b"\n") for buf in iter(lambda: f.read(1 << 24), b""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="chunk files (default: all chunks/*.jsonl)")
    ap.add_argument("--collection", default="corpus")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--qdrant-path", help="use embedded local Qdrant in this folder instead of a server")
    ap.add_argument("--batch", type=int, default=256, help="chunks per upsert")
    ap.add_argument("--encode-batch", type=int, default=64, help="GPU batch; lower it on out-of-memory")
    ap.add_argument("--max-length", type=int, default=1024, help="max tokens per chunk")
    ap.add_argument("--limit", type=int, default=0, help="stop after N chunks (speed test)")
    ap.add_argument("--fake", action="store_true", help="hash vectors instead of bge-m3 (pipeline test)")
    ap.add_argument("--recreate", action="store_true", help="DELETE the collection and start over")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()

    cfg = load_config()
    chunk_dir = cfg["paths"]["chunks"]
    state = sqlite3.connect(chunk_dir / "indexed.sqlite")
    state.execute("create table if not exists files(path text primary key, next_line int, done int, "
                  "chunks int, updated real)")
    q = connect(a)

    if a.status:
        if q.collection_exists(a.collection):
            print(f"Qdrant '{a.collection}': {q.count(a.collection, exact=True).count} points")
        for row in state.execute("select path, next_line, done, chunks from files"):
            print(f"  {'done   ' if row[2] else 'partial'} {row[0]}  next_line={row[1]} chunks={row[3]}")
        return

    if a.recreate:
        state.execute("delete from files")
        state.commit()
    ensure_collection(q, a.collection, a.recreate)

    files = [pathlib.Path(f) for f in a.files] or sorted(chunk_dir.glob("*.jsonl"))
    todo = []
    for f in files:
        r = state.execute("select next_line, done from files where path=?", (str(f.resolve()),)).fetchone()
        if r and r[1] and not a.files:
            continue
        todo.append((f, r[0] if r else 0))
    if not todo:
        print("Nothing new to index. (Pass a file explicitly to re-index it.)")
        return

    print(f"Loading {'fake encoder' if a.fake else 'bge-m3'} ...")
    enc = Encoder(fake=a.fake, max_length=a.max_length, batch_size=a.encode_batch)

    grand, t_all = 0, time.time()
    for path, start in todo:
        key = str(path.resolve())
        total_lines = count_lines(path)
        print(f"\n{path.name}: {total_lines} chunks" + (f", resuming at line {start}" if start else ""))
        stream = chunk_stream(path, start)
        done_here, t0, n_batches = 0, time.time(), 0
        while True:
            want = a.batch if not a.limit else min(a.batch, a.limit - grand)
            if want <= 0:
                break
            batch = list(itertools.islice(stream, want))
            if not batch:
                break
            dense, sparse = enc.encode([c["embed_text"] for _, c in batch])
            points = []
            for (_, c), dv, (si, sv) in zip(batch, dense, sparse):
                vec = {"dense": dv}
                if si:
                    vec["sparse"] = m.SparseVector(indices=si, values=sv)
                points.append(m.PointStruct(id=point_id(c["doc"], c["section"]), vector=vec,
                                            payload={k: c.get(k, "") for k in PAYLOAD_FIELDS}))
            n_batches += 1
            q.upsert(a.collection, points, wait=(n_batches % 20 == 0) or bool(a.qdrant_path))
            done_here += len(batch)
            grand += len(batch)
            # resume point = start of the document the last chunk belongs to (re-upserting is harmless)
            state.execute("insert or replace into files values(?,?,?,?,?)",
                          (key, batch[-1][0], 0, start + done_here, time.time()))
            state.commit()
            if n_batches % 40 == 0:
                rate = done_here / (time.time() - t0)
                left = max(0, total_lines - start - done_here)
                print(f"  {start + done_here}/{total_lines}  {rate:,.0f} chunks/s  "
                      f"ETA {left / rate / 3600:.1f} h", flush=True)
        finished = not a.limit or grand < a.limit
        state.execute("insert or replace into files values(?,?,?,?,?)",
                      (key, total_lines if finished else batch[-1][0] if batch else start,
                       int(finished), start + done_here, time.time()))
        state.commit()
        rate = done_here / max(time.time() - t0, 1e-6)
        print(f"  indexed {done_here} chunks at {rate:,.0f} chunks/s")
        if a.limit and grand >= a.limit:
            print(f"\nSpeed test: {rate:,.1f} chunks/s. "
                  f"Full corpus estimate: {sum(count_lines(f) for f in files) / rate / 3600:.1f} hours.")
            break

    time.sleep(0 if a.qdrant_path else 5)          # let the last async upserts land
    n = q.count(a.collection, exact=True).count
    print(f"\nIndexed {grand} chunks in {(time.time() - t_all) / 60:.1f} min. Collection now has {n} points.")
    print("Check it:  python 08_search.py \"your question\"")


if __name__ == "__main__":
    sys.exit(main())
