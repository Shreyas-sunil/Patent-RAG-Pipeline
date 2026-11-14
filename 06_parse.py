"""Step 4 - parse every document into patent-aware / clause-aware chunks.

Reads corpus/ (patent XML, BigQuery .jsonl, PDFs, standards .docx/.pdf/.txt/.html),
writes chunks/<batch>.jsonl (one chunk per line, all chunks of a document together)
and chunks/<batch>.report.txt. Uses every CPU core. Files already parsed in an
earlier batch are skipped, so re-run it whenever new downloads arrive.

    python 06_parse.py --test corpus/patents/area-wireless/US11234567B2.xml   # print chunks of one file
    python 06_parse.py --sample 20        # parse 20 random files, print a summary (no output written)
    python 06_parse.py                    # full run -> chunks/batch-YYYYMMDD-HHMM.jsonl
    python 06_parse.py --batch wireless1 --only corpus/patents/area-wireless
"""
import argparse
import collections
import datetime as dt
import json
import pathlib
import random
from concurrent.futures import ProcessPoolExecutor

from common import load_config
from parsers import PATENT_EXT, STANDARD_EXT, chunk_any


def work(path):
    try:
        return path, chunk_any(pathlib.Path(path)), None
    except Exception as e:
        return path, [], f"{type(e).__name__}: {e}"


def find_files(roots):
    out = []
    for root in roots:
        for p in pathlib.Path(root).rglob("*"):
            if not p.is_file() or p.name.startswith("."):
                continue
            s = p.suffix.lower()
            if "standards" in p.parts:
                if s in STANDARD_EXT:
                    out.append(p)
            elif s in PATENT_EXT:
                out.append(p)
    return sorted(set(out))


class Stats:
    """Running parse statistics (keeps counts only, so memory stays flat on huge runs)."""

    def __init__(self):
        self.st = collections.Counter()

    def add(self, path, chunks, err):
        st = self.st
        kind = "standards" if "standards" in pathlib.Path(path).parts else "patents"
        st[f"{kind}_files"] += 1
        if err:
            st[f"{kind}_errors"] += 1
            return
        if not chunks:
            st[f"{kind}_empty"] += 1
            return
        docs = collections.defaultdict(set)
        for c in chunks:
            docs[c["doc"]].add(c["section"].split(" ")[0])
        for secs in docs.values():
            st[f"{kind}_docs"] += 1
            if kind == "patents":
                st["patents_with_claims"] += "claim" in secs
                st["patents_with_abstract"] += "abstract" in secs
        st[f"{kind}_chunks"] += len(chunks)

    def report(self):
        st, lines = self.st, []
        for kind in ("patents", "standards"):
            if st[f"{kind}_files"]:
                lines.append(f"{kind}: files={st[kind + '_files']} docs={st[kind + '_docs']} "
                             f"chunks={st[kind + '_chunks']} empty={st[kind + '_empty']} "
                             f"errors={st[kind + '_errors']}")
        if st["patents_docs"]:
            lines.append(f"patents with claims:   {100 * st['patents_with_claims'] / st['patents_docs']:.1f}%"
                         "  (goal: 95%+)")
            lines.append(f"patents with abstract: {100 * st['patents_with_abstract'] / st['patents_docs']:.1f}%")
        return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=f"batch-{dt.datetime.now():%Y%m%d-%H%M}")
    ap.add_argument("--only", nargs="*", help="parse only these folders/files")
    ap.add_argument("--test", metavar="FILE", help="print the chunks of one file and exit")
    ap.add_argument("--sample", type=int, default=0, help="parse N random files, report, write nothing")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()

    if a.test:
        _, chunks, err = work(a.test)
        if err:
            raise SystemExit(err)
        for c in chunks:
            dep = f"  (depends on claim {c['depends_on']})" if c.get("depends_on") else ""
            print(f"--- {c['doc']} | {c['section']}{dep} | {len(c['text'].split())} words")
            print(c["text"][:400] + ("..." if len(c["text"]) > 400 else ""))
        print(f"\n{len(chunks)} chunks; title={chunks[0].get('title') if chunks else ''!r}")
        return

    cfg = load_config()
    out_dir = cfg["paths"]["chunks"]
    roots = a.only or [cfg["paths"]["corpus"]]
    files = [str(p) for p in find_files(roots)]

    if a.sample:
        pick = random.sample(files, min(a.sample, len(files)))
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(work, pick))
        for path, chunks, err in res:
            secs = collections.Counter(c["section"].split(" ")[0] for c in chunks)
            print(f"{'ERROR ' + err if err else dict(secs)}  <- {path}")
        stats = Stats()
        for r in res:
            stats.add(*r)
        print("\n" + stats.report())
        return

    done_file = out_dir / "done.txt"
    done = set(done_file.read_text().split("\n")) if done_file.exists() else set()
    files = [f for f in files if f not in done]
    print(f"{len(files)} new files to parse -> {out_dir / (a.batch + '.jsonl')}")
    if not files:
        return

    stats, n_chunks = Stats(), 0
    with ProcessPoolExecutor(a.workers) as ex, \
            open(out_dir / f"{a.batch}.jsonl", "a", encoding="utf-8") as out, \
            open(out_dir / "errors.log", "a") as err_log, open(done_file, "a") as dn:
        for n, (path, chunks, e) in enumerate(ex.map(work, files, chunksize=8), 1):
            if e:
                err_log.write(f"{path}\t{e}\n")
            else:
                for c in chunks:
                    out.write(json.dumps(c, ensure_ascii=False) + "\n")
                n_chunks += len(chunks)
                dn.write(path + "\n")
            stats.add(path, chunks, e)
            if n % 1000 == 0:
                print(f"  {n}/{len(files)} files, {n_chunks} chunks")

    report = stats.report()
    (out_dir / f"{a.batch}.report.txt").write_text(report + "\n")
    print("\n" + report)
    print(f"\nWrote {n_chunks} chunks to {out_dir / (a.batch + '.jsonl')}; errors in {out_dir / 'errors.log'}")


if __name__ == "__main__":
    main()
