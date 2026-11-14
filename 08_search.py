"""Step 6 - check the index: run a question and print the best passages.

    python 08_search.py "battery thermal runaway detection"
    python 08_search.py "RRC reconfiguration procedure" --kind standards
    python 08_search.py "beam failure recovery" --area area-wireless --rerank -k 5
    python 08_search.py "claim 1" --doc US11234567B2
"""
import argparse
import textwrap

from retrieve import Retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--kind", choices=["patents", "standards"])
    ap.add_argument("--area", help="e.g. area-wireless")
    ap.add_argument("--doc", help="limit to one document, e.g. US11234567B2")
    ap.add_argument("--rerank", action="store_true", help="use bge-reranker-v2-m3 (~2 GB download)")
    ap.add_argument("--collection", default="corpus")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--qdrant-path")
    ap.add_argument("--fake", action="store_true", help="match an index built with --fake")
    a = ap.parse_args()

    r = Retriever(a.collection, a.qdrant_url, a.qdrant_path, rerank=a.rerank, fake=a.fake)
    hits = r.search(a.question, k=a.k, kind=a.kind, area=a.area, doc=a.doc)
    if not hits:
        print("No results. Is the index built and is Qdrant running?")
    for i, h in enumerate(hits, 1):
        print(f"\n[{i}] {h['doc']} — {h['section']}   score={h['score']:.3f}  {h.get('kind')} {h.get('area')}")
        if h.get("title"):
            print(f"    {h['title']}")
        print(textwrap.indent(textwrap.fill(h["text"][:500] + ("..." if len(h["text"]) > 500 else ""), 100), "    "))


if __name__ == "__main__":
    main()
