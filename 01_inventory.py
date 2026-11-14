"""Step 1 - take stock of the standards you already have.

Walks corpus/standards/ and writes inventory.csv (one row per file) plus a summary:
totals by standards body and format, exact duplicates, multiple versions of the
same spec, and files with no extractable text (scans that need OCR).

    python 01_inventory.py                 # full run
    python 01_inventory.py --no-text-check # faster: skip opening every file
"""
import argparse
import collections
import csv
import hashlib
import os
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor

from common import identify, load_config, vkey

FORMATS = {".pdf", ".docx", ".doc", ".zip", ".html", ".htm", ".txt", ".xml", ".rtf"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def text_check(path, ext):
    """Return (pages, has_text). has_text is '' when unknown."""
    try:
        if ext == ".pdf":
            import pymupdf
            with pymupdf.open(path) as d:
                sample = "".join(d[i].get_text() for i in range(min(5, d.page_count)))
                return d.page_count, "yes" if len(sample.strip()) > 200 else "no (scan? needs OCR)"
        if ext == ".docx":
            import docx
            t = " ".join(p.text for p in docx.Document(path).paragraphs[:400])
            return "", "yes" if len(t.strip()) > 200 else "no"
        if ext == ".zip":
            with zipfile.ZipFile(path) as z:
                inner = [i.filename for i in z.infolist() if not i.is_dir()]
            return "", "archive: " + ", ".join(inner[:3])
        if ext in {".txt", ".html", ".htm", ".xml", ".rtf"}:
            with open(path, "rb") as f:
                return "", "yes" if len(f.read(4000).strip()) > 200 else "no"
        if ext == ".doc":
            return "", "convert to .docx (05_extract.py)"
    except Exception as e:
        return "", f"error: {type(e).__name__}"
    return "", ""


def row(args):
    path, check = args
    ext = os.path.splitext(path)[1].lower()
    body, num, ver = identify(os.path.basename(path))
    pages, has_text = text_check(path, ext) if check else ("", "")
    return {"path": path, "body": body, "number": num, "version": ver, "format": ext,
            "bytes": os.path.getsize(path), "pages": pages, "has_text": has_text, "sha256": sha256(path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-text-check", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_config()
    root = cfg["paths"]["corpus"] / "standards"
    root.mkdir(parents=True, exist_ok=True)
    out = a.out or cfg.get("standards", {}).get("inventory", "inventory.csv")

    files = [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs
             if os.path.splitext(f)[1].lower() in FORMATS and not f.endswith(".part")]
    print(f"Scanning {len(files)} files under {root} ...")
    rows = []
    with ProcessPoolExecutor() as ex:
        for i, r in enumerate(ex.map(row, [(f, not a.no_text_check) for f in files], chunksize=32), 1):
            rows.append(r)
            if i % 2000 == 0:
                print(f"  {i}/{len(files)}")

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["path"])
        w.writeheader()
        w.writerows(rows)

    # ---- summary
    print(f"\nWrote {out}: {len(rows)} files, {sum(r['bytes'] for r in rows) / 1e9:.2f} GB")
    print("\nBy body:  ", dict(collections.Counter(r["body"] for r in rows).most_common()))
    print("By format:", dict(collections.Counter(r["format"] for r in rows).most_common()))

    by_hash = collections.defaultdict(list)
    for r in rows:
        by_hash[r["sha256"]].append(r["path"])
    dups = [p for p in by_hash.values() if len(p) > 1]
    print(f"\nExact duplicates: {len(dups)} groups, {sum(len(p) - 1 for p in dups)} removable files")
    for p in dups[:10]:
        print("   ", " == ".join(p))

    by_spec = collections.defaultdict(list)
    for r in rows:
        if r["number"]:
            by_spec[(r["body"], r["number"])].append(r)
    multi = {k: v for k, v in by_spec.items() if len({x["version"] for x in v}) > 1}
    print(f"\nSpecs with several versions: {len(multi)} (keep the latest unless you need history)")
    for (body, num), v in list(multi.items())[:10]:
        vs = sorted({x["version"] for x in v}, key=vkey)
        print(f"    {body} {num}: {', '.join(vs)}")

    scans = [r for r in rows if r["has_text"].startswith("no")]
    print(f"\nNo extractable text: {len(scans)} files (OCR with 05_extract.py --ocr)")
    unknown = sum(1 for r in rows if r["body"] == "unknown")
    print(f"Unrecognised file names: {unknown} (fine; they still get parsed)")


if __name__ == "__main__":
    main()
