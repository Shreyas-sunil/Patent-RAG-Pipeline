"""Step 3 - turn raw downloads into files the parser can read.

  1. Unzips every .zip under corpus/ (3GPP specs ship as zips; nested zips too),
     keeping only document files (.docx .doc .pdf .txt .html .xml).
  2. Converts old Word .doc (and .rtf) files to .docx with LibreOffice.
  3. Optional: adds a text layer to scanned PDFs with ocrmypdf (--ocr).

Safe to re-run; finished work is skipped.

    python 05_extract.py                 # unzip + convert
    python 05_extract.py --ocr           # also OCR PDFs that have no text
    python 05_extract.py --delete-zips   # remove archives after unpacking
"""
import argparse
import os
import pathlib
import shutil
import subprocess
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor

from common import load_config

KEEP = {".docx", ".doc", ".pdf", ".txt", ".html", ".htm", ".xml", ".rtf", ".zip"}


def safe_extract(zpath):
    """Extract document files next to the zip, into <zipname>/ ; returns count."""
    zpath = pathlib.Path(zpath)
    out = zpath.with_suffix("")
    marker = out / ".extracted"
    if marker.exists():
        return 0
    n = 0
    try:
        with zipfile.ZipFile(zpath) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                name = pathlib.PurePosixPath(info.filename)
                if name.suffix.lower() not in KEEP or ".." in name.parts or name.is_absolute():
                    continue
                target = out / name.name          # flatten; spec zips hold 1-3 files
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                n += 1
    except zipfile.BadZipFile:
        print(f"  bad zip: {zpath}")
        return 0
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok")
    return n


def unzip_all(root, delete):
    total = 0
    for _ in range(3):                             # handles zips inside zips
        zips = [p for p in root.rglob("*.zip") if not (p.with_suffix("") / ".extracted").exists()]
        # USPTO weekly zips live in downloads/, not corpus/, so they're never touched here
        if not zips:
            break
        with ProcessPoolExecutor() as ex:
            total += sum(ex.map(safe_extract, zips, chunksize=16))
        if delete:
            for z in zips:
                z.unlink(missing_ok=True)
    print(f"Unzipped {total} document files")


def convert_doc(root):
    olds = [p for p in root.rglob("*") if p.suffix.lower() in {".doc", ".rtf"}
            and not p.with_suffix(".docx").exists()]
    if not olds:
        print("No .doc/.rtf files need converting")
        return
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        print(f"{len(olds)} .doc/.rtf files need LibreOffice: sudo apt install -y libreoffice-writer")
        return
    print(f"Converting {len(olds)} .doc/.rtf files to .docx ...")
    by_dir = {}
    for p in olds:
        by_dir.setdefault(p.parent, []).append(p)
    for d, files in by_dir.items():
        for i in range(0, len(files), 200):
            batch = files[i:i + 200]
            with tempfile.TemporaryDirectory() as prof:   # private profile: allows parallel runs
                subprocess.run([soffice, f"-env:UserInstallation=file://{prof}", "--headless",
                                "--convert-to", "docx", "--outdir", str(d), *map(str, batch)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3600)
    left = sum(1 for p in olds if not p.with_suffix(".docx").exists())
    print(f"Converted {len(olds) - left}; failed {left}")


def needs_ocr(path):
    import pymupdf
    try:
        with pymupdf.open(path) as d:
            text = "".join(d[i].get_text() for i in range(min(5, d.page_count)))
        return len(text.strip()) < 200
    except Exception:
        return False


def ocr(root):
    if not shutil.which("ocrmypdf"):
        print("Install OCR first: sudo apt install -y ocrmypdf tesseract-ocr")
        return
    pdfs = [p for p in root.rglob("*.pdf") if not p.name.endswith(".ocr.pdf")
            and not p.with_suffix(".ocr.pdf").exists()]
    with ProcessPoolExecutor() as ex:
        scans = [p for p, s in zip(pdfs, ex.map(needs_ocr, pdfs, chunksize=32)) if s]
    print(f"OCR: {len(scans)} of {len(pdfs)} PDFs have no text layer")
    jobs = max(1, (os.cpu_count() or 2) // 2)
    for i, p in enumerate(scans, 1):
        out = p.with_suffix(".ocr.pdf")
        r = subprocess.run(["ocrmypdf", "--skip-text", "--jobs", str(jobs), "--quiet", str(p), str(out)])
        if r.returncode == 0:
            p.rename(p.with_suffix(".pdf.scan"))      # keep original, hide it from the parser
        print(f"  {i}/{len(scans)} {p.name}: {'ok' if r.returncode == 0 else 'failed'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--delete-zips", action="store_true")
    a = ap.parse_args()
    root = load_config()["paths"]["corpus"]
    unzip_all(root, a.delete_zips)
    convert_doc(root)
    if a.ocr:
        ocr(root)


if __name__ == "__main__":
    main()
