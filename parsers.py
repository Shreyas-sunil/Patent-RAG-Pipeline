"""Format-specific parsers: every function returns a list of chunk dicts.

Chunk fields:
    doc         publication number or spec file name, e.g. US11234567B2, 38331-i60
    section     "abstract", "claim 3", "description part 12", "clause 5.3.5 RRC reconfiguration (part 1)"
    depends_on  parent claim number for dependent claims ("" otherwise)
    text        the chunk text
    kind        "patents" | "standards"
    area        "area-wireless" etc. ("" for standards)
    title, date, cpc, assignees, source   metadata for filters and display

Patents are split the patent way: one chunk per claim, the abstract as one chunk,
and the description in overlapping ~350-word windows. Standards are split on
their numbered clauses, then windowed.
"""
import json
import re

from lxml import etree

from common import identify

PATENT_WINDOW = (350, 60)        # words, overlap
STANDARD_WINDOW = (350, 60)
MAX_CLAIM_WORDS = 1200           # very long claims are windowed into parts

XML_PARSER = etree.XMLParser(recover=True, resolve_entities=False, no_network=True,
                             load_dtd=False, huge_tree=True)


# ---------------------------------------------------------------- helpers
def windows(text, size=350, overlap=60):
    words = (text or "").split()
    step = max(1, size - overlap)
    for i in range(0, len(words), step):
        piece = " ".join(words[i:i + size])
        if len(piece) > 200:
            yield piece
        if i + size >= len(words):
            break


def _t(el):
    if el is None:
        return ""
    return re.sub(r"\s+([,.;:)])", r"\1", " ".join(" ".join(el.itertext()).split()))


def norm_pub(num):
    """'EP-1234567-B1' -> 'EP1234567B1'"""
    return re.sub(r"[^A-Za-z0-9]", "", num or "")


CLAIM_CANDIDATE = re.compile(r"(?:^|(?<=[\s.;:]))(\d{1,3})\s*[.)]\s+(?=[A-Z一-鿿(])")


def split_claims(text):
    """Split a claims block into [(num, text)].

    Accepts only claim numbers that run 1, 2, 3 ... in order, so numbers inside
    a claim ('as in claim 1. The ...', '5. 10 mm') don't cause false splits.
    Works with and without line breaks between claims.
    """
    text = text or ""
    cuts, want = [], 1
    for m in CLAIM_CANDIDATE.finditer(text):
        if int(m.group(1)) == want:
            cuts.append((want, m.start(), m.end()))
            want += 1
    out = []
    for i, (num, _, body_start) in enumerate(cuts):
        end = cuts[i + 1][1] if i + 1 < len(cuts) else len(text)
        body = " ".join(text[body_start:end].split())
        if body:
            out.append((str(num), body))
    return out


DEP = re.compile(r"\b(?:of|in|to|by|with|from|according to|as claimed in|as defined in|as recited in)"
                 r"\s+(?:any (?:one )?of\s+)?claims?\s+(\d+)", re.I)


def claim_chunks(doc, claims):
    out = []
    for num, body in claims:
        dep = DEP.search(body[:300])
        words = body.split()
        if len(words) <= MAX_CLAIM_WORDS:
            out.append({"doc": doc, "section": f"claim {num}", "depends_on": dep.group(1) if dep else "",
                        "text": body})
        else:
            for j, piece in enumerate(windows(body, MAX_CLAIM_WORDS, 100)):
                out.append({"doc": doc, "section": f"claim {num} (part {j + 1})",
                            "depends_on": dep.group(1) if dep else "", "text": piece})
    return out


def uniq(chunks):
    """Make (doc, section) unique; it becomes the index id."""
    seen = {}
    for c in chunks:
        k = (c["doc"], c["section"])
        if k in seen:
            seen[k] += 1
            c["section"] = f"{c['section']} #{seen[k]}"
        else:
            seen[k] = 1
    return chunks


def with_meta(chunks, **meta):
    for c in chunks:
        for k, v in meta.items():
            c.setdefault(k, v)
    return chunks


# ---------------------------------------------------------------- patents: USPTO XML
def chunk_uspto_xml(path):
    root = etree.parse(str(path), XML_PARSER).getroot()
    doc = path.stem
    bib = root.find("us-bibliographic-data-grant")
    if bib is None:
        bib = root.find("us-bibliographic-data-application")
    title = _t(bib.find("invention-title")) if bib is not None else ""
    date = (root.findtext(".//publication-reference/document-id/date") or "").strip()
    cpc = []
    for c in root.iter("classification-cpc"):
        g = lambda t: (c.findtext(t) or "").strip()
        if g("section"):
            cpc.append(f"{g('section')}{g('class')}{g('subclass')}{g('main-group')}/{g('subgroup')}")
    assignees = [_t(a.find(".//orgname")) or _t(a.find(".//last-name"))
                 for a in root.iter("assignee")]

    chunks = []
    ab = root.find(".//abstract")
    if ab is not None and _t(ab):
        chunks.append({"doc": doc, "section": "abstract", "text": _t(ab)})
    for c in root.iter("claim"):
        ref = c.find(".//claim-ref")
        dep = re.sub(r"\D", "", ref.get("idref", "")).lstrip("0") if ref is not None else ""
        num = (c.get("num") or "").lstrip("0") or str(len(chunks))
        text = re.sub(r"^\s*\d+\s*\.\s*", "", _t(c))
        claim = claim_chunks(doc, [(num, text)])
        for x in claim:
            x["depends_on"] = dep
        chunks += claim
    desc = root.find(".//description")
    if desc is not None:
        paras = [_t(p) for p in desc.iter("p")] or [_t(desc)]
        for i, piece in enumerate(windows("\n".join(paras), *PATENT_WINDOW)):
            chunks.append({"doc": doc, "section": f"description part {i + 1}", "text": piece})
    return with_meta(uniq(chunks), title=title, date=date, cpc=cpc[:20],
                     assignees=[a for a in assignees if a][:5], source="uspto")


# ---------------------------------------------------------------- patents: BigQuery jsonl
def chunk_record(rec):
    """One patent row exported by 03_download_bigquery.py."""
    doc = norm_pub(rec["publication_number"])
    chunks = []
    if rec.get("abstract"):
        chunks.append({"doc": doc, "section": "abstract", "text": " ".join(rec["abstract"].split())})
    chunks += claim_chunks(doc, split_claims(rec.get("claims", "")))
    for i, piece in enumerate(windows(rec.get("description") or "", *PATENT_WINDOW)):
        chunks.append({"doc": doc, "section": f"description part {i + 1}", "text": piece})
    return with_meta(uniq(chunks), title=rec.get("title") or "", date=str(rec.get("publication_date") or ""),
                     cpc=(rec.get("cpc") or [])[:20], assignees=(rec.get("assignees") or [])[:5],
                     source="bigquery")


def chunk_jsonl(path):
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks += chunk_record(json.loads(line))
    return chunks


# ---------------------------------------------------------------- PDFs
def pdf_text(path):
    import pymupdf
    with pymupdf.open(path) as d:
        return "\n".join(page.get_text() for page in d)


CLAIMS_START = re.compile(r"\n\s*(what is claimed is|what we claim is|we claim|i claim|the invention claimed is|"
                          r"claims?)\s*:?\s*\n", re.I)


def chunk_patent_pdf(path):
    text, doc = pdf_text(path), path.stem.replace(".ocr", "")
    chunks = []
    starts = list(CLAIMS_START.finditer(text))
    # the real claims section is the last heading that is followed by "1."
    m = next((s for s in reversed(starts) if re.match(r"\s*1\s*[.)]", text[s.end():s.end() + 40])), None)
    body, claims = (text[:m.start()], text[m.end():]) if m else (text, "")
    abstract = re.search(r"\babstract\b[^\n]*\n(.{100,3000}?)(?:\n\s*\n|\n[A-Z][A-Z ]{5,}\n|$)", text, re.I | re.S)
    if abstract:
        chunks.append({"doc": doc, "section": "abstract", "text": " ".join(abstract.group(1).split())})
    chunks += claim_chunks(doc, split_claims(claims))
    for i, piece in enumerate(windows(body, *PATENT_WINDOW)):
        chunks.append({"doc": doc, "section": f"description part {i + 1}", "text": piece})
    return with_meta(uniq(chunks), source="pdf")


TOC_LINE = re.compile(r"^.*\.{4,}\s*\d+\s*$", re.M)
CLAUSE = re.compile(r"\n((?:\d+|[A-H])(?:\.\d+){0,5})\s+([A-Z][^\n]{2,100})\n")


def chunk_standard_text(text, doc):
    text = TOC_LINE.sub("", "\n" + text)
    parts = CLAUSE.split(text)
    chunks = []
    front = parts[0]
    for j, piece in enumerate(windows(front, *STANDARD_WINDOW)):
        chunks.append({"doc": doc, "section": f"front matter (part {j + 1})", "text": piece})
    for num, title, body in zip(parts[1::3], parts[2::3], parts[3::3]):
        label = f"clause {num} {title.strip()}"[:140]
        pieces = list(windows(body, *STANDARD_WINDOW))
        if not pieces and len(body.split()) > 15:
            pieces = [" ".join(body.split())]
        for j, piece in enumerate(pieces):
            chunks.append({"doc": doc, "section": f"{label} (part {j + 1})", "text": f"{title.strip()}. {piece}"})
    return chunks


def standard_meta(doc):
    body, num, ver = identify(doc + ".x")
    return {"title": f"{body} {num} v{ver}".strip() if body != "unknown" else doc, "source": body}


def chunk_standard_pdf(path):
    doc = path.stem.replace(".ocr", "")
    return with_meta(uniq(chunk_standard_text(pdf_text(path), doc)), **standard_meta(doc))


# ---------------------------------------------------------------- standards: Word
HEADING = re.compile(r"^(heading \d|h\d|annex heading|title)", re.I)
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def chunk_standard_docx(path):
    """3GPP/ETSI .docx: Word heading styles mark the clauses exactly. Tables are kept."""
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    d = docx.Document(str(path))
    doc, chunks = path.stem, []
    clause, buf = "front matter", []

    def flush():
        body = "\n".join(buf)
        pieces = list(windows(body, *STANDARD_WINDOW))
        if not pieces and len(body.split()) > 15:
            pieces = [" ".join(body.split())]
        for j, piece in enumerate(pieces):
            prefix = "" if clause == "front matter" else f"{clause}. "
            chunks.append({"doc": doc, "section": f"clause {clause} (part {j + 1})"
                           if clause != "front matter" else f"front matter (part {j + 1})",
                           "text": prefix + piece})

    for el in d.element.body.iterchildren():
        if el.tag == W + "p":
            p = Paragraph(el, d)
            style = (p.style.name if p.style is not None else "") or ""
            txt = p.text.strip()
            if not txt or style.lower().startswith("toc"):
                continue
            if HEADING.match(style):
                flush()
                clause, buf = re.sub(r"\s+", " ", txt)[:120], []
            else:
                buf.append(txt)
        elif el.tag == W + "tbl":
            t = Table(el, d)
            for row in t.rows:
                cells = []
                for cell in row.cells:
                    ct = " ".join(cell.text.split())
                    if ct and (not cells or cells[-1] != ct):   # merged cells repeat
                        cells.append(ct)
                if cells:
                    buf.append(" | ".join(cells))
    flush()
    return with_meta(uniq(chunks), **standard_meta(doc))


def chunk_standard_plain(path):
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() in {".html", ".htm"}:
        import lxml.html
        raw = lxml.html.fromstring(raw).text_content()
    return with_meta(uniq(chunk_standard_text(raw, path.stem)), **standard_meta(path.stem))


# ---------------------------------------------------------------- dispatcher
PATENT_EXT = {".xml", ".pdf", ".jsonl"}
STANDARD_EXT = {".pdf", ".docx", ".txt", ".html", ".htm"}


def chunk_any(path):
    kind = "standards" if "standards" in path.parts else "patents"
    area = next((x for x in path.parts if x.startswith("area-")), "")
    s = path.suffix.lower()
    if kind == "patents":
        fn = {".xml": chunk_uspto_xml, ".jsonl": chunk_jsonl, ".pdf": chunk_patent_pdf}.get(s)
    else:
        fn = {".docx": chunk_standard_docx, ".pdf": chunk_standard_pdf}.get(s, chunk_standard_plain)
    if fn is None:
        return []
    chunks = fn(path)
    for c in chunks:
        c.update(kind=kind, area=area)
        c.setdefault("depends_on", "")
    return chunks
