"""Step 2a - download US patents from the USPTO Open Data Portal and keep your areas.

Lists the weekly full-text XML files (newest first), downloads each one, splits it
into individual patents, keeps the patents whose CPC codes match an area in
config.toml, and saves each as corpus/patents/area-<name>/US<number><kind>.xml.
Stops when every area reaches its target. Safe to stop and restart at any time.

Needs a free ODP API key:   export USPTO_API_KEY=xxxxxxxx

    python 02_download_uspto.py              # run (use tmux; it takes hours)
    python 02_download_uspto.py --status     # progress so far
    python 02_download_uspto.py --list       # show which weekly files would be fetched
    python 02_download_uspto.py --max-files 2   # quick test with two weekly files
"""
import argparse
import datetime as dt
import io
import os
import pathlib
import sqlite3
import sys
import time
import zipfile

from lxml import etree

from common import AreaMatcher, Downloader, load_config

API = "https://api.uspto.gov/api/v1/datasets/products"
PARSER = etree.XMLParser(recover=True, resolve_entities=False, no_network=True,
                         load_dtd=False, huge_tree=True)


# ---------------------------------------------------------------- listing
def list_files(dl, product, from_date, to_date):
    """All weekly archive files of a product between two dates, newest first."""
    files, offset, limit = [], 0, 100
    while True:
        r = dl.get(f"{API}/{product}", params={
            "fileDataFromDate": from_date, "fileDataToDate": to_date,
            "includeFiles": "true", "offset": offset, "limit": limit})
        if r.status_code in (401, 403):
            sys.exit(f"USPTO API refused the request ({r.status_code}). Check USPTO_API_KEY.")
        r.raise_for_status()
        data = r.json()
        prods = data.get("bulkDataProductBag") or [data]
        prod = next((p for p in prods if p.get("productIdentifier", product) == product), prods[0])
        bag = (prod.get("productFileBag") or {}).get("fileDataBag") or []
        for f in bag:
            name = f.get("fileName", "")
            if not name.lower().endswith(".zip"):
                continue                                  # skip DTDs, docs, readmes
            url = f.get("fileDownloadURI") or f.get("fileDownloadUrl") or f"{API}/files/{product}/{name}"
            files.append({"product": product, "name": name, "url": url,
                          "date": f.get("fileDataToDate") or f.get("fileDataFromDate") or "",
                          "size": f.get("fileSize") or 0})
        total = (prod.get("productFileBag") or {}).get("count") or prod.get("productFileTotalQuantity")
        offset += limit
        if len(bag) < limit or (total and offset >= int(total)):
            break
    uniq = {f["name"]: f for f in files}
    return sorted(uniq.values(), key=lambda f: (f["date"], f["name"]), reverse=True)


# ---------------------------------------------------------------- splitting
def iter_docs(zip_path):
    """Yield the raw XML of each patent inside a weekly zip (one file = many concatenated XML docs)."""
    with zipfile.ZipFile(zip_path) as z:
        for member in z.infolist():
            if not member.filename.lower().endswith(".xml"):
                continue
            buf = []
            with z.open(member) as raw:
                for line in io.TextIOWrapper(raw, encoding="utf-8", errors="replace"):
                    if line.startswith("<?xml") and buf:
                        yield "".join(buf)
                        buf = []
                    buf.append(line)
            if buf:
                yield "".join(buf)


def cpc_codes(root):
    out = []
    for c in root.iter("classification-cpc"):
        g = lambda t: (c.findtext(t) or "").strip()
        if g("section"):
            out.append(f"{g('section')}{g('class')}{g('subclass')}{g('main-group')}/{g('subgroup')}")
    return out


def pub_id(root):
    d = root.find("./us-bibliographic-data-grant/publication-reference/document-id")
    if d is None:
        d = root.find("./us-bibliographic-data-application/publication-reference/document-id")
    if d is None:
        d = root.find(".//publication-reference/document-id")
    if d is None:
        return None
    num = (d.findtext("doc-number") or "").strip()
    kind = (d.findtext("kind") or "").strip()
    if num.isdigit():
        num = str(int(num))
    return f"{(d.findtext('country') or 'US').strip()}{num}{kind}" if num else None


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--max-files", type=int, default=0, help="stop after N weekly files (testing)")
    a = ap.parse_args()

    cfg = load_config()
    u = cfg.get("uspto", {})
    paths = cfg["paths"]
    matcher = AreaMatcher(cfg["areas"])
    targets = matcher.targets()

    mpath = paths["manifests"] / "uspto.sqlite"
    key = os.environ.get("USPTO_API_KEY", "")
    dl = Downloader(mpath, rate_per_sec=u.get("rate_per_sec", 0.5),
                    headers={"X-API-KEY": key} if key else None)
    db = dl.db
    db.execute("create table if not exists weekly(name text primary key, status text, kept int, scanned int, at real)")
    db.execute("create table if not exists patents(id text primary key, area text, weekly text, path text)")
    db.commit()

    def counts():
        return dict(db.execute("select area, count(*) from patents group by area").fetchall())

    if a.status:
        dl.print_status("weekly zips:")
        w = dict(db.execute("select status,count(*) from weekly group by status").fetchall())
        print("weekly files processed:", w)
        c = counts()
        for name, t in targets.items():
            print(f"  area-{name}: {c.get(name, 0)} / {t}")
        return

    if not key:
        sys.exit("Set USPTO_API_KEY first (free key from data.uspto.gov -> My ODP).")

    to_date = u.get("to_date") or dt.date.today().isoformat()
    files = []
    for product in u.get("products", ["PTGRXML"]):
        got = list_files(dl, product, u.get("from_date", "2015-01-01"), to_date)
        print(f"{product}: {len(got)} weekly files between {u.get('from_date')} and {to_date}")
        files += got
    files.sort(key=lambda f: (f["date"], f["name"]), reverse=True)

    if a.list:
        for f in files:
            print(f"{f['date']}  {f['product']:8s} {f['name']:20s} {int(f['size'] or 0) / 1e6:8.0f} MB")
        return

    n_files = 0
    for f in files:
        c = counts()
        open_areas = {n for n, t in targets.items() if c.get(n, 0) < t}
        if not open_areas:
            print("All areas reached their targets.")
            break
        if db.execute("select 1 from weekly where name=? and status='done'", (f["name"],)).fetchone():
            continue

        zpath = paths["downloads"] / f["product"] / f["name"]
        fid = f"{f['product']}/{f['name']}"
        dl.add(fid, f["url"], zpath)
        if not zpath.exists():
            print(f"Downloading {f['name']} ({int(f['size'] or 0) / 1e6:.0f} MB) ...")
            if dl.run_one(fid, f["url"], zpath) is None:
                continue

        t0, kept, scanned = time.time(), 0, 0
        try:
            for xml in iter_docs(zpath):
                scanned += 1
                try:
                    root = etree.fromstring(xml.encode("utf-8"), PARSER)
                except etree.XMLSyntaxError:
                    continue
                if root is None or root.tag not in ("us-patent-grant", "us-patent-application"):
                    continue
                area = matcher.match(cpc_codes(root))
                if area not in open_areas:
                    continue
                pid = pub_id(root)
                if not pid or db.execute("select 1 from patents where id=?", (pid,)).fetchone():
                    continue
                out = paths["corpus"] / "patents" / f"area-{area}" / f"{pid}.xml"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(xml, encoding="utf-8")
                db.execute("insert into patents values(?,?,?,?)", (pid, area, f["name"], str(out)))
                kept += 1
                c[area] = c.get(area, 0) + 1
                if c[area] >= targets[area]:
                    open_areas.discard(area)
                    if not open_areas:
                        break
        except zipfile.BadZipFile:
            print(f"  {f['name']} is corrupt; deleting so it re-downloads next run")
            zpath.unlink(missing_ok=True)
            db.execute("update items set status='failed', attempts=attempts+1 where id=?", (fid,))
            db.commit()
            continue
        db.execute("insert or replace into weekly values(?,?,?,?,?)",
                   (f["name"], "done", kept, scanned, time.time()))
        db.commit()
        if not u.get("keep_zips", False):
            zpath.unlink(missing_ok=True)
        c = counts()
        print(f"  {f['name']}: kept {kept} of {scanned} in {time.time() - t0:.0f}s | " +
              ", ".join(f"{n}={c.get(n, 0)}/{t}" for n, t in targets.items()))
        n_files += 1
        if a.max_files and n_files >= a.max_files:
            break


if __name__ == "__main__":
    main()
