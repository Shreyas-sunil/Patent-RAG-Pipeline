"""Step 2c - fill gaps in your standards collection (3GPP and ETSI, both free).

3GPP: walks 3gpp.org/ftp/Specs/latest/<release>/<series>_series/ for the releases
and series in config.toml, compares with inventory.csv (from 01_inventory.py) and
downloads only specs you don't already have at that version or newer.
ETSI: fetches the latest published PDF of each document listed in config.toml.

Files land in corpus/standards/3gpp/... and corpus/standards/etsi/...
(3GPP zips are unpacked by 05_extract.py).

    python 04_download_standards.py --dry-run   # list what is missing
    python 04_download_standards.py             # download it
    python 04_download_standards.py --status
"""
import argparse
import csv
import pathlib
import re
from html.parser import HTMLParser
from urllib.parse import urljoin

from common import Downloader, identify, load_config, vkey

BASE_3GPP = "https://www.3gpp.org/ftp/Specs/latest/"
BASE_ETSI = "https://www.etsi.org/deliver/"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)


def links(dl, url):
    r = dl.get(url)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    p = Links()
    p.feed(r.text)
    return [urljoin(url, h) for h in p.links]


def have_versions(inventory_csv):
    """{(body, number): best version tuple} from inventory.csv."""
    best = {}
    if not pathlib.Path(inventory_csv).exists():
        print(f"(no {inventory_csv}; run 01_inventory.py first to avoid re-downloading what you have)")
        return best
    with open(inventory_csv) as f:
        for r in csv.DictReader(f):
            if r.get("number"):
                k = (r["body"], r["number"])
                best[k] = max(best.get(k, (0,)), vkey(r["version"]))
    return best


def plan_3gpp(dl, cfg, have):
    s = cfg["standards"].get("3gpp", {})
    only = set(s.get("only", []))
    out_root = cfg["paths"]["corpus"] / "standards" / "3gpp"
    todo = []
    for rel in s.get("releases", []):
        for series in s.get("series", []):
            url = f"{BASE_3GPP}{rel}/{series}_series/"
            zips = [u for u in links(dl, url) if u.lower().endswith(".zip")]
            print(f"{rel} {series}_series: {len(zips)} specs listed")
            for u in zips:
                name = u.rsplit("/", 1)[-1]
                body, num, ver = identify(name)
                if only and num not in only:
                    continue
                if have.get((body, num), (0,)) >= vkey(ver):
                    continue
                have[(body, num)] = vkey(ver)    # newest release listed first wins
                todo.append((f"3gpp/{name}", u, out_root / rel / f"{series}_series" / name))
    return todo


def etsi_dir(doc):
    """'TS 103 645' -> 'etsi_ts/103600_103699/103645/'; 'ES 202 706-1' -> .../20270601/"""
    m = re.match(r"\s*([A-Z]{2})\s*(\d{3})\s*(\d{3})(?:-(\d+))?", doc.upper())
    if not m:
        raise ValueError(f"can't read ETSI number {doc!r}")
    kind, num = m.group(1).lower(), m.group(2) + m.group(3)
    lo = int(num) // 100 * 100
    folder = num + (f"{int(m.group(4)):02d}" if m.group(4) else "")
    return f"etsi_{kind}/{lo:06d}_{lo + 99:06d}/{folder}/"


def plan_etsi(dl, cfg):
    out_root = cfg["paths"]["corpus"] / "standards" / "etsi"
    todo = []
    for doc in cfg["standards"].get("etsi", {}).get("documents", []):
        base = BASE_ETSI + etsi_dir(doc)
        versions = sorted((u for u in links(dl, base) if re.search(r"/\d{2}\.\d{2}\.\d{2}_\d{2}/$", u)),
                          key=lambda u: re.findall(r"(\d{2})\.(\d{2})\.(\d{2})", u)[-1])
        if not versions:
            print(f"ETSI {doc}: not found at {base}")
            continue
        pdfs = [u for u in links(dl, versions[-1]) if u.lower().endswith(".pdf")]
        for u in pdfs:
            name = u.rsplit("/", 1)[-1]
            todo.append((f"etsi/{name}", u, out_root / name))
        print(f"ETSI {doc}: latest {versions[-1].rstrip('/').rsplit('/', 1)[-1]} ({len(pdfs)} PDF)")
    return todo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    s = cfg.get("standards", {})
    dl = Downloader(cfg["paths"]["manifests"] / "standards.sqlite", rate_per_sec=s.get("rate_per_sec", 1.0))
    if a.status:
        dl.print_status("standards:")
        return

    have = have_versions(s.get("inventory", "inventory.csv"))
    todo = plan_3gpp(dl, cfg, have) + plan_etsi(dl, cfg)
    todo = [t for t in todo if not dl.is_done(t[0]) and not pathlib.Path(t[2]).exists()]
    print(f"\n{len(todo)} documents to download")
    if a.dry_run:
        for id, url, dest in todo:
            print("  ", url)
        return
    for id, url, dest in todo:
        dl.add(id, url, dest)
    dl.run()
    dl.print_status("standards:")


if __name__ == "__main__":
    main()
