"""Shared helpers: config loading, CPC matching, and a resumable, polite downloader.

Every download script uses Downloader, which keeps an SQLite manifest so a crash,
reboot or network drop never re-downloads finished files.
"""
import hashlib
import os
import pathlib
import re
import sqlite3
import sys
import time

try:
    import tomllib                      # Python 3.11+
except ModuleNotFoundError:             # Python 3.10
    import tomli as tomllib

import requests

ROOT = pathlib.Path(__file__).resolve().parent


def load_config(path=None):
    path = pathlib.Path(path or os.environ.get("PATENT_RAG_CONFIG", ROOT / "config.toml"))
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    p = cfg.setdefault("paths", {})
    for key, default in (("corpus", "corpus"), ("manifests", "manifests"),
                         ("chunks", "chunks"), ("downloads", "downloads")):
        d = pathlib.Path(p.get(key, default))
        p[key] = d if d.is_absolute() else ROOT / d
        p[key].mkdir(parents=True, exist_ok=True)
    return cfg


# ---------------------------------------------------------------- CPC matching
def norm_cpc(code):
    """'H04W 72/04' -> 'H04W72/04'"""
    return re.sub(r"\s+", "", code or "").upper()


class AreaMatcher:
    """Assigns a patent to the first area whose CPC prefix matches any of its codes."""

    def __init__(self, areas):
        self.areas = [(a["name"], [norm_cpc(c) for c in a["cpc"]], int(a.get("target", 10**9)))
                      for a in areas]

    def match(self, codes):
        codes = [norm_cpc(c) for c in codes]
        for name, prefixes, _ in self.areas:
            if any(c.startswith(p) for c in codes for p in prefixes):
                return name
        return None

    def targets(self):
        return {name: t for name, _, t in self.areas}


# ---------------------------------------------------------------- downloader
class Downloader:
    """Manifest-backed downloader.

    - status per item: pending / done / failed
    - writes to <dest>.part, renames only after the SHA-256 is recorded
    - resumes partial files with HTTP Range when the server supports it
    - fixed request rate, exponential back-off, honours HTTP 429 Retry-After
    """

    def __init__(self, manifest, rate_per_sec=1.0, headers=None, max_attempts=6):
        pathlib.Path(manifest).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(manifest)
        self.db.execute("""create table if not exists items(
            id text primary key, url text, dest text, status text default 'pending',
            bytes int default 0, sha256 text, attempts int default 0, error text,
            updated real)""")
        self.db.commit()
        self.gap, self.last = 1.0 / max(rate_per_sec, 1e-6), 0.0
        self.max_attempts = max_attempts
        self.http = requests.Session()
        self.http.headers["User-Agent"] = "patent-rag/1.0 (research corpus builder)"
        if headers:
            self.http.headers.update(headers)

    # -- manifest
    def add(self, id, url, dest):
        self.db.execute("insert or ignore into items(id,url,dest,updated) values(?,?,?,?)",
                        (id, url, str(dest), time.time()))
        self.db.commit()

    def is_done(self, id):
        r = self.db.execute("select status from items where id=?", (id,)).fetchone()
        return bool(r and r[0] == "done")

    def pending(self):
        return self.db.execute("select id,url,dest,attempts from items where status!='done' "
                               "and attempts<? order by rowid", (self.max_attempts,)).fetchall()

    def status(self):
        rows = dict(self.db.execute("select status,count(*) from items group by status").fetchall())
        size = self.db.execute("select coalesce(sum(bytes),0) from items where status='done'").fetchone()[0]
        return rows, size

    def print_status(self, label=""):
        rows, size = self.status()
        print(f"{label} done={rows.get('done', 0)} pending={rows.get('pending', 0)} "
              f"failed={rows.get('failed', 0)} downloaded={size / 1e9:.2f} GB")

    # -- fetching
    def _wait(self):
        wait = self.gap - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        self.last = time.time()

    def get(self, url, **kw):
        """Rate-limited GET with retries; used for listings and API calls."""
        for attempt in range(self.max_attempts):
            self._wait()
            try:
                r = self.http.get(url, timeout=kw.pop("timeout", 120), **kw)
                if r.status_code == 429 or r.status_code >= 500:
                    delay = int(r.headers.get("Retry-After", 0) or 0) or min(300, 5 * 2 ** attempt)
                    print(f"  HTTP {r.status_code} on {url}; waiting {delay}s", file=sys.stderr)
                    time.sleep(delay)
                    continue
                return r
            except requests.RequestException as e:
                print(f"  {type(e).__name__} on {url}; retrying", file=sys.stderr)
                time.sleep(min(300, 5 * 2 ** attempt))
        raise IOError(f"gave up on {url}")

    def fetch(self, id, url, dest):
        """Download one item now. Returns the final path, or raises."""
        dest = pathlib.Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        headers = {}
        have = part.stat().st_size if part.exists() else 0
        if have:
            headers["Range"] = f"bytes={have}-"
        self._wait()
        r = self.http.get(url, stream=True, timeout=300, headers=headers, allow_redirects=True)
        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", 60) or 60))
            raise IOError("rate limited (429)")
        if r.status_code == 416:            # range not satisfiable: start over
            part.unlink(missing_ok=True)
            raise IOError("bad range; restarting")
        r.raise_for_status()
        mode = "ab" if (have and r.status_code == 206) else "wb"
        h = hashlib.sha256()
        if mode == "ab":
            with open(part, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
        with open(part, mode) as f:
            for block in r.iter_content(1 << 20):
                f.write(block)
                h.update(block)
        size = part.stat().st_size
        part.rename(dest)
        self.db.execute("update items set status='done', sha256=?, bytes=?, error=null, updated=? "
                        "where id=?", (h.hexdigest(), size, time.time(), id))
        self.db.commit()
        return dest

    def run_one(self, id, url, dest, attempts=0):
        try:
            return self.fetch(id, url, dest)
        except Exception as e:
            self.db.execute("update items set status='failed', attempts=attempts+1, error=?, updated=? "
                            "where id=?", (f"{type(e).__name__}: {e}"[:300], time.time(), id))
            self.db.commit()
            print(f"  failed {id}: {e}", file=sys.stderr)
            time.sleep(min(300, 2 ** (attempts + 1)))
            return None

    def run(self):
        for id, url, dest, attempts in self.pending():
            self.run_one(id, url, dest, attempts)


# ---------------------------------------------------------------- standards file names
def v3gpp(code):
    """3GPP version code 'i60' -> '18.6.0'  (0-9 then a=10 ... z=35)."""
    conv = lambda ch: int(ch, 36)
    if len(code) == 3:
        return ".".join(str(conv(c)) for c in code)
    if len(code) == 6:                       # newer 2-digit form, e.g. 181000
        return f"{int(code[:2])}.{int(code[2:4])}.{int(code[4:])}"
    return code


def identify(name):
    """Best-effort (body, number, version) from a file name."""
    n = name.lower()
    m = re.match(r"(\d{2})(\d{3})(?:-(\d{2}))?-([0-9a-z]{3,6})\b", n)          # 38331-i60.zip
    if m:
        num = f"{m.group(1)}.{m.group(2)}" + (f"-{m.group(3)}" if m.group(3) else "")
        return "3GPP", num, v3gpp(m.group(4))
    m = re.match(r"(ts|tr|en|es|eg|gs|gr|sr)_?(\d{6,9})v(\d{2})(\d{2})(\d{2})", n)   # ts_138331v180600p.pdf
    if m:
        d = m.group(2)
        return "ETSI", f"{m.group(1).upper()} {d[:3]} {d[3:6]}{(' ' + d[6:]) if len(d) > 6 else ''}", \
            f"{int(m.group(3))}.{int(m.group(4))}.{int(m.group(5))}"
    m = re.search(r"ieee[\s_-]*(std[\s_-]*)?([\d.]+[a-z]?)[\s_-]*(\d{4})?", n)
    if m:
        return "IEEE", m.group(2), m.group(3) or ""
    m = re.search(r"t-rec-([a-z])\.(\d+(?:\.\d+)?)-(\d{6})", n)
    if m:
        return "ITU", f"{m.group(1).upper()}.{m.group(2)}", m.group(3)
    m = re.search(r"\brfc[\s_-]?(\d{3,5})", n)
    if m:
        return "IETF", f"RFC {m.group(1)}", ""
    m = re.search(r"iso[\s_/-]*(iec)?[\s_-]*(\d{3,5}(?:-\d+)*)[\s_:-]*(\d{4})?", n)
    if m:
        return "ISO/IEC" if m.group(1) else "ISO", m.group(2), m.group(3) or ""
    return "unknown", "", ""


def vkey(v):
    return tuple(int(x) if x.isdigit() else 0 for x in re.split(r"[.-]", v or "0"))
