"""Run the whole pipeline in order, in the background, with pause / continue / stop.

    python pipeline.py run                 # start (or resume) in the background; safe to log out
    python pipeline.py status              # which step is running, what's done, last log lines
    python pipeline.py logs                # follow the live log (Ctrl+C stops watching, not the run)
    python pipeline.py pause               # freeze the current step (keeps its place, frees CPU)
    python pipeline.py continue            # unfreeze  (alias: resume)
    python pipeline.py stop                # stop cleanly; `run` later picks up at the same step
    python pipeline.py skip                # abandon the current step and move on to the next
    python pipeline.py reset               # forget progress; next `run` starts from step 1
    python pipeline.py steps               # list the steps and whether each is enabled

Options for `run`:
    --from parse            start at this step (marks earlier ones as skipped for this run)
    --only parse index      run just these steps
    --skip uspto bigquery   leave these out
    --fresh                 start again from the first step, ignoring earlier progress
    --keep-going            continue to the next step when one fails
    --loop 168              after finishing, wait 168 hours and run again (weekly updates)
    --foreground            run in this terminal instead of the background

Every step is itself resumable, so stopping or a reboot never loses work: the
downloaders skip finished files, the parser skips parsed files, and the indexer
continues from its checkpoint. Linux/macOS only (pause uses SIGSTOP/SIGCONT).

Step settings live in the [pipeline] section of config.toml.
"""
import argparse
import datetime as dt
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

from common import ROOT, load_config

STATE_DIR = ROOT / "state"
STATE = STATE_DIR / "pipeline.json"
CONTROL = STATE_DIR / "control.json"
LOG = STATE_DIR / "pipeline.log"
PY = sys.executable


# ---------------------------------------------------------------- steps
def build_steps(cfg):
    """[(name, description, argv or None if disabled)] in run order."""
    p = cfg.get("pipeline", {})
    on = lambda k, default=True: p.get(k, default)
    bq = p.get("bigquery", "dry-run")
    steps = [
        ("inventory", "count the standards you have", ["01_inventory.py"] if on("inventory") else None),
        ("uspto", "download US patents (USPTO ODP)", ["02_download_uspto.py"] if on("uspto") else None),
        ("bigquery", f"non-US patents from BigQuery ({bq})",
         None if bq == "skip" else ["03_download_bigquery.py"] + (["--run"] if bq == "run" else [])),
        ("standards", "fill missing 3GPP / ETSI specs", ["04_download_standards.py"] if on("standards") else None),
        ("extract", "unzip, convert .doc" + (", OCR" if on("ocr", False) else ""),
         ["05_extract.py"] + (["--ocr"] if on("ocr", False) else []) if on("extract") else None),
        ("parse", "split documents into chunks", ["06_parse.py"] if on("parse") else None),
        ("index", "embed with bge-m3 and store in Qdrant",
         ["07_build_index.py"] + list(p.get("index_args", [])) if on("index") else None),
    ]
    q = p.get("check_question", "")
    steps.append(("check", "test search", ["08_search.py", q, "-k", "3"] + list(p.get("search_args", []))
                  if q else None))
    return steps


# ---------------------------------------------------------------- state helpers
def read_json(path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path, data):
    STATE_DIR.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:                                   # a zombie counts as dead
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split()[2] != "Z"
    except FileNotFoundError:
        return True                        # macOS: no /proc
    return True


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def log(msg):
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)


def signal_group(st, sig):
    pgid = st.get("child_pgid")
    if pgid:
        try:
            os.killpg(pgid, sig)
            return True
        except ProcessLookupError:
            pass
    return False


# ---------------------------------------------------------------- worker
def worker(opts):
    cfg = load_config()
    steps = build_steps(cfg)
    names = [s[0] for s in steps]
    st = read_json(STATE, {})
    if opts["fresh"] or not st.get("steps"):
        st = {"steps": {n: "pending" for n in names}, "cycle": st.get("cycle", 0) + 1}
    for n in names:
        st["steps"].setdefault(n, "pending")
    st.update(worker_pid=os.getpid(), status="running", started=now(), child_pgid=None,
              current=None, opts=opts)
    write_json(STATE, st)
    write_json(CONTROL, {})

    def handle_term(*_):                   # `kill <worker>` behaves like `stop`
        write_json(CONTROL, {"request": "stop"})
        signal_group(read_json(STATE, {}), signal.SIGCONT)
        signal_group(read_json(STATE, {}), signal.SIGTERM)
    signal.signal(signal.SIGTERM, handle_term)

    while True:
        selected = names
        if opts["only"]:
            selected = [n for n in names if n in opts["only"]]
        if opts["from_step"]:
            selected = selected[selected.index(opts["from_step"]):] if opts["from_step"] in selected else selected
        selected = [n for n in selected if n not in opts["skip"]]

        failed = False
        for name, desc, argv in steps:
            if name not in selected:
                continue
            if st["steps"].get(name) == "done":
                continue
            if argv is None:
                st["steps"][name] = "disabled"
                write_json(STATE, st)
                continue
            log(f"=== step {name}: {desc}")
            log("    $ python " + " ".join(argv))
            t0 = time.time()
            child = subprocess.Popen([PY, "-u", *argv], cwd=ROOT, start_new_session=True,
                                     stdout=sys.stdout, stderr=subprocess.STDOUT)
            st.update(current=name, child_pid=child.pid, child_pgid=os.getpgid(child.pid),
                      step_started=now(), status="running")
            st["steps"][name] = "running"
            write_json(STATE, st)
            rc = child.wait()
            ctl = read_json(CONTROL, {})
            mins = (time.time() - t0) / 60
            st = read_json(STATE, st)
            st.update(child_pid=None, child_pgid=None)
            if ctl.get("request") == "stop":
                st["steps"][name] = "stopped"
                st.update(status="stopped", stopped=now(), current=name)
                write_json(STATE, st)
                write_json(CONTROL, {})
                log(f"--- stopped during {name} after {mins:.1f} min. `python pipeline.py run` resumes here.")
                return
            if ctl.get("request") == "skip":
                st["steps"][name] = "skipped"
                write_json(CONTROL, {})
                log(f"--- skipped {name} on request")
            elif rc == 0:
                st["steps"][name] = "done"
                log(f"--- {name} finished in {mins:.1f} min")
            else:
                st["steps"][name] = f"failed (exit {rc})"
                log(f"--- {name} FAILED (exit {rc}) after {mins:.1f} min")
                failed = True
            write_json(STATE, st)
            if failed and not opts["keep_going"]:
                st.update(status="failed", current=name)
                write_json(STATE, st)
                log("Pipeline halted. Fix the problem, then `python pipeline.py run` retries this step.")
                return

        st.update(current=None, finished=now(), status="finished with errors" if failed else "finished")
        write_json(STATE, st)
        log(f"=== pipeline {st['status']}")
        if not opts["loop"]:
            return

        # ---- loop mode: wait, then start a new cycle (downloads/parse/index pick up only new data)
        wake = time.time() + opts["loop"] * 3600
        st.update(status="waiting", next_run=dt.datetime.fromtimestamp(wake).isoformat(timespec="seconds"))
        write_json(STATE, st)
        log(f"Next cycle at {st['next_run']}")
        while time.time() < wake:
            ctl = read_json(CONTROL, {})
            if ctl.get("request") == "stop":
                st.update(status="stopped", stopped=now())
                write_json(STATE, st)
                write_json(CONTROL, {})
                log("--- stopped while waiting")
                return
            if ctl.get("request") == "now":
                write_json(CONTROL, {})
                break
            time.sleep(10)
        st = {"steps": {n: "pending" for n in names}, "cycle": st.get("cycle", 0) + 1,
              "worker_pid": os.getpid(), "status": "running", "started": now(), "opts": opts}
        write_json(STATE, st)


# ---------------------------------------------------------------- commands
def cmd_run(a):
    st = read_json(STATE, {})
    if alive(st.get("worker_pid")) and st.get("status") in ("running", "paused", "waiting"):
        sys.exit(f"Already {st['status']} (pid {st['worker_pid']}). Use status / pause / stop.")
    cfg = load_config()
    names = [s[0] for s in build_steps(cfg)]
    for n in (a.only or []) + (a.skip or []) + ([a.from_step] if a.from_step else []):
        if n not in names:
            sys.exit(f"Unknown step {n!r}. Steps: {', '.join(names)}")
    opts = {"only": a.only or [], "skip": a.skip or [], "from_step": a.from_step,
            "fresh": a.fresh, "keep_going": a.keep_going, "loop": a.loop}
    if a.from_step or a.only:
        st = read_json(STATE, {})
        steps = st.get("steps", {})
        wanted = a.only or names[names.index(a.from_step):]
        for n in wanted:
            steps[n] = "pending"              # re-run the chosen steps even if done before
        if steps:
            st["steps"] = steps
            write_json(STATE, st)
    STATE_DIR.mkdir(exist_ok=True)
    if a.foreground:
        worker(opts)
        return
    with open(LOG, "a") as out:
        out.write(f"\n\n######## pipeline run started {now()} ########\n")
        proc = subprocess.Popen([PY, "-u", __file__, "_worker", json.dumps(opts)], cwd=ROOT,
                                stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
    time.sleep(1.5)
    if proc.poll() is not None:
        sys.exit(f"Worker exited immediately; see {LOG}")
    print(f"Pipeline running in the background (pid {proc.pid}).")
    print(f"  python pipeline.py status    # progress\n  python pipeline.py logs      # live log")


def cmd_pause(a):
    st = read_json(STATE, {})
    if st.get("status") != "running" or not alive(st.get("worker_pid")):
        sys.exit(f"Nothing running (status: {st.get('status', 'never run')}).")
    if not signal_group(st, signal.SIGSTOP):
        sys.exit("No step process to pause right now; try again in a moment.")
    st.update(status="paused", paused=now())
    write_json(STATE, st)
    print(f"Paused step '{st.get('current')}'. `python pipeline.py continue` resumes it.")
    if st.get("current") == "index":
        print("Note: a paused indexer still holds its GPU memory. Use `stop` to free the GPU.")


def cmd_continue(a):
    st = read_json(STATE, {})
    if st.get("status") == "waiting":
        write_json(CONTROL, {"request": "now"})
        print("Starting the next cycle now.")
        return
    if st.get("status") != "paused":
        if st.get("status") in ("stopped", "failed") or not alive(st.get("worker_pid")):
            sys.exit("Not paused. To pick up a stopped/failed run use: python pipeline.py run")
        sys.exit(f"Not paused (status: {st.get('status')}).")
    signal_group(st, signal.SIGCONT)
    st.update(status="running")
    st.pop("paused", None)
    write_json(STATE, st)
    print(f"Resumed step '{st.get('current')}'.")


def request(kind):
    st = read_json(STATE, {})
    if not alive(st.get("worker_pid")):
        sys.exit(f"Nothing running (status: {st.get('status', 'never run')}).")
    write_json(CONTROL, {"request": kind})
    if st.get("status") == "waiting":
        return st
    signal_group(st, signal.SIGCONT)          # a paused process can't handle SIGTERM
    signal_group(st, signal.SIGTERM)
    return st


def cmd_stop(a):
    st = request("stop")
    for _ in range(60):
        if not alive(st.get("worker_pid")):
            break
        time.sleep(1)
    else:
        print("Step didn't exit within 60 s; forcing it.")
        signal_group(read_json(STATE, {}), signal.SIGKILL)
        time.sleep(2)
    print("Stopped. `python pipeline.py run` continues from the same step.")


def cmd_skip(a):
    st = request("skip")
    print(f"Skipping step '{st.get('current')}'; the pipeline moves on to the next one.")


def cmd_status(a):
    st = read_json(STATE, {})
    if not st:
        print("Never run. Start with: python pipeline.py run")
        return
    status = st.get("status")
    if status in ("running", "paused", "waiting") and not alive(st.get("worker_pid")):
        status = "interrupted (machine restarted or process killed) - `run` resumes"
    print(f"Status: {status}   cycle {st.get('cycle', 1)}   started {st.get('started', '?')}")
    if st.get("next_run"):
        print(f"Next cycle: {st['next_run']}")
    marks = {"done": "✔", "running": "▶", "pending": "·", "disabled": "-", "skipped": "»", "stopped": "■"}
    cfg = load_config()
    for name, desc, argv in build_steps(cfg):
        s = st.get("steps", {}).get(name, "pending")
        if s == "running" and status == "paused":
            s = "paused"
        mark = marks.get(s, "✖" if s.startswith("failed") else "‖")
        extra = f"  (since {st.get('step_started')})" if name == st.get("current") and s in ("running", "paused") else ""
        print(f"  {mark} {name:10s} {s:16s} {desc}{extra}")
    if LOG.exists():
        lines = LOG.read_text(errors="replace").splitlines()[-a.lines:]
        print(f"\nLast {len(lines)} log lines ({LOG}):")
        for line in lines:
            print("   ", line[:200])


def cmd_logs(a):
    if not LOG.exists():
        sys.exit("No log yet.")
    try:
        subprocess.run(["tail", "-n", str(a.lines), "-f", str(LOG)])
    except KeyboardInterrupt:
        pass


def cmd_reset(a):
    st = read_json(STATE, {})
    if alive(st.get("worker_pid")) and st.get("status") in ("running", "paused", "waiting"):
        sys.exit("Stop the pipeline first: python pipeline.py stop")
    STATE.unlink(missing_ok=True)
    CONTROL.unlink(missing_ok=True)
    print("Progress cleared. Downloaded files, chunks and the index are untouched.")


def cmd_steps(a):
    for name, desc, argv in build_steps(load_config()):
        print(f"  {name:10s} {'python ' + ' '.join(argv) if argv else '(disabled in config.toml)'}")


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "_worker":
        worker(json.loads(sys.argv[2]))
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="start or resume")
    r.add_argument("--from", dest="from_step")
    r.add_argument("--only", nargs="+")
    r.add_argument("--skip", nargs="+")
    r.add_argument("--fresh", action="store_true")
    r.add_argument("--keep-going", action="store_true")
    r.add_argument("--loop", type=float, default=0, metavar="HOURS")
    r.add_argument("--foreground", action="store_true")
    for name in ("pause", "continue", "resume", "stop", "skip", "reset", "steps"):
        sub.add_parser(name)
    for name in ("status", "logs"):
        s = sub.add_parser(name)
        s.add_argument("-n", "--lines", type=int, default=15 if name == "status" else 50)
    a = ap.parse_args()
    {"run": cmd_run, "pause": cmd_pause, "continue": cmd_continue, "resume": cmd_continue,
     "stop": cmd_stop, "skip": cmd_skip, "status": cmd_status, "logs": cmd_logs,
     "reset": cmd_reset, "steps": cmd_steps}[a.cmd](a)


if __name__ == "__main__":
    main()
