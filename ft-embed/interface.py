
import os, threading, traceback

from flask import Flask, jsonify, render_template, request

from vec import VecStore

app = Flask(__name__)

CONFIG = {
    "csv_dir":   r"C:\Users\vince\Documents\GitHub\CIPBD\Dallas\CSV",
    "out_dir":   r"C:\Users\vince\Documents\GitHub\CIPBD-Embedding\Outputs\Dallas",
    "model_dir": r"C:\Users\vince\Documents\GitHub\CIPBD-Embedding\ft-embed\training\cip-model\final",
}
JOB   = {"running": False, "stage": "", "log": []}
STORE = None
LOCK  = threading.Lock()


def log(line):
    with LOCK:
        JOB["log"].append(line)
        del JOB["log"][:-400]                 # keep the tail only


def run_job(fn, *a, **kw):
    def worker():
        JOB.update(running=True, stage=fn.__name__, log=[])
        try:
            fn(*a, **kw)
        except Exception as e:
            log(f"ERROR {type(e).__name__}: {e}")
            log(traceback.format_exc(limit=3))
        finally:
            JOB.update(running=False, stage="")
    threading.Thread(target=worker, daemon=True).start()


# ------------------------------------------------------------------ pages --
@app.route("/")
def index():
    return render_template("Temp.html")


@app.route("/api/status")
def status():
    d = dict(JOB)
    d["config"] = CONFIG
    d["store"] = STORE.stats() if STORE else None
    return jsonify(d)


# ------------------------------------------------------------ folder picker --
@app.route("/api/browse")
def browse():
    """List sub-folders so the page can walk the filesystem."""
    path = request.args.get("path") or os.path.expanduser("~")
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        return jsonify({"ok": False, "error": f"not a folder: {path}"}), 400
    try:
        entries = sorted(e for e in os.listdir(path)
                         if os.path.isdir(os.path.join(path, e))
                         and not e.startswith("."))
    except PermissionError:
        return jsonify({"ok": False, "error": "permission denied"}), 403
    n_csv = len([e for e in os.listdir(path) if e.lower().endswith(".csv")])
    return jsonify({"ok": True, "path": path, "parent": os.path.dirname(path),
                    "dirs": entries, "csv_count": n_csv})


@app.route("/api/config", methods=["POST"])
def set_config():
    d = request.get_json() or {}
    for k in ("csv_dir", "out_dir", "model_dir"):
        if d.get(k):
            CONFIG[k] = os.path.abspath(d[k])
    return jsonify({"ok": True, "config": CONFIG})


# -------------------------------------------------------------- pair side --
@app.route("/api/pair", methods=["POST"])
def pair():
    if JOB["running"]:
        return jsonify({"ok": False, "error": "a job is already running"}), 409
    steps = (request.get_json() or {}).get("steps", ["pull", "section"])

    def work():
        from pair import loss_pairing            # imported late: it loads models
        p = loss_pairing(csv_dir=CONFIG["csv_dir"])
        for s in steps:
            log(f"{s}: started")
            JOB["stage"] = s
            getattr(p, s)()
            log(f"{s}: done")

    run_job(work)
    return jsonify({"ok": True})


# --------------------------------------------------------------- vec side --
@app.route("/api/vec/load", methods=["POST"])
def vec_load():
    if JOB["running"]:
        return jsonify({"ok": False, "error": "a job is already running"}), 409
    do_embed = bool((request.get_json() or {}).get("embed", True))

    def work():
        global STORE
        STORE = VecStore(CONFIG["csv_dir"], CONFIG["out_dir"], CONFIG["model_dir"])
        log(f"reading {CONFIG['csv_dir']}")
        log(f"{STORE.load_rows()} rows")
        if not do_embed and STORE.load_vectors():
            log("reused vecs.npy")
        else:
            STORE.embed(progress=log)
        s = STORE.stats()
        log(f"{s['anchored']} with an id, {s['orphans']} without"
            + ("" if s["vectors"] else "  (names-only scoring)"))

    run_job(work)
    return jsonify({"ok": True})


@app.route("/api/vec/queue")
def vec_queue():
    if not STORE:
        return jsonify({"items": [], "error": "load first"})
    dept = request.args.get("dept") or ""
    todo = [r for r in STORE.orphans if not dept or r.get("department") == dept]
    out = []
    for r in todo[:200]:
        cands, margin = STORE.candidates(r, k=1)
        out.append({"i": r["_i"], "name": r["project_name"], "year": r["_year"],
                    "department": r.get("department", ""),
                    "top": cands[0]["pid"] if cands else None,
                    "score": cands[0]["score"] if cands else 0,
                    "margin": margin})
    out.sort(key=lambda d: -d["margin"])
    return jsonify({"items": out, "remaining": len(todo),
                    "departments": sorted({r.get("department", "")
                                           for r in STORE.orphans})})


@app.route("/api/vec/row/<int:i>")
def vec_row(i):
    cands, margin = STORE.candidates(STORE.rows[i])
    row = {k: v for k, v in STORE.rows[i].items() if not k.startswith("_")}
    return jsonify({"row": row, "i": i, "margin": margin, "candidates": cands})


@app.route("/api/vec/assign", methods=["POST"])
def vec_assign():
    d = request.get_json() or {}
    row = STORE.assign(int(d["i"]), d["pid"])
    return jsonify({"ok": True, "file": row["_file"], "pid": row["_pid"],
                    "orphans": len(STORE.orphans)})


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, threaded=True)
