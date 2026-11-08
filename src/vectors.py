"""Embedding store and id-review queue — self-contained Flask blueprint.

Wraps vec.VecStore the way paths.py wraps the folder picker: the host app
registers a blueprint and injects two callables, and never touches the store
directly unless it wants to.

    import vectors

    app.register_blueprint(vectors.bp)
    vectors.configure(log=log, runner=run_job)      # both optional

    vectors.ready()      -> True once rows are loaded
    vectors.store()      -> the live VecStore, or None
    vectors.stats()      -> {'rows': .., 'anchored': .., 'orphans': .., 'vectors': ..}

Routes live under /api/vec/ so they cannot collide with yours.

Folders come from `paths`, so this module reads whatever the picker last
stored and nothing has to pass paths around:

    csv_dir     the books to read and write back to
    out_dir     where vecs.npy is cached
    weight_dir  the fine-tuned model, if there is one

`runner` is injected rather than imported so this module never imports the
app (which would be circular). Give it your run_job; the default runs the
work synchronously, which is fine for a script but will hang a request.
"""

from __future__ import annotations

import os
import threading

from flask import Blueprint, jsonify, request

import paths
from embed import VecStore

__all__ = ["bp", "configure", "ready", "store", "stats", "load"]

bp = Blueprint("vectors", __name__, url_prefix="/api/vec")

_store: VecStore | None = None
_lock = threading.Lock()

#: Replaced by configure(). Defaults keep the module usable from a REPL.
_log = lambda msg, level="write": print(f"[{level}] {msg}")
_run = lambda fn, *a, **kw: fn(*a, **kw)        # synchronous fallback

def _usable_model(d):
    """A folder is a model only if it has the files transformers needs."""
    if not d or not os.path.isdir(d):
        return None
    if os.path.isfile(os.path.join(d, "config.json")) or \
       os.path.isfile(os.path.join(d, "modules.json")):
        return d
    # a training run leaves the usable copy in cip-model/final
    nested = os.path.join(d, "cip-model", "final")
    if os.path.isfile(os.path.join(nested, "config.json")):
        return nested
    return None

def configure(log=None, runner=None) -> None:
    """Inject the host app's logger and background-job runner."""
    global _log, _run
    if log is not None:
        _log = log
    if runner is not None:
        _run = runner


# ------------------------------------------------------------------ public API

def store() -> VecStore | None:
    return _store


def ready() -> bool:
    return _store is not None and bool(_store.rows)


def stats() -> dict:
    return _store.stats() if _store else {"rows": 0, "anchored": 0,
                                          "orphans": 0, "vectors": 0}


def load(embed: bool = True) -> dict:
    """Build the store from the currently chosen folders. Blocking.

    Call this through `runner` (the routes do) — embedding 22k names takes
    a while and must not happen inside a request handler.
    """
    global _store
    csv_dir = paths.get("csv_dir")
    out_dir = paths.get("out_dir")    
    
    model = _usable_model(paths.get("weight_dir")) \
         or _usable_model(paths.get("out_dir"))
    s = VecStore(csv_dir, out_dir, model)
    if model:
        _log(f"model: {model}")
    else:
        _log("no fine-tuned model found — scoring on names only", "yield")

    s = VecStore(csv_dir, out_dir, model if model and os.path.isdir(model) else None)
    _log(f"reading {csv_dir}")
    n = s.load_rows()
    _log(f"{n} rows")
    if n == 0:
        _log(f"no NNNN.csv rows found in {csv_dir}", "error")

    if not embed and s.load_vectors():
        _log("reused vecs.npy")
    else:
        s.embed(progress=lambda m: _log(m))

    with _lock:
        _store = s

    st = s.stats()
    _log(f"{st['anchored']} with an id, {st['orphans']} without"
         + ("" if st.get("vectors") else "  (names-only scoring)"))
    return st


# ------------------------------------------------------------------- guard --

def _need():
    """(store, None) when loaded, else (None, flask response)."""
    if _store is None:
        return None, (jsonify({"ok": False, "error": "load the store first"}), 409)
    return _store, None


# ---------------------------------------------------------------------- routes

@bp.get("")
def api_status():
    return jsonify({
        "ok": True,
        "loaded": ready(),
        "stats": stats(),
        "weights": _store.weights if _store else None,
        "paths": paths.all(),
    })


@bp.post("/load")
def api_load():
    if not paths.ready("csv_dir"):
        return jsonify({"ok": False, "error": "choose a CSV folder first"}), 400
    do_embed = bool((request.get_json(silent=True) or {}).get("embed", True))
    _run(load, do_embed)
    return jsonify({"ok": True, "embedding": do_embed})


@bp.get("/queue")
def api_queue():
    s, err = _need()
    if err:
        return err
    dept  = request.args.get("dept") or ""
    limit = max(1, min(int(request.args.get("limit", 200)), 1000))

    todo = [r for r in s.orphans if not dept or r.get("department") == dept]
    items = []
    for r in todo[:limit]:
        cands, margin = s.candidates(r, k=1)
        items.append({
            "i": r["_i"], "name": r["project_name"], "year": r["_year"],
            "department": r.get("department", ""),
            "top":    cands[0]["pid"]   if cands else None,
            "score":  cands[0]["score"] if cands else 0,
            "margin": margin,
        })
    items.sort(key=lambda d: -d["margin"])
    return jsonify({
        "ok": True, "items": items, "remaining": len(todo),
        "departments": sorted({r.get("department", "") for r in s.orphans}),
    })


@bp.get("/row/<int:i>")
def api_row(i):
    s, err = _need()
    if err:
        return err
    if not 0 <= i < len(s.rows):
        return jsonify({"ok": False, "error": f"no row {i}"}), 404
    cands, margin = s.candidates(s.rows[i])
    row = {k: v for k, v in s.rows[i].items() if not k.startswith("_")}
    return jsonify({"ok": True, "row": row, "i": i,
                    "margin": margin, "candidates": cands})


@bp.post("/assign")
def api_assign():
    s, err = _need()
    if err:
        return err
    d = request.get_json(silent=True) or {}
    try:
        i = int(d["i"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"ok": False, "error": "missing row index"}), 400
    pid = str(d.get("pid") or "").strip()
    if not pid:
        return jsonify({"ok": False, "error": "missing project_id"}), 400
    if not 0 <= i < len(s.rows):
        return jsonify({"ok": False, "error": f"no row {i}"}), 404

    with _lock:                       # one writer at a time: assign rewrites a CSV
        row = s.assign(i, pid)
    _log(f"{row['_file']}: row {row['_line']} -> {pid}")
    return jsonify({"ok": True, "file": row["_file"], "pid": row["_pid"],
                    "orphans": len(s.orphans)})


@bp.post("/weights")
def api_weights():
    s, err = _need()
    if err:
        return err
    w = s.set_weights(request.get_json(silent=True) or {})
    return jsonify({"ok": True, "weights": w})
