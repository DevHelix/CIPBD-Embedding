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
    vectors.assign(i, pid)              -> write one id (marked 'manual')
    vectors.auto(0.7, dry_run=True)     -> preview / write every id scoring >= 0.7
    vectors.cluster_fit()               -> cluster into an editable draft
    vectors.cluster_edit("move", rows=[3, 9], to=7)   -> edit the draft
    vectors.cluster(dry_run=True)       -> preview / write the draft's cluster ids

Every write goes through writeback.py: the source CSV keeps all its rows and
columns, and changed rows are flagged in writeback.MARK_COL.

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

import clusters
import paths
from embed import VecStore

__all__ = ["bp", "configure", "ready", "store", "stats", "load", "assign", "auto", "cluster",
           "cluster_fit", "cluster_view", "cluster_members", "cluster_find",
           "cluster_edit", "draft"]

bp = Blueprint("vectors", __name__, url_prefix="/api/vec")

_store: VecStore | None = None
_draft: clusters.Draft | None = None    # cluster labels being inspected/edited
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
    global _store, _draft
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
        _draft = None

    st = s.stats()
    _log(f"{st['anchored']} with an id, {st['orphans']} without"
         + ("" if st.get("vectors") else "  (names-only scoring)"))
    return st


def assign(i: int, pid: str) -> dict:
    """Write one id to row `i` of the loaded store. Thread-safe."""
    if _store is None:
        raise RuntimeError("load the store first")
    with _lock:                       # one writer at a time
        row = _store.assign(i, pid)
    _log(f"{row['_file']}: row {row['_line']} -> {row['_pid']}")
    return row


def auto(threshold: float = 0.7, dept: str | None = None,
         dry_run: bool = False) -> dict:
    """Write every orphan's top id whose score >= threshold. Thread-safe.

    Returns VecStore.auto_assign()'s report plus 'orphans' (left afterwards).
    """
    if _store is None:
        raise RuntimeError("load the store first")
    with _lock:
        res = _store.auto_assign(threshold, dept or None, dry_run=dry_run)
    if not dry_run:
        _log(f"auto >= {threshold:.2f}: wrote {res['written']} ids "
             f"across {len(res['files'])} file(s)")
        for s in res["skipped"]:
            _log(f"skipped {s['file']} row {s['row']}: {s['reason']}", "yield")
    return {**res, "orphans": len(_store.orphans)}


def draft() -> clusters.Draft | None:
    """The cluster draft being edited, or None before cluster_fit()."""
    return _draft


def _need_draft() -> clusters.Draft:
    if _store is None:
        raise RuntimeError("load the store first")
    if _draft is None or len(_draft.labels) != len(_store.rows):
        raise RuntimeError("run clustering first")
    return _draft


def cluster_fit() -> dict:
    """Cluster every row's vector into a fresh draft. Returns cluster_view()."""
    global _draft
    if _store is None:
        raise RuntimeError("load the store first")
    if _store.vecs is None:
        raise RuntimeError("no vectors: create or load an embedding first")
    if len(_store.vecs) != len(_store.rows):
        raise RuntimeError(f"{len(_store.vecs)} vectors for {len(_store.rows)} rows; "
                           "re-embed so they line up")
    d = clusters.Draft.fit(_store.vecs)
    with _lock:
        _draft = d
    _log(f"clustered {len(d.labels)} rows into {d.new_label()} clusters")
    return cluster_view()


def cluster_view(overwrite: bool = False) -> dict:
    """Cluster list + what a write would do. See clusters.overview()."""
    d = _need_draft()
    return {**clusters.overview(_store.rows, d.labels, overwrite=overwrite),
            "can_undo": d.can_undo, "can_redo": d.can_redo,
            "undo_label": d.undo_label, "redo_label": d.redo_label}


def cluster_members(label: int, overwrite: bool = False) -> dict:
    """Members of one draft cluster (-1 = noise). See clusters.detail()."""
    d = _need_draft()
    return clusters.detail(_store.rows, d.labels, label, _store.vecs,
                           overwrite=overwrite)


def cluster_find(query: str, limit: int = 50) -> list[dict]:
    """Rows whose name contains every word of `query`, with their draft cluster."""
    return clusters.find(_store.rows, _need_draft().labels, query, limit)


def cluster_edit(op: str, *, rows=None, to=None, sources=None, target=None,
                 label=None, threshold=None, dry_run: bool = False) -> dict:
    """Edit the draft. Returns {'label': affected cluster} or {'undone': bool}.

    op = "move"      rows, to (cluster number, -1 noise, None new cluster)
         "merge"     sources, target (None = largest source)
         "dissolve"  label
         "prune"     threshold, label (None = every cluster), dry_run
                     -> {'removed': n, 'clusters': clusters touched}
         "undo" / "redo"  -> {'undone' / 'redone': what it was, or None}
    """
    d = _need_draft()
    with _lock:
        if op == "move":
            return {"label": d.move(rows or [], to)}
        if op == "merge":
            return {"label": d.merge(sources or [], target)}
        if op == "dissolve":
            return {"label": d.dissolve(label)}
        if op == "prune":
            if threshold is None:
                raise ValueError("missing threshold")
            if _store.vecs is None:
                raise RuntimeError("no vectors to measure fit against")
            before = d.labels.copy()
            rows = d.prune(_store.vecs, float(threshold),
                           None if label is None else int(label), dry_run=dry_run)
            return {"removed": int(len(rows)),
                    "clusters": int(len(set(before[rows].tolist()))),
                    "dry_run": bool(dry_run)}
        if op == "undo":
            return {"undone": d.undo()}
        if op == "redo":
            return {"redone": d.redo()}
    raise ValueError(f"unknown edit {op!r}")


def cluster(dry_run: bool = False, overwrite: bool = False,
            recluster: bool = False) -> dict:
    """Write the draft's cluster ids into id_modified (fits one if there is none).

    The ids written are exactly the ones shown in the draft (no renumbering).
    Returns clusters.run()'s report without the label list.
    """
    if _store is None:
        raise RuntimeError("load the store first")
    if recluster or _draft is None or len(_draft.labels) != len(_store.rows):
        cluster_fit()
    with _lock:
        res = clusters.run(_store, labels=_draft.labels, relabel=False,
                           overwrite=overwrite, dry_run=dry_run)
        res.pop("labels")
    if not dry_run:
        _log(f"clusters: {res['clusters']} clusters, {res['noise']} noise; "
             f"wrote {res['written']} rows across {len(res['files'])} file(s)")
        for sk in res["skipped"]:
            _log(f"skipped {sk['file']} row {sk['row']}: {sk['reason']}", "yield")
    return res


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

    try:
        row = assign(i, pid)
    except (RuntimeError, OSError) as e:
        return jsonify({"ok": False, "error": str(e)}), 409
    return jsonify({"ok": True, "file": row["_file"], "pid": row["_pid"],
                    "orphans": len(s.orphans)})


@bp.post("/auto")
def api_auto():
    """Body: {threshold, dept?, dry_run?}. Dry runs return a sample to preview."""
    s, err = _need()
    if err:
        return err
    d = request.get_json(silent=True) or {}
    try:
        t = float(d.get("threshold"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "missing threshold"}), 400
    if not 0 < t <= 1:
        return jsonify({"ok": False, "error": "threshold must be in (0, 1]"}), 400
    try:
        res = auto(t, d.get("dept") or None, dry_run=bool(d.get("dry_run")))
    except OSError as e:
        return jsonify({"ok": False, "error": str(e)}), 409
    props = res.pop("proposals")
    return jsonify({"ok": True, **res, "sample": props[:15]})


@bp.post("/weights")
def api_weights():
    s, err = _need()
    if err:
        return err
    w = s.set_weights(request.get_json(silent=True) or {})
    return jsonify({"ok": True, "weights": w})


def _cluster_call(fn, *a, **kw):
    """Run a cluster_* function and turn its errors into JSON responses."""
    s, err = _need()
    if err:
        return err
    try:
        return jsonify({"ok": True, **fn(*a, **kw)})
    except ImportError:
        return jsonify({"ok": False, "error": "evoc is not installed (pip install evoc)"}), 500
    except (RuntimeError, OSError, ValueError, TypeError) as e:
        return jsonify({"ok": False, "error": str(e)}), 409


def _flag(name) -> bool:
    return request.args.get(name, "").lower() in ("1", "true", "yes")


@bp.post("/cluster/fit")
def api_cluster_fit():
    return _cluster_call(cluster_fit)


@bp.get("/cluster")
def api_cluster_view():
    return _cluster_call(cluster_view, overwrite=_flag("overwrite"))


@bp.get("/cluster/<int(signed=True):label>")
def api_cluster_members(label):
    return _cluster_call(cluster_members, label, overwrite=_flag("overwrite"))


@bp.get("/cluster/find")
def api_cluster_find():
    q = request.args.get("q", "")
    return _cluster_call(lambda: {"rows": cluster_find(q)})


@bp.post("/cluster/edit")
def api_cluster_edit():
    """Body: {op: move|merge|dissolve|prune|undo|redo, rows?, to?, sources?, target?,
    label?, threshold?, dry_run?}."""
    d = request.get_json(silent=True) or {}
    return _cluster_call(cluster_edit, str(d.get("op", "")), rows=d.get("rows"),
                         to=d.get("to"), sources=d.get("sources"),
                         target=d.get("target"), label=d.get("label"),
                         threshold=d.get("threshold"), dry_run=bool(d.get("dry_run")))


@bp.post("/cluster")
def api_cluster():
    """Write the draft. Body: {dry_run?, overwrite?, recluster?}."""
    d = request.get_json(silent=True) or {}
    return _cluster_call(cluster, dry_run=bool(d.get("dry_run")),
                         overwrite=bool(d.get("overwrite")),
                         recluster=bool(d.get("recluster")))
