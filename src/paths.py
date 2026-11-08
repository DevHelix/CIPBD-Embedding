"""Folder picker — self-contained Flask blueprint.

Nothing here touches the host application. Register the blueprint, then ask
this module for paths whenever you need them.

    from paths import bp as paths_bp
    import paths

    app.register_blueprint(paths_bp)

    paths.get("csv_dir")    -> 'C:\\...\\Dallas\\CSV'  or None
    paths.csv_files()       -> ['C:\\...\\2010.csv', ...]   sorted, absolute
    paths.ready()           -> True once a csv_dir has been chosen
    paths.all()             -> {'csv_dir': ..., 'weight_dir': ..., 'out_dir': ...}

Routes are namespaced under /api/paths/ so they cannot collide with yours.
State lives in one JSON file and is reloaded on import, so a restart keeps
whatever the user picked.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import threading

from flask import Blueprint, jsonify, request

__all__ = ["bp", "get", "all", "csv_files", "ready", "configure", "KEYS"]

# --------------------------------------------------------------------- config

#: The paths this picker manages. Add a key here and a matching
#: `.row.pick[data-key="..."]` in the HTML; nothing else needs to change.
KEYS = ("csv_dir", "weight_dir", "out_dir")

_HERE = pathlib.Path(__file__).resolve().parent
_store = _HERE.parent / "data" / "config.json"
_lock = threading.Lock()
_paths: dict[str, str] = {}

YEAR_CSV = re.compile(r"^\d{4}\.csv$")

bp = Blueprint("paths", __name__, url_prefix="/api/paths")

DEFAULTS: dict[str, str | None] = {
    "csv_dir":    None,                                        # must be picked
    "weight_dir": str(_HERE.parent / "data" / "weights"),
    "out_dir":    str(_HERE.parent / "data" / "outputs"),
}

MAKE_DEFAULT_DIRS = True        # create them on import

def configure(store: str | os.PathLike | None = None) -> None:
    """Point the JSON store somewhere other than ../data/dirs.json."""
    global _store
    if store is not None:
        _store = pathlib.Path(store).resolve()
    _load()


# ---------------------------------------------------------------- persistence

def _load() -> None:
    """Defaults first, then anything the user saved on top."""
    global _paths
    try:
        raw = json.loads(_store.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}

    merged = {k: v for k, v in DEFAULTS.items() if k in KEYS and v}
    merged.update({k: str(v) for k, v in raw.items() if k in KEYS and v})

    if MAKE_DEFAULT_DIRS:
        for k, v in DEFAULTS.items():
            if v and merged.get(k) == v:
                os.makedirs(v, exist_ok=True)

    with _lock:
        _paths = merged

def _save() -> None:
    _store.parent.mkdir(parents=True, exist_ok=True)
    tmp = _store.with_suffix(_store.suffix + ".tmp")
    with _lock:
        snapshot = dict(_paths)
    tmp.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    os.replace(tmp, _store)          # atomic: never leaves a half-written file


_load()


# ------------------------------------------------------------------ public API

def get(key: str) -> str | None:
    """The chosen path for `key`, or None if the user has not picked one."""
    if key not in KEYS:
        raise KeyError(f"unknown path key {key!r}; expected one of {KEYS}")
    with _lock:
        return _paths.get(key)


def all() -> dict[str, str]:                       # noqa: A001 - deliberate name
    """Every chosen path. A copy, so callers cannot mutate the store."""
    with _lock:
        return dict(_paths)


def ready(key: str = "csv_dir") -> bool:
    """True when `key` has been chosen and still exists on disk."""
    p = get(key)
    return bool(p) and os.path.isdir(p)


def csv_files(key: str = "csv_dir", year_only: bool = True) -> list[str]:
    """Absolute paths of the CSVs in the chosen folder, sorted by name.

    year_only=True keeps just `NNNN.csv`, which skips the `.bak` copies and
    any stray exports. Returns [] when nothing has been chosen yet, so callers
    can treat "not picked" and "empty folder" the same way if they want to.
    """
    d = get(key)
    if not d or not os.path.isdir(d):
        return []
    names = sorted(
        e for e in os.listdir(d)
        if (YEAR_CSV.match(e) if year_only else e.lower().endswith(".csv"))
    )
    return [os.path.join(d, e) for e in names]


# ---------------------------------------------------------------------- routes

@bp.get("/browse")
def browse():
    """List sub-folders so the page can walk the filesystem."""
    path = os.path.abspath(request.args.get("path") or os.path.expanduser("~"))
    if not os.path.isdir(path):
        return jsonify({"ok": False, "error": f"not a folder: {path}"}), 400
    try:
        names = os.listdir(path)
    except PermissionError:
        return jsonify({"ok": False, "error": "permission denied"}), 403
    except OSError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    dirs = sorted(
        e for e in names
        if not e.startswith(".") and os.path.isdir(os.path.join(path, e))
    )
    return jsonify({
        "ok": True,
        "path": path,
        "parent": os.path.dirname(path) or path,
        "dirs": dirs,
        "csv_count": sum(1 for e in names if e.lower().endswith(".csv")),
    })


@bp.get("")
def read():
    return jsonify({"ok": True, "paths": all(), "keys": list(KEYS),
                    "ready": ready()})


@bp.post("")
def write():
    d = request.get_json(silent=True) or {}
    bad = []
    with _lock:
        for k in KEYS:
            v = d.get(k)
            if not v:
                continue
            p = os.path.abspath(str(v))
            if not os.path.isdir(p):
                bad.append(p)
                continue
            _paths[k] = p
    _save()
    if bad:
        return jsonify({"ok": False, "error": f"not a folder: {bad[0]}",
                        "paths": all()}), 400
    return jsonify({"ok": True, "paths": all(), "ready": ready()})
