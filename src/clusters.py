"""Cluster rows by their embeddings and write a cluster id into MARK_COL.

This is separate from the cosine-similarity review in embed.py: nothing here
scores candidates or touches project_id. Rows whose name vectors land in the
same cluster get the same id (e.g. "cl0007") in the `id_modified` column, across
every year's CSV. Use it standalone or through vectors.cluster_*().

One call, no editing:

    import clusters
    from embed import VecStore

    s = VecStore(csv_dir, out_dir, model_dir); s.load_rows(); s.load_vectors()
    report = clusters.run(s, dry_run=True)       # preview, writes nothing
    report = clusters.run(s, labels=report["labels"], relabel=False)  # write it

Inspect and edit first (this is what the Clusters tab does):

    d = clusters.Draft.fit(s.vecs)               # labels numbered by size
    clusters.overview(s.rows, d.labels)          # one dict per cluster
    clusters.detail(s.rows, d.labels, 3, s.vecs) # members of cl0003
    d.move([12, 40], to=7)                       # rows 12, 40 -> cl0007
    d.move([5], to=None)                         # row 5 -> a new cluster
    d.merge([3, 9], target=3)                    # cl0009 folded into cl0003
    d.dissolve(11)                               # cl0011 -> noise
    d.prune(s.vecs, 0.6, label=3)                # cl0003 members with fit < 0.6 -> noise
    d.prune(s.vecs, 0.6)                         # same, in every cluster
    d.undo(); d.redo()
    clusters.run(s, labels=d.labels, relabel=False)

Rules for the id_modified column
    * after fitting, cl0000 is the largest cluster; edits keep numbers stable
      (a new cluster gets the next free number, merged ones leave a gap)
    * numbers are NOT stable between fits: re-clustering renumbers
    * writing replaces old cluster ids; a row that is now noise has its old
      cluster id cleared
    * a row marked by the review tool ("manual", "auto 0.842") is left alone
      unless overwrite=True
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

import writeback as wb

__all__ = ["PREFIX", "NOISE", "is_cluster_id", "fit", "relabel", "cluster_ids",
           "row_action", "plan", "summarize", "overview", "detail", "find",
           "fits", "Draft", "write", "run"]

PREFIX = "cl"
NOISE = -1


# -------------------------------------------------------------- labelling --

def is_cluster_id(value, prefix: str = PREFIX) -> bool:
    """True for a marker this module wrote, e.g. 'cl0007'."""
    return bool(re.fullmatch(rf"{re.escape(prefix)}\d+", str(value or "").strip()))


def fit(vecs, clusterer=None, **kw) -> np.ndarray:
    """Cluster `vecs` (n x d). Returns one int label per row, -1 for noise.

    `clusterer` is anything with fit_predict() (EVoC, sklearn HDBSCAN, ...).
    Without one, EVoC(**kw) is used.
    """
    if clusterer is None:
        from evoc import EVoC
        clusterer = EVoC(**kw)
    return np.asarray(clusterer.fit_predict(np.asarray(vecs)), dtype=int)


def relabel(labels) -> np.ndarray:
    """Renumber clusters by size (largest -> 0); ties by first row. -1 stays."""
    labels = np.asarray(labels, dtype=int)
    size = Counter(int(l) for l in labels if l >= 0)
    first = {}
    for i, l in enumerate(labels):
        first.setdefault(int(l), i)
    order = sorted(size, key=lambda l: (-size[l], first[l]))
    new = {old: n for n, old in enumerate(order)}
    return np.array([new.get(int(l), NOISE) for l in labels], dtype=int)


def _width(labels) -> int:
    labels = np.asarray(labels, dtype=int)
    return max(4, len(str(max(int(labels.max(initial=0)), 0))))


def label_id(label: int, width: int = 4, prefix: str = PREFIX) -> str | None:
    """'cl0007' for 7, None for noise."""
    return f"{prefix}{int(label):0{width}d}" if label >= 0 else None


def cluster_ids(labels, prefix: str = PREFIX, relabel_first: bool = True) -> list[str | None]:
    """Turn labels into marker strings ('cl0007'); noise becomes None.

    relabel_first=True renumbers by size first (use for fresh fit() output);
    pass False for edited Draft labels so the numbers you saw are the ones
    written. Zero-padded to at least 4 digits.
    """
    labels = relabel(labels) if relabel_first else np.asarray(labels, dtype=int)
    w = _width(labels)
    return [label_id(l, w, prefix) for l in labels]


# ------------------------------------------------------------- planning --

def row_action(current_mark, cid, *, overwrite: bool = False,
               prefix: str = PREFIX) -> str:
    """What writing `cid` would do to a row whose mark is `current_mark`.

    One of 'set', 'cleared', 'unchanged', 'protected'.
    """
    cur = str(current_mark or "").strip()
    if cid is None:                                  # noise
        return "cleared" if is_cluster_id(cur, prefix) else "unchanged"
    if cur and not is_cluster_id(cur, prefix) and not overwrite:
        return "protected"                           # manual / auto mark: keep
    return "unchanged" if cur == cid else "set"


def plan(rows, ids, *, overwrite: bool = False, prefix: str = PREFIX,
         mark_col: str = wb.MARK_COL) -> tuple[list[wb.Change], dict]:
    """Work out which rows need a new marker. Writes nothing.

    rows  VecStore.rows (need _file, _line, project_name, current mark_col)
    ids   cluster_ids() output, same order and length as rows
    Returns (changes, counts) with counts = {'set', 'cleared', 'unchanged',
    'protected'}.
    """
    if len(rows) != len(ids):
        raise ValueError(f"{len(ids)} cluster ids for {len(rows)} rows")
    changes = []
    counts = Counter(set=0, cleared=0, unchanged=0, protected=0)
    for row, cid in zip(rows, ids):
        act = row_action(row.get(mark_col), cid, overwrite=overwrite, prefix=prefix)
        counts[act] += 1
        if act in ("set", "cleared"):
            changes.append(wb.Change(row["_file"], row["_line"], None, cid or "",
                                     row.get("project_name")))
    return changes, dict(counts)


# ------------------------------------------------------------- inspecting --

def fits(labels, vecs) -> np.ndarray:
    """Each row's cosine to the mean vector of its own cluster; NaN for noise.

    This is the 'fit' column in the Clusters tab: low values are the rows most
    likely misplaced. Computed for every cluster in one pass.
    """
    labels = np.asarray(labels, dtype=int)
    vecs = np.asarray(vecs, dtype=float)
    out = np.full(len(labels), np.nan)
    for l in np.unique(labels[labels >= 0]):
        idx = np.flatnonzero(labels == l)
        c = vecs[idx].mean(axis=0)
        c /= (np.linalg.norm(c) or 1.0)
        out[idx] = vecs[idx] @ c
    return out


def summarize(rows, ids, top: int = 10, per: int = 6) -> dict:
    """Cluster count, noise count, and the `top` largest clusters with members."""
    groups: dict[str, list] = {}
    for row, cid in zip(rows, ids):
        if cid:
            groups.setdefault(cid, []).append(row)
    biggest = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:top]
    return {
        "clusters": len(groups),
        "noise": sum(1 for c in ids if c is None),
        "rows": len(ids),
        "top": [{"id": cid, "size": len(m),
                 "members": [{"year": r.get("_year"), "name": r.get("project_name"),
                              "department": r.get("department", "")}
                             for r in sorted(m, key=lambda r: r.get("_year", 0))[:per]]}
                for cid, m in biggest],
    }


def overview(rows, labels, *, overwrite: bool = False, prefix: str = PREFIX,
             mark_col: str = wb.MARK_COL) -> dict:
    """Everything the cluster list needs, for edited labels (not renumbered).

    Returns {'clusters': [{'label', 'id', 'size', 'name', 'years': [lo, hi],
             'dup_years', 'departments', 'changes'}], 'noise': {...},
             'counts': {...}, 'rows'}. 'name' is the most common name in the
    cluster; 'dup_years' counts years holding more than one member;
    'changes' is how many rows a write would alter.
    """
    labels = np.asarray(labels, dtype=int)
    w = _width(labels)
    groups: dict[int, list[int]] = {}
    for i, l in enumerate(labels):
        groups.setdefault(int(l), []).append(i)

    counts = Counter(set=0, cleared=0, unchanged=0, protected=0)
    out = []
    for l, idx in groups.items():
        cid = label_id(l, w, prefix)
        acts = Counter(row_action(rows[i].get(mark_col), cid, overwrite=overwrite,
                                  prefix=prefix) for i in idx)
        counts.update(acts)
        if l < 0:
            continue
        years = Counter(rows[i].get("_year") for i in idx)
        depts = Counter(rows[i].get("department", "") for i in idx)
        names = Counter(rows[i].get("project_name", "") for i in idx)
        out.append({"label": l, "id": cid, "size": len(idx),
                    "name": names.most_common(1)[0][0],
                    "years": [min(years), max(years)],
                    "dup_years": sum(1 for n in years.values() if n > 1),
                    "departments": [d for d, _ in depts.most_common(3)],
                    "changes": acts["set"] + acts["cleared"]})
    out.sort(key=lambda c: c["label"])
    noise = groups.get(NOISE, [])
    return {"clusters": out, "rows": len(labels), "counts": dict(counts),
            "noise": {"size": len(noise)}}


def detail(rows, labels, label: int, vecs=None, *, overwrite: bool = False,
           prefix: str = PREFIX, mark_col: str = wb.MARK_COL) -> dict:
    """Members of one cluster (label -1 = noise), worst fit first when vecs given.

    Each member: {'i', 'name', 'year', 'department', 'total', 'project_id',
    'mark', 'action', 'fit', 'dup_year'}. 'fit' is cosine to the cluster's
    mean vector, so low values are the rows most likely misplaced.
    """
    labels = np.asarray(labels, dtype=int)
    idx = np.flatnonzero(labels == label)
    cid = label_id(label, _width(labels), prefix)
    fit = None
    if vecs is not None and len(idx) and label >= 0:
        v = np.asarray(vecs, dtype=float)[idx]
        c = v.mean(axis=0)
        c /= (np.linalg.norm(c) or 1.0)
        fit = v @ c
    years = Counter(rows[i].get("_year") for i in idx)
    members = []
    for n, i in enumerate(idx):
        r = rows[i]
        members.append({
            "i": int(i), "name": r.get("project_name", ""), "year": r.get("_year"),
            "department": r.get("department", ""), "total": r.get("_total"),
            "project_id": r.get("_pid", ""), "mark": r.get(mark_col, "") or "",
            "action": row_action(r.get(mark_col), cid, overwrite=overwrite, prefix=prefix),
            "fit": round(float(fit[n]), 4) if fit is not None else None,
            "dup_year": years[r.get("_year")] > 1,
        })
    if fit is not None:
        members.sort(key=lambda m: m["fit"])
    else:
        members.sort(key=lambda m: (m["year"] or 0, m["name"]))
    return {"label": int(label), "id": cid, "size": len(members), "members": members}


def find(rows, labels, query: str, limit: int = 50, prefix: str = PREFIX) -> list[dict]:
    """Rows whose name contains every word of `query` (case-insensitive)."""
    words = query.lower().split()
    if not words:
        return []
    labels = np.asarray(labels, dtype=int)
    w = _width(labels)
    out = []
    for i, r in enumerate(rows):
        name = (r.get("project_name") or "").lower()
        if all(x in name for x in words):
            out.append({"i": i, "name": r.get("project_name", ""), "year": r.get("_year"),
                        "department": r.get("department", ""),
                        "label": int(labels[i]), "id": label_id(labels[i], w, prefix)})
            if len(out) >= limit:
                break
    return out


# ---------------------------------------------------------------- editing --

class Draft:
    """Editable cluster labels with undo. Never writes anything itself.

    labels   np.ndarray, one label per store row (-1 = noise)
    Every edit returns the label it affected and can be undone, then redone.
    A new edit after an undo discards the redo steps, as in any editor.
    """

    def __init__(self, labels, history: int = 100):
        self.labels = np.asarray(labels, dtype=int).copy()
        self._hist: list[tuple[np.ndarray, str]] = []   # (labels before, what)
        self._redo: list[tuple[np.ndarray, str]] = []   # (labels after, what)
        self._limit = history

    @classmethod
    def fit(cls, vecs, clusterer=None, **kw) -> "Draft":
        """Cluster `vecs` and start a draft numbered by size."""
        return cls(relabel(fit(vecs, clusterer, **kw)))

    # ---------------------------------------------------------- helpers
    @property
    def can_undo(self) -> bool:
        return bool(self._hist)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def undo_label(self) -> str | None:
        """What undo() would revert, e.g. 'move 2 rows to cl0007'."""
        return self._hist[-1][1] if self._hist else None

    @property
    def redo_label(self) -> str | None:
        """What redo() would re-apply."""
        return self._redo[-1][1] if self._redo else None

    def _id(self, label: int) -> str:
        return label_id(label, _width(self.labels)) or "noise"

    def new_label(self) -> int:
        return int(self.labels.max(initial=-1)) + 1

    def _commit(self, new: np.ndarray, what: str) -> None:
        self._hist.append((self.labels, what))
        del self._hist[:-self._limit]
        self._redo.clear()
        self.labels = new

    def _check(self, rows) -> list[int]:
        rows = [int(i) for i in rows]
        bad = [i for i in rows if not 0 <= i < len(self.labels)]
        if bad:
            raise ValueError(f"no row {bad[0]}")
        return rows

    # ------------------------------------------------------------ edits
    def move(self, rows, to: int | None = None) -> int:
        """Put `rows` in cluster `to` (-1 = noise, None = a new cluster)."""
        rows = self._check(rows)
        if not rows:
            raise ValueError("no rows selected")
        to = self.new_label() if to is None else int(to)
        if to < NOISE:
            raise ValueError(f"bad cluster {to}")
        new = self.labels.copy()
        new[rows] = to
        self._commit(new, f"move {len(rows)} row{'s' * (len(rows) != 1)} to "
                          f"{'new cluster ' if to == self.new_label() else ''}{self._id(to)}")
        return to

    def merge(self, sources, target: int | None = None) -> int:
        """Fold every cluster in `sources` into `target` (default: the largest)."""
        sources = {int(s) for s in sources if int(s) >= 0}
        if target is None:
            if not sources:
                raise ValueError("nothing to merge")
            target = max(sources, key=lambda l: (int((self.labels == l).sum()), -l))
        target = int(target)
        if target < 0:
            raise ValueError("merge into noise is dissolve()")
        others = sorted(sources - {target})
        new = self.labels.copy()
        new[np.isin(new, others)] = target
        self._commit(new, f"merge {', '.join(self._id(l) for l in others) or 'nothing'} "
                          f"into {self._id(target)}")
        return target

    def dissolve(self, label: int) -> int:
        """Send every member of `label` to noise."""
        label = int(label)
        if label < 0:
            raise ValueError("noise is not a cluster")
        new = self.labels.copy()
        new[new == label] = NOISE
        self._commit(new, f"dissolve {self._id(label)}")
        return NOISE

    def prune(self, vecs, threshold: float, label: int | None = None,
              dry_run: bool = False) -> np.ndarray:
        """Send members whose fit is below `threshold` to noise.

        label    one cluster to prune, or None for every cluster
        dry_run  only report which rows would go; the draft is not changed
        Fit is measured against each cluster's centre before anything moves
        (see fits()). Returns the indices of the rows removed (or that would be).
        """
        f = fits(self.labels, vecs)
        hit = (f < float(threshold)) & (self.labels >= 0)   # NaN compares False
        if label is not None:
            hit &= self.labels == int(label)
        rows = np.flatnonzero(hit)
        if len(rows) and not dry_run:
            new = self.labels.copy()
            new[rows] = NOISE
            where = self._id(label) if label is not None else "all clusters"
            self._commit(new, f"prune {len(rows)} rows below {float(threshold):.2f} in {where}")
        return rows

    def undo(self) -> str | None:
        """Revert the last edit. Returns what was undone, or None if nothing."""
        if not self._hist:
            return None
        before, what = self._hist.pop()
        self._redo.append((self.labels, what))
        self.labels = before
        return what

    def redo(self) -> str | None:
        """Re-apply the last undone edit. Returns what was redone, or None."""
        if not self._redo:
            return None
        after, what = self._redo.pop()
        self._hist.append((self.labels, what))
        self.labels = after
        return what


# ---------------------------------------------------------------- writing --

def write(store, changes: list[wb.Change], **kw) -> dict:
    """Write `changes` to the store's CSVs and update the rows in memory.

    Keyword arguments go to writeback.write_changes(). Returns its report.
    """
    res = wb.write_changes(store.csv_dir, changes, **kw)
    bad = {(s["file"], s["row"]) for s in res["skipped"]}
    by_pos = {(r["_file"], r["_line"]): r for r in store.rows}
    for c in changes:
        if (c.file, c.row) not in bad and (c.file, c.row) in by_pos:
            by_pos[(c.file, c.row)][wb.MARK_COL] = c.mark
    return res


def run(store, *, vecs=None, labels=None, clusterer=None, overwrite: bool = False,
        dry_run: bool = False, prefix: str = PREFIX, relabel: bool = True) -> dict:
    """Cluster, plan and (unless dry_run) write, in one call.

    vecs     defaults to store.vecs; must line up with store.rows
    labels   use these instead of clustering (e.g. Draft.labels)
    relabel  renumber by size before writing; pass False for Draft labels so
             the ids written are the ids you inspected
    Returns summarize() + {'labels', 'counts', 'written', 'files', 'skipped'}.
    'written' is what would be written when dry_run is True.
    """
    if labels is None:
        vecs = store.vecs if vecs is None else vecs
        if vecs is None:
            raise RuntimeError("no vectors: create or load an embedding first")
        if len(vecs) != len(store.rows):
            raise RuntimeError(f"{len(vecs)} vectors for {len(store.rows)} rows; "
                               "re-embed so they line up")
        labels = fit(vecs, clusterer)
    labels = np.asarray(labels, dtype=int)
    if len(labels) != len(store.rows):
        raise RuntimeError(f"{len(labels)} labels for {len(store.rows)} rows")

    ids = cluster_ids(labels, prefix, relabel_first=relabel)
    changes, counts = plan(store.rows, ids, overwrite=overwrite, prefix=prefix)
    out = {**summarize(store.rows, ids), "labels": labels.tolist(), "counts": counts}
    if dry_run:
        return {**out, "written": len(changes),
                "files": sorted({c.file for c in changes}), "skipped": []}
    res = write(store, changes)
    return {**out, "written": res["written"], "files": res["files"],
            "skipped": res["skipped"]}
