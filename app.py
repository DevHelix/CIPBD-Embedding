"""Interactive browser for the embedding hierarchy.

Run:  python app.py      then open http://127.0.0.1:5000
"""
import os, sys, re
from collections import Counter, defaultdict

import numpy as np
from flask import Flask, jsonify, render_template, request
from scipy.cluster.hierarchy import linkage, fcluster, to_tree

sys.setrecursionlimit(100_000)

CITY = "Dallas"
OUT_PATH = os.environ.get(
    "DUMP_PATH",
    rf"C:\Users\vince\Documents\GitHub\CIPBD-Embedding\Outputs\{CITY}\dump.npy")
DEDUPE = True                      # collapse repeated names before clustering

app = Flask(__name__)


# --------------------------------------------------------------- data ------
class Model:
    """Loads once, clusters once. linkage is the expensive step - keep it cached."""

    def __init__(self, path):
        X     = np.load(path, mmap_mode="r")
        names = np.load(path.replace(".npy", "_names.npy"), allow_pickle=True)
        ypath = path.replace(".npy", "_years.npy")
        years = np.load(ypath, allow_pickle=True) if os.path.exists(ypath) else None

        if len(X) != len(names):
            raise SystemExit(
                f"{len(X)} vectors but {len(names)} names - the files were written "
                f"by different runs and cannot be aligned. Re-embed before using this.")

        if DEDUPE:
            first, order = {}, []
            spans = defaultdict(set)
            for i, n in enumerate(names):
                n = str(n)
                if n not in first:
                    first[n] = i
                    order.append(i)
                if years is not None:
                    spans[n].add(int(years[i]))
            X     = np.asarray(X)[order]
            names = np.array([str(names[i]) for i in order])
            years = (np.array([",".join(map(str, sorted(spans[str(n)]))) for n in names])
                     if years is not None else None)
        else:
            X = np.asarray(X)

        self.names = names
        self.years = years
        self.Z     = linkage(X.astype(np.float64), method="average", metric="cosine")
        self.root, _ = to_tree(self.Z, rd=True)
        self.n     = len(names)

        self.members = {}
        self._leaves(self.root)

    def _leaves(self, nd):
        if nd.id in self.members:
            return self.members[nd.id]
        s = {nd.id} if nd.is_leaf() else self._leaves(nd.left) | self._leaves(nd.right)
        self.members[nd.id] = s
        return s

    # ---- cluster roots at a given cut -------------------------------------
    def tops(self, threshold):
        labels = fcluster(self.Z, t=threshold, criterion="distance")
        counts = Counter(labels)
        out = {}

        def find(nd):
            labs = {labels[i] for i in self.members[nd.id]}
            if len(labs) == 1:
                out.setdefault(labs.pop(), nd)
                return
            if not nd.is_leaf():
                find(nd.left); find(nd.right)

        find(self.root)
        return out, counts

    def node_json(self, nd):
        if nd.is_leaf():
            return {"leaf": True, "name": str(self.names[nd.id]),
                    "years": str(self.years[nd.id]) if self.years is not None else ""}
        return {"leaf": False, "sim": round(1 - nd.dist, 4),
                "children": [self.node_json(nd.left), self.node_json(nd.right)]}

    def span_years(self, nd):
        if self.years is None:
            return 0
        ys = set()
        for i in self.members[nd.id]:
            ys.update(str(self.years[i]).split(","))
        return len({y for y in ys if y})


MODEL = Model(OUT_PATH)


# -------------------------------------------------------------- routes -----
@app.route("/")
def index():
    return render_template("index.html", total=MODEL.n,
                           has_years=MODEL.years is not None)


@app.route("/api/clusters")
def clusters():
    threshold = float(request.args.get("threshold", 0.25))
    min_size  = int(request.args.get("min_size", 2))
    cross     = request.args.get("cross") == "1"
    q         = (request.args.get("q") or "").strip().lower()
    limit     = int(request.args.get("limit", 100))

    tops, counts = MODEL.tops(threshold)
    groups = [(lab, nd) for lab, nd in tops.items() if counts[lab] >= min_size]
    groups.sort(key=lambda t: t[1].dist)

    out, shown = [], 0
    for lab, nd in groups:
        member_names = [str(MODEL.names[i]) for i in sorted(MODEL.members[nd.id])]
        if q and not any(q in m.lower() for m in member_names):
            continue
        spans = MODEL.span_years(nd)
        if cross and spans < 2:
            continue
        shown += 1
        if shown > limit:
            break
        out.append({"label": int(lab), "size": int(counts[lab]),
                    "sim": round(1 - nd.dist, 4), "years_spanned": spans,
                    "tree": MODEL.node_json(nd)})

    return jsonify({"clusters": out, "total_clusters": len(tops),
                    "matching": shown, "truncated": shown > limit})


@app.route("/api/pairs")
def pairs():
    k = int(request.args.get("k", 50))
    n = MODEL.n
    rep, rows = {i: str(MODEL.names[i]) for i in range(n)}, []
    for m, (a, b, d, size) in enumerate(MODEL.Z):
        a, b = int(a), int(b)
        rep[n + m] = rep[a]
        rows.append((float(d), rep[a], rep[b], int(size)))
    rows.sort()
    return jsonify([{"sim": round(1 - d, 4), "size": s, "a": x, "b": y}
                    for d, x, y, s in rows[:k]])


if __name__ == "__main__":
    app.run(debug=False, port=int(os.environ.get("PORT", 5000)))
