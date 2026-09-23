import sys
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, to_tree
from collections import Counter

sys.setrecursionlimit(100000)      # the tree is walked recursively


def build(X):
    return linkage(np.asarray(X), method="average", metric="cosine")


def hierarchy(Z, names, years=None, threshold=0.25, min_size=2, max_clusters=None):
    """Print each cluster as an indented merge tree, tightest first."""
    labels = fcluster(Z, t=threshold, criterion="distance")
    counts = Counter(labels)
    root, nodes = to_tree(Z, rd=True)

    members = {}                                   # node id -> set of leaf ids
    def leaves(nd):
        if nd.id in members:
            return members[nd.id]
        s = {nd.id} if nd.is_leaf() else leaves(nd.left) | leaves(nd.right)
        members[nd.id] = s
        return s
    leaves(root)

    # the highest node whose leaves all share one label is that cluster's root
    tops = {}
    def find(nd):
        labs = {labels[i] for i in members[nd.id]}
        if len(labs) == 1:
            tops.setdefault(labs.pop(), nd)
            return
        if not nd.is_leaf():
            find(nd.left); find(nd.right)
    find(root)

    groups = [(lab, nd) for lab, nd in tops.items() if counts[lab] >= min_size]
    groups.sort(key=lambda t: t[1].dist)           # tightest cluster first
    if max_clusters:
        groups = groups[:max_clusters]

    for lab, nd in groups:
        print(f"\n=== cluster {lab}  ({counts[lab]} names, "
              f"similarity {1 - nd.dist:.3f})")
        show(nd, names, years, "", True)

    print(f"\n{len(tops)} clusters at threshold {threshold}, "
          f"{len(groups)} with {min_size}+ names")


def show(nd, names, years, pad, last):
    branch = "└─ " if last else "├─ "
    if nd.is_leaf():
        tag = f"  ({years[nd.id]})" if years is not None else ""
        print(f"{pad}{branch}{names[nd.id]}{tag}")
        return
    print(f"{pad}{branch}[{1 - nd.dist:.3f}]")     # similarity at this merge
    pad2 = pad + ("   " if last else "│  ")
    show(nd.left,  names, years, pad2, False)
    show(nd.right, names, years, pad2, True)


def top_pairs(Z, names, k=25):
    """The k tightest merges in the whole tree, regardless of level."""
    n = len(names)
    rep, rows = {i: names[i] for i in range(n)}, []
    for m, (a, b, d, size) in enumerate(Z):
        a, b = int(a), int(b)
        rep[n + m] = rep[a]
        rows.append((d, rep[a], rep[b], int(size)))
    rows.sort()
    print(f"{'sim':>6}  {'n':>3}  pair")
    for d, x, y, size in rows[:k]:
        print(f"{1 - d:6.3f}  {size:3d}  {x}  <->  {y}")

CITY = "Dallas"
OUT_PATH = rf'C:\Users\vince\Documents\GitHub\CIPDB-Embedding\Outputs\{CITY}\dump.npy'
X     = np.load(OUT_PATH, mmap_mode='r')
names = np.load(OUT_PATH.replace('.npy', '_names.npy'))
years = np.load(OUT_PATH.replace('.npy', '_years.npy'))   # optional

Z = build(X)
top_pairs(Z, names, k=25)
hierarchy(Z, names, years, threshold=0.25, min_size=2)