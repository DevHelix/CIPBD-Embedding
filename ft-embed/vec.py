import csv, math, os, re, shutil
from collections import defaultdict

import numpy as np

try:                                     # optional: falls back to string distance
    from sentence_transformers import SentenceTransformer
except ImportError:                      # pragma: no cover
    SentenceTransformer = None
from rapidfuzz import fuzz

YEAR_FILE = re.compile(r"^\d{4}\.csv$")


def num(v):
    s = str(v or "").strip().replace(",", "").replace("$", "")
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s) if s else 0.0
    except ValueError:
        return 0.0


class VecStore:
    """Holds the rows, their vectors, and the suggestion logic."""

    def __init__(self, csv_dir, out_dir, model_dir=None):
        self.csv_dir   = csv_dir
        self.out_dir   = out_dir
        self.model_dir = model_dir
        self._model    = None
        self.rows, self.fields_of = [], {}
        self.vecs = None
        os.makedirs(out_dir, exist_ok=True)

    # ---------------------------------------------------------------- model
    @property
    def model(self):
        if self._model is None and SentenceTransformer and self.model_dir \
                and os.path.isdir(self.model_dir):
            self._model = SentenceTransformer(self.model_dir)
        return self._model

    @property
    def using_vectors(self):
        return self.vecs is not None

    # ----------------------------------------------------------- load/embed
    def load_rows(self):
        """Read every <year>.csv, remembering where each row came from."""
        self.rows, self.fields_of = [], {}
        for fn in sorted(os.listdir(self.csv_dir)):
            if not YEAR_FILE.match(fn):
                continue
            path = os.path.join(self.csv_dir, fn)
            with open(path, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                self.fields_of[fn] = list(reader.fieldnames)
                for n, row in enumerate(reader):
                    if not (row.get("project_name") or "").strip():
                        continue
                    row["_file"] = fn            # so an edit can be written back
                    row["_line"] = n
                    row["_i"] = len(self.rows)
                    row["_year"] = int(num(row.get("cip_year")))
                    row["_total"] = num(row.get("project_total"))
                    row["_pid"] = (row.get("project_id") or "").strip()
                    self.rows.append(row)
        self._index()
        return len(self.rows)

    def _index(self):
        self.anchored = [r for r in self.rows if r["_pid"]]
        self.orphans  = [r for r in self.rows if not r["_pid"]]
        self.by_pid = defaultdict(list)
        for r in self.anchored:
            self.by_pid[r["_pid"]].append(r)
        self.years_of = {p: {x["_year"] for x in v} for p, v in self.by_pid.items()}

    def embed(self, progress=None):
        names = [r["project_name"] for r in self.rows]
        if self.model is None:
            self.vecs = None                       # scoring falls back to fuzz
            if progress:
                progress("no model found - scoring on names only")
            return 0
        out, step = [], 512
        for i in range(0, len(names), step):
            out.extend(self.model.encode(names[i:i + step],
                                         normalize_embeddings=True))
            if progress:
                progress(f"embedded {min(i + step, len(names))}/{len(names)}")
        self.vecs = np.asarray(out)
        np.save(os.path.join(self.out_dir, "vecs.npy"), self.vecs, allow_pickle=False)
        return len(self.vecs)

    def load_vectors(self):
        p = os.path.join(self.out_dir, "vecs.npy")
        if not os.path.exists(p):
            return False
        v = np.load(p)
        if len(v) != len(self.rows):
            return False                           # stale: lengths must agree
        self.vecs = v
        return True

    # -------------------------------------------------------------- scoring
    def sim(self, a, b):
        if self.vecs is not None:
            return float(self.vecs[a["_i"]] @ self.vecs[b["_i"]])
        return fuzz.token_sort_ratio(a["project_name"], b["project_name"]) / 100

    @staticmethod
    def money_fit(a, b):
        d = abs(math.log1p(max(a["_total"], 0)) - math.log1p(max(b["_total"], 0)))
        return 1 - min(d / 3.0, 1.0)

    def score(self, orphan, cand):
        return (0.60 * self.sim(orphan, cand)
                + 0.20 * self.money_fit(orphan, cand)
                + 0.10 * (orphan.get("department") == cand.get("department"))
                + 0.10 * (1 / (1 + abs(orphan["_year"] - cand["_year"]))))

    def candidates(self, orphan, k=6):
        pool = [r for r in self.anchored
                if r.get("department") == orphan.get("department")] or self.anchored
        best = {}
        for c in pool:
            if c["_year"] == orphan["_year"]:         # one row per project per book
                continue
            s = self.score(orphan, c)
            if c["_pid"] not in best or s > best[c["_pid"]][0]:
                best[c["_pid"]] = (s, c)
        best = {p: v for p, v in best.items()          # year slot already taken
                if orphan["_year"] not in self.years_of.get(p, set())}
        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[:k]

        out = []
        for pid, (s, c) in ranked:
            members = sorted(self.by_pid[pid], key=lambda r: r["_year"])
            out.append({"pid": pid, "score": round(s, 4),
                        "sim": round(self.sim(orphan, c), 4),
                        "money": round(self.money_fit(orphan, c), 3),
                        "same_dept": orphan.get("department") == c.get("department"),
                        "best_name": c["project_name"],
                        "department": c.get("department", ""),
                        "series": [[m["_year"], int(m["_total"])] for m in members]})
        margin = round(out[0]["score"] - out[1]["score"], 4) if len(out) > 1 else \
                 (out[0]["score"] if out else 0.0)
        return out, margin

    # -------------------------------------------------------------- editing
    def assign(self, row_index, pid):
        """Set project_id on one row and rewrite that row's source CSV."""
        row = self.rows[row_index]
        row["project_id"] = pid
        row["_pid"] = pid
        self._rewrite(row["_file"])
        self._index()
        return row

    def _rewrite(self, fn):
        path = os.path.join(self.csv_dir, fn)
        backup = path + ".bak"
        if not os.path.exists(backup):              # one backup per file, once
            shutil.copy2(path, backup)

        fields = self.fields_of[fn]
        if "project_id" not in fields:              # the early books have no such
            at = fields.index("project_name") + 1 if "project_name" in fields \
                 else len(fields)                   # column; add it rather than
            fields = fields[:at] + ["project_id"] + fields[at:]   # silently drop
            self.fields_of[fn] = fields

        rows = sorted((r for r in self.rows if r["_file"] == fn),
                      key=lambda r: r["_line"])
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields,
                               restval="", extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    def stats(self):
        return {"rows": len(self.rows), "anchored": len(self.anchored),
                "orphans": len(self.orphans), "vectors": bool(self.using_vectors),
                "csv_dir": self.csv_dir}
