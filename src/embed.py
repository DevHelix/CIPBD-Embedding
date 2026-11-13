import csv, math, os, re
from collections import defaultdict

import numpy as np

try:                                     # optional: falls back to string distance
    from sentence_transformers import SentenceTransformer
except ImportError:                      # pragma: no cover
    SentenceTransformer = None
from rapidfuzz import fuzz

import writeback

YEAR_FILE = re.compile(r"^\d{4}\.csv$")
DEFAULT_WEIGHTS = {"sim": 0.60, "money": 0.20, "dept": 0.10, "year": 0.10}

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
        self.weights = dict(DEFAULT_WEIGHTS)
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

    def set_weights(self, w):
        """Accept a partial dict, coerce to float, normalise to sum 1."""
        new = dict(self.weights)
        for k in DEFAULT_WEIGHTS:
            if k in w:
                try:
                    new[k] = max(0.0, float(w[k]))
                except (TypeError, ValueError):
                    pass
        total = sum(new.values())
        if total <= 0:
            new = dict(DEFAULT_WEIGHTS)
            total = 1.0
        self.weights = {k: v / total for k, v in new.items()}
        return self.weights

    def score(self, orphan, cand):
        w = self.weights
        return (w["sim"]   * self.sim(orphan, cand)
              + w["money"] * self.money_fit(orphan, cand)
              + w["dept"]  * (orphan.get("department") == cand.get("department"))
              + w["year"]  * (1 / (1 + abs(orphan["_year"] - cand["_year"]))))

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
    def assign(self, row_index, pid, how="manual", score=None):
        """Set project_id on one row and write it to its source CSV.

        Raises RuntimeError if the row could not be written (e.g. the file was
        edited on disk since load_rows()).
        """
        res = self.assign_many([(row_index, pid, writeback.mark_value(how, score))])
        if res["skipped"]:
            raise RuntimeError(res["skipped"][0]["reason"])
        return self.rows[row_index]

    def assign_many(self, items):
        """Write many ids at once; each touched CSV is rewritten exactly once.

        items: iterable of (row_index, pid, mark), mark from writeback.mark_value().
        Rows that the write skipped keep their old id in memory too.
        Returns writeback.write_changes()'s report.
        """
        staged = {}
        for i, pid, mark in items:
            row = self.rows[i]
            c = writeback.Change(row["_file"], row["_line"], str(pid).strip(),
                                 mark, row.get("project_name"))
            staged[(c.file, c.row)] = (row, c)

        res = writeback.write_changes(self.csv_dir, [c for _, c in staged.values()])

        bad = {(s["file"], s["row"]) for s in res["skipped"]}
        for key, (row, c) in staged.items():
            if key not in bad:
                row["project_id"] = row["_pid"] = c.pid
                row[writeback.MARK_COL] = c.mark
        self._index()
        return res

    def propose(self, threshold=0.7, dept=None):
        """Top candidate for every orphan whose score >= threshold. Writes nothing.

        Two orphans from the same year never get the same id: the higher score
        wins and the other is left for manual review (counted in 'conflicts').
        Returns {'considered', 'conflicts', 'proposals': [{'i', 'name', 'year',
        'department', 'file', 'pid', 'score', 'margin'}, ...]} best first.
        """
        considered, picks = 0, []
        for r in self.orphans:
            if dept and r.get("department") != dept:
                continue
            considered += 1
            cands, margin = self.candidates(r, k=2)
            if cands and cands[0]["score"] >= threshold:
                picks.append({"i": r["_i"], "name": r["project_name"],
                              "year": r["_year"], "department": r.get("department", ""),
                              "file": r["_file"], "pid": cands[0]["pid"],
                              "score": cands[0]["score"], "margin": margin})
        picks.sort(key=lambda p: -p["score"])
        taken, out = set(), []
        for p in picks:
            if (p["pid"], p["year"]) not in taken:
                taken.add((p["pid"], p["year"]))
                out.append(p)
        return {"considered": considered, "conflicts": len(picks) - len(out),
                "proposals": out}

    def auto_assign(self, threshold=0.7, dept=None, dry_run=False):
        """propose() and then write every proposal, marked 'auto <score>'.

        With dry_run=True nothing is written; 'written' is what would be.
        Returns propose()'s keys plus 'written', 'files', 'skipped'.
        """
        plan = self.propose(threshold, dept)
        if dry_run:
            return {**plan, "written": len(plan["proposals"]),
                    "files": sorted({p["file"] for p in plan["proposals"]}),
                    "skipped": []}
        res = self.assign_many(
            (p["i"], p["pid"], writeback.mark_value("auto", p["score"]))
            for p in plan["proposals"])
        return {**plan, "written": res["written"], "files": res["files"],
                "skipped": res["skipped"]}

    def stats(self):
        return {"rows": len(self.rows), "anchored": len(self.anchored),
                "orphans": len(self.orphans), "vectors": bool(self.using_vectors),
                "csv_dir": self.csv_dir}
