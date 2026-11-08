from embed import VecStore
import os
import paths
from evoc import EVoC
import numpy as np
from collections import defaultdict
import csv

city = "Syracuse"
csv_dir = rf"C:\Users\vince\Documents\GitHub\CIPBD\{city}\CSV"
out_dir = r"C:\Users\vince\Documents\GitHub\embed-revamp\data\outputs"

clusterer = EVoC()
p = os.path.join(out_dir, "vecs.npy")
vecs   = np.load(p)
labels = clusterer.fit_predict(vecs)

s = VecStore(csv_dir, out_dir, None)
s.load_rows()
# assert len(vecs) == len(s.rows), f"{len(vecs)} vectors vs {len(s.rows)} rows"

groups = defaultdict(list)
noise  = 0
for row, lab in zip(s.rows, labels):
    if lab < 0:
        noise += 1
        continue
    groups[int(lab)].append(row)

print(f"{len(groups)} clusters, {noise} unassigned of {len(s.rows)} rows\n")

for lab, members in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
    print(f"── cluster {lab}  ({len(members)} rows)")
    for r in sorted(members, key=lambda r: (r["_year"], r["project_name"])):
        pid = r["_pid"] or "—"
        print(f"   {r['_year']}  {pid:<8}  {r['project_name']}")
    print()

with open(os.path.join(out_dir, f"clusters-{city}.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["cluster", "size", "cip_year", "project_id", "project_name", "department"])
    for lab, members in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        for r in sorted(members, key=lambda r: r["_year"]):
            w.writerow([lab, len(members), r["_year"], r["_pid"],
                        r["project_name"], r.get("department", "")])