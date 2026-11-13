"""Write project ids back into the source CSVs — no Flask, no model.

Every function here works on plain paths and values, so a script can use it
without loading the app or the embedding store:

    import writeback as wb

    changes = [
        wb.Change("2012.csv", row=14, pid="P-0231", mark=wb.mark_value("manual")),
        wb.Change("2013.csv", row=3,  pid="P-0231", mark=wb.mark_value("auto", 0.84)),
    ]
    report = wb.write_changes(r"C:\\...\\Dallas\\CSV", changes)
    # {'files': ['2012.csv', '2013.csv'], 'written': 2, 'skipped': []}

What a write does to a file
    * every row is kept, in order, including rows the review tool ignores
      (blank names, blank lines) and any columns it does not know about
    * `project_id` is added after `project_name` if the book has no such column
    * a marker column (MARK_COL) is appended once; rows this tool changed get a
      value such as "manual" or "auto 0.842", every other row is left blank,
      and marks from earlier runs are preserved
    * the first write to a file copies it to `<file>.bak`; later writes do not
      touch that backup, so it always holds the untouched original
    * the file is written to a temp file and swapped in, so a crash never
      leaves half a CSV; encoding (BOM or not) and line endings are preserved

`row` is the 0-based index of the data row as csv.DictReader numbers it (blank
lines are not counted), which is what VecStore stores as `_line`.
"""

from __future__ import annotations

import csv
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass

__all__ = [
    "ID_COL", "MARK_COL", "NAME_COL", "Change",
    "mark_value", "read_csv", "write_csv", "ensure_columns",
    "backup_once", "apply_changes", "write_file", "write_changes",
]

ID_COL   = "project_id"
MARK_COL = "id_modified"       # the column that says "this tool set this id"
NAME_COL = "project_name"


@dataclass(frozen=True)
class Change:
    """One id to write: put `pid` in data row `row` of `file`.

    file   base name of the CSV inside the csv folder, e.g. "2012.csv"
    row    0-based data-row index (csv.DictReader numbering)
    pid    the project id to write, or None to leave project_id untouched
           and write only the marker (used by clusters.py)
    mark   value for MARK_COL; see mark_value()
    name   project_name the row had when it was loaded. When given, the write
           refuses that row if the file now says something else there, which
           catches a CSV edited by hand since the store was loaded.
    """
    file: str
    row: int
    pid: str | None
    mark: str = "manual"
    name: str | None = None


def mark_value(how: str = "manual", score: float | None = None) -> str:
    """Marker text: 'manual', or 'auto 0.842' when a score is given."""
    return f"{how} {score:.3f}" if score is not None else how


# ----------------------------------------------------------------- file I/O --

def read_csv(path: str) -> dict:
    """Read a CSV without losing anything.

    Returns {'header': [...], 'rows': [[...], ...], 'data': [i, ...],
             'bom': bool, 'newline': '\\r\\n' | '\\n'}
    `rows` holds every line after the header, blank ones included; `data[n]`
    is the position in `rows` of data row n (blank lines skipped, the same
    numbering csv.DictReader uses).
    """
    with open(path, "rb") as f:
        head = f.read(65536)
    bom = head.startswith(b"\xef\xbb\xbf")
    newline = "\r\n" if b"\r\n" in head or b"\n" not in head else "\n"

    with open(path, newline="", encoding="utf-8-sig") as f:
        lines = list(csv.reader(f))
    header, rows = (lines[0], lines[1:]) if lines else ([], [])
    data = [i for i, r in enumerate(rows) if r]
    return {"header": header, "rows": rows, "data": data,
            "bom": bom, "newline": newline}


def write_csv(path: str, header: list, rows: list, *, bom: bool = False,
              newline: str = "\r\n") -> None:
    """Write header + rows atomically (temp file, then replace)."""
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8-sig" if bom else "utf-8") as f:
        w = csv.writer(f, lineterminator=newline)
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, path)


def backup_once(path: str) -> str | None:
    """Copy `path` to `path.bak` unless a backup already exists.

    Returns the backup path when one was made now, else None.
    """
    bak = path + ".bak"
    if os.path.exists(bak):
        return None
    shutil.copy2(path, bak)
    return bak


# ------------------------------------------------------------ transforms --

def ensure_columns(header: list, rows: list, id_col: str = ID_COL,
                   mark_col: str = MARK_COL, name_col: str = NAME_COL,
                   add_id: bool = True) -> dict:
    """Add id_col / mark_col to the header if missing, shifting rows to match.

    Mutates `header` and `rows` in place. id_col goes right after name_col (or
    at the end when there is no name column); mark_col goes at the end.
    add_id=False never adds id_col (for marker-only writes).
    Returns {'id': index or None, 'mark': index, 'name': index or None}.
    """
    width_before = len(header)
    if add_id and id_col not in header:
        at = header.index(name_col) + 1 if name_col in header else len(header)
        header.insert(at, id_col)
        for r in rows:
            if r and len(r) >= at:
                r.insert(at, "")
    if mark_col not in header:
        at = len(header)
        header.append(mark_col)
        for r in rows:                 # malformed rows with surplus cells keep
            if len(r) > at:            # them, shifted right of the new column
                r.insert(at, "")
    if len(header) > width_before:        # a column was added: give every
        for r in rows:                     # row a cell for it, so untouched
            if r and len(r) < len(header): # rows read as blank, not missing
                r.extend([""] * (len(header) - len(r)))
    return {"id": header.index(id_col) if id_col in header else None,
            "mark": header.index(mark_col),
            "name": header.index(name_col) if name_col in header else None}


def apply_changes(table: dict, changes: list[Change], cols: dict) -> tuple[list, list]:
    """Set id and mark for each change in a table from read_csv(). In place.

    Returns (applied, skipped, unchanged): applied are rows that now differ,
    skipped holds (change, reason) pairs, unchanged already had these values.
    """
    rows, data, width = table["rows"], table["data"], len(table["header"])
    applied, skipped, unchanged = [], [], []
    for c in changes:
        if not 0 <= c.row < len(data):
            skipped.append((c, f"row {c.row} not in file"))
            continue
        r = rows[data[c.row]]
        if len(r) < width:
            r.extend([""] * (width - len(r)))
        if c.name is not None and cols["name"] is not None \
                and r[cols["name"]].strip() != c.name.strip():
            skipped.append((c, "row changed on disk since load"))
            continue
        if c.pid is not None and cols["id"] is None:
            skipped.append((c, "file has no id column"))
            continue
        before = list(r)
        if c.pid is not None:
            r[cols["id"]] = c.pid
        r[cols["mark"]] = c.mark
        (applied if r != before else unchanged).append(c)
    return applied, skipped, unchanged


# --------------------------------------------------------------- writers --

def write_file(path: str, changes: list[Change], *, id_col: str = ID_COL,
               mark_col: str = MARK_COL, name_col: str = NAME_COL,
               make_backup: bool = True) -> dict:
    """Apply `changes` to one CSV and write it back. Rows not named are untouched.

    Returns {'file', 'written', 'unchanged', 'skipped': [{'row', 'pid', 'reason'}],
             'backup'}. Nothing is written (and no backup made) when no row
    actually changes.
    """
    table = read_csv(path)
    cols = ensure_columns(table["header"], table["rows"], id_col, mark_col, name_col,
                          add_id=any(c.pid is not None for c in changes))
    applied, skipped, unchanged = apply_changes(table, changes, cols)
    bak = None
    if applied:
        if make_backup:
            bak = backup_once(path)
        write_csv(path, table["header"], table["rows"],
                  bom=table["bom"], newline=table["newline"])
    return {"file": os.path.basename(path), "written": len(applied),
            "unchanged": len(unchanged),
            "skipped": [{"row": c.row, "pid": c.pid, "reason": why}
                        for c, why in skipped],
            "backup": bak}


def write_changes(csv_dir: str, changes: list[Change], **kw) -> dict:
    """Group `changes` by file and write each file once.

    Keyword arguments are passed to write_file(). Returns
    {'files': [names written], 'written': n, 'unchanged': n,
     'skipped': [...], 'per_file': [...]}.
    """
    by_file: dict[str, list[Change]] = defaultdict(list)
    for c in changes:
        by_file[c.file].append(c)

    per_file, skipped = [], []
    for fn in sorted(by_file):
        path = os.path.join(csv_dir, fn)
        if not os.path.isfile(path):
            skipped += [{"file": fn, "row": c.row, "pid": c.pid,
                         "reason": "file not found"} for c in by_file[fn]]
            continue
        res = write_file(path, by_file[fn], **kw)
        per_file.append(res)
        skipped += [{"file": fn, **s} for s in res["skipped"]]

    return {"files": [r["file"] for r in per_file if r["written"]],
            "written": sum(r["written"] for r in per_file),
            "unchanged": sum(r["unchanged"] for r in per_file),
            "skipped": skipped, "per_file": per_file}
