import pandas as pd
import pickle
from collections import defaultdict
import itertools
import re
from rapidfuzz import fuzz
import csv
import os
from datasets import Dataset
from datasets import Features, Value, Sequence

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses
)

class loss_pairing:
    def __init__(self, csv_dir=None, train_dir=None, anchorColumn=None):
        
        self.csv_dir   = csv_dir
        self.train_dir = train_dir
        self.anchorColumn = anchorColumn

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.guide = SentenceTransformer("all-MiniLM-L6-v2")

        self.ds = {}
        self.differences = defaultdict(list)

        self.YEAR_COL = re.compile(r"^year_(\d{4})$")
        self.span = []
        self.name = lambda k: f"plan_m{-k}" if k < 0 else f"plan_{k}"
        self.lo, self.hi = 0, 0

    def getData(self):
        files = [f for f in sorted(os.listdir(self.csv_dir)) if re.fullmatch(r'\d{4}\.csv', f)]
        def num(v):
            s = str(v or '').strip().replace(',', '').replace('$', '')
            if s.startswith('(') and s.endswith(')'):
                s = '-' + s[1:-1]
            if not s or s in ('-', '—', 'N/A', 'NA', 'TBD'):
                return 0
            try:
                return int(float(s))
            except ValueError:
                return 0

        def txt(v):
            return (v or '').strip()

        full = []

        def offsets_in(header, cip):
            """{offset: column_name} for the year columns this file actually has."""
            out = {}
            for c in header:
                m = self.YEAR_COL.match(c)
                if m:
                    out[int(m.group(1)) - cip] = c
            return out

        def getRange(file):
            with open(os.path.join(self.csv_dir, file), newline="", encoding="utf-8-sig") as fh:
                r = csv.DictReader(fh)
                header, first = r.fieldnames, next(r, None)
                if not first:
                    return
                offs = offsets_in(header, int(first["cip_year"]))
                if offs:
                    self.lo, self.hi = min(self.lo, min(offs)), max(self.hi, max(offs))

        def accumulate(file):
            file_dir = os.path.join(self.csv_dir, file)
            with open(file_dir, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                header = reader.fieldnames
                for row in reader:
                    cip = int(row["cip_year"])
                    offs = offsets_in(header,cip)

                    rec = {
                        "project_name": txt(row.get("project_name")),
                        "project_id":   txt(row.get("project_id")),
                        "cip_year":     cip,
                        "department":   txt(row.get("department")),
                        "project_type": txt(row.get("project_type")),
                        "previous_appropriations": num(row.get("previous_appropriations")),
                        "project_total": num(row.get("project_total")),
                        
                        "horizon": len(offs),
                        **{self.name(k): (num(row[offs[k]]) if k in offs else None) for k in self.span},
                    }

                    full.append(rec)

        for file in files:
            getRange(file)
            
        self.span = [k for k in range(self.lo, self.hi + 1)]

        for file in files:
            accumulate(file)

        features = Features({
            "project_name": Value("string"),
            "project_id":   Value("string"),
            "cip_year":     Value("int32"),
            "department":   Value("string"),
            "project_type": Value("string"),
            "previous_appropriations": Value("int64"),
            "project_total": Value("int64"),
            "horizon": Value("int32"),
            **{self.name(k): Value("int64") for k in self.span},
        })

        self.ds = Dataset.from_list(full, features=features)

    def getDiff(self):
        ds_p = pd.DataFrame(self.ds)

        for pid, sub in ds_p[ds_p[f'{self.anchorColumn}'] != ''].groupby(f'{self.anchorColumn}', sort=False):
            if len(sub) <= 1:
                continue
            for col in ds_p.columns:
                vals = sub[col].tolist()
                if vals.count(vals[0]) != len(vals):
                    self.differences[pid].append((col, sorted(set(vals))))

    def getPairs(self):
        anchor = []
        positive = []

        for _,cols in self.differences.items():
            names = dict(cols).get('project_name')
            if not names:
                continue
            for a,b in itertools.combinations(names, 2):
                if fuzz.ratio(a,b) < 60:
                    continue
                anchor.append(a)
                positive.append(b)

        train_dataset = Dataset.from_dict({'anchor':anchor, 'positive':positive})
        return train_dataset
    def train(self, train_dataset, epochs=3, batch_size=32):
        if len(train_dataset) == 0:
            raise RuntimeError(
                "no training pairs — no project_id has a name that changed "
                "between books, so there is nothing to learn from")
        if len(train_dataset) < batch_size:
            batch_size = max(2, len(train_dataset))

        args = SentenceTransformerTrainingArguments(
            output_dir=os.path.join(self.train_dir, 'cip-model'),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=2e-5,
            warmup_steps=0.1,          # a float is read as a ratio; warmup_ratio
            save_strategy="epoch",     # is deprecated on Transformers 5
            logging_steps=10,
        )

        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset.shuffle(seed=0),
            loss=losses.MultipleNegativesRankingLoss(self.model),
        )
        trainer.train()

        final = os.path.join(self.train_dir, 'cip-model', 'final')
        self.model.save_pretrained(final)
        return final

    def run(self, identity_col="project_id", epochs=3):
        self.getData()
        self.getDiff(identity_col)
        return self.train(self.getPairs(), epochs=epochs)
    