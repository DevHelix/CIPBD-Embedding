import csv
import os
from datasets import Dataset, load_from_disk
from datasets import Features, Value, Sequence
import pandas as pd
import pickle
from collections import defaultdict
import itertools
from rapidfuzz import fuzz

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses
)

CITY = 'Dallas'
CSV_DIR = rf'C:\Users\vince\Documents\GitHub\CIPBD\{CITY}\CSV\\'
TRAIN_DIR = r'C:\Users\vince\Documents\GitHub\CIPBD-Embedding\ft-embed\training'

class pair:
    def __init__(self):

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.guide = SentenceTransformer("all-MiniLM-L6-v2")
        
    def pull(self):
        full = []

        PLAN = range(0, 7)                       # widest horizon in the corpus

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

        def t(file):
            file_dir = os.path.join(CSV_DIR,file)
            with open(file_dir, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
            
                header = reader.fieldnames
                years = []

                for cn in header:
                    if 'year_' in cn:
                        years.append(cn)

                for row in reader:
                    cip = int(row["cip_year"])
                    year_cols = {f"plan_{k}": num(row.get(f"year_{cip + k}")) for k in PLAN}

                    a = {
                        "project_name": txt(row.get("project_name")),
                        "project_id":   txt(row.get("project_id")),
                        "cip_year":     cip,
                        "department":   txt(row.get("department")),
                        "project_type": txt(row.get("project_type")),
                        "previous_appropriations": num(row.get("previous_appropriations")),
                        "project_total": num(row.get("project_total")),
                        **year_cols,
                    }

                    full.append(a)

        for file in os.listdir(CSV_DIR):
            t(file)

        features = Features({
            "project_name": Value("string"),
            "project_id":   Value("string"),
            "cip_year":     Value("int32"),
            "department":   Value("string"),
            "project_type": Value("string"),
            "previous_appropriations": Value("int64"),
            "project_total": Value("int64"),
            **{f"plan_{k}": Value("int64") for k in PLAN},
        })

        ds = Dataset.from_list(full, features=features)
        ds.save_to_disk(rf'{TRAIN_DIR}\\temp_ds')

    def section(self):
        ds = load_from_disk(rf'{TRAIN_DIR}\\temp_ds')
        ds_p = pd.DataFrame(ds)

        differences = defaultdict(list)
        counts = ds_p['project_id'].value_counts()
        for pid, count in counts.items():
            if not pid or count <= 1:
                continue
            sub = ds_p[ds_p['project_id'] == pid]   
            for col in ds_p.columns:
                vals = sub[col].tolist()
                if vals.count(vals[0]) != len(vals):
                    differences[pid].append((col, sorted(set(vals))))

        #print(differences) # {id : (column_name,[value of column of row w/ same id])}
        pickle.dump(differences, open(rf'{TRAIN_DIR}\\differences.pkl', "wb"))

    def form_data(self):
        anchor = []
        positive = []

        diff = pickle.load(open(rf'{TRAIN_DIR}\\differences.pkl', "rb"))
        for id,cols in diff.items():
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

    def run(self):
        self.pull()
        self.section()
        train_dataset = self.form_data()

        loss = losses.CachedGISTEmbedLoss(
            self.model,
            self.guide,
            mini_batch_size=64,
            margin_strategy="absolute",   # or "relative" (e.g., margin=0.05 for max. 95% of positive similarity)
            margin=0.1
        )
        args = SentenceTransformerTrainingArguments(
            output_dir=rf'{TRAIN_DIR}/cip-model',
            num_train_epochs=3,
            per_device_train_batch_size=32,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            save_strategy="epoch",
            logging_steps=10,
        )

        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            loss=losses.MultipleNegativesRankingLoss(self.model),
        )

        trainer.train()
        self.model.save_pretrained(rf'{TRAIN_DIR}/cip-model/final')
