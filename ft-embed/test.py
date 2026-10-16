from sentence_transformers import SentenceTransformer
import os
import csv
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

TRAIN_DIR = r'C:\Users\vince\Documents\GitHub\CIPBD-Embedding\ft-embed\training'

CITY = 'Dallas'
CSV_DIR = rf'C:\Users\vince\Documents\GitHub\CIPBD\{CITY}\CSV\\' # input directory here
OUT_DIR = rf'C:\Users\vince\Documents\GitHub\CIPBD-Embedding\Outputs\{CITY}\\'

model = SentenceTransformer(rf"{TRAIN_DIR}\cip-model\final")
def vectorize(): # produces vecs.npy file
    
    rows = []
    names = []
    fieldnames = []
    for file in os.listdir(CSV_DIR):
        file_dir = os.path.join(CSV_DIR,file)
        with open(file_dir, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                if row['project_name'] != '':
                    rows.append(row)
                    names.append(row['project_name'])
                    
    vecs  = model.encode(names, normalize_embeddings=True)

    np.save(os.path.join(OUT_DIR,'vecs.npy'), np.array(vecs), allow_pickle=False)
    with open(os.path.join(OUT_DIR, 'rows.csv'),mode='w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval='', extrasaction='ignore')
        
        writer.writeheader()  # Writes the header row
        writer.writerows(rows)  # Writes all data rows

    return vecs

#vectorize()

vecs = np.load(os.path.join(OUT_DIR,'vecs.npy'))
vecs = np.load(os.path.join(OUT_DIR, 'vecs.npy'))

with open(os.path.join(OUT_DIR, 'rows.csv'), mode='r', newline='', encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))

names = [r['project_name'] for r in rows]
years = [r['cip_year'] for r in rows]
pid   = [r.get('project_id', '') for r in rows]

def nearest():
    nn = NearestNeighbors(metric='cosine').fit(vecs)
    dists, idxs = nn.radius_neighbors(vecs, radius=1 - 0.80)   # cosine distance

    pairs = []
    for a, (ds, js) in enumerate(zip(dists, idxs)):
        for dist, b in zip(ds, js):
            if a < b:                                   # each pair once, skip self
                pairs.append((names[a], names[b], 1 - dist))

    pairs.sort(key=lambda t: -t[2])
    print(pairs[:10])

def compare(pn, k=7):
    print(f"Input: {pn}")
    q = model.encode(pn, normalize_embeddings=True)
    sims = vecs @ q                                
    order = np.argsort(-sims)[:k]
    return [
            rows[i] for i in order
        ]

count = defaultdict(int)
# sim = compare("Flood Management Area -  Major Maintenance (Bar X, Baypoint ")
# for row in sim:
#     count[row['project_id']] += 1

