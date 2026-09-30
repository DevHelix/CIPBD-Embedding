from sentence_transformers import SentenceTransformer
import os
import csv
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

TRAIN_DIR = r'C:\Users\vince\Documents\GitHub\CIPBD-Embedding\ft-embed\training'

CITY = 'Dallas'
CSV_DIR = rf'C:\Users\vince\Documents\GitHub\CIPBD\{CITY}\CSV\\'
OUT_DIR = rf'C:\Users\vince\Documents\GitHub\CIPBD-Embedding\Outputs\{CITY}\\'

model = SentenceTransformer(rf"{TRAIN_DIR}\cip-model\final")
def vectorize():
    
    names = []
    years = []
    for file in os.listdir(CSV_DIR):
        file_dir = os.path.join(CSV_DIR,file)
        with open(file_dir, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            for row in reader:
                if row['project_name'] != '':
                    names.append(row['project_name'])
                    years.append(row['cip_year'])
                    
    vecs  = model.encode(names, normalize_embeddings=True)

    np.save(os.path.join(OUT_DIR,'names.npy'), np.array(names), allow_pickle=False)
    np.save(os.path.join(OUT_DIR,'vecs.npy'), np.array(vecs), allow_pickle=False)
    np.save(os.path.join(OUT_DIR,'years.npy'), np.array(years), allow_pickle=False)

    return vecs, names, years

vectorize()

vecs, names, years = np.load(os.path.join(OUT_DIR,'vecs.npy')), np.load(os.path.join(OUT_DIR,'names.npy')), np.load(os.path.join(OUT_DIR,'years.npy'))

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

def compare(pn, year, k=7):
    print(f"Input: {pn}")
    q = model.encode(pn, normalize_embeddings=True)
    sims = vecs @ q                                
    order = np.argsort(-sims)[:k]
    return [(f'{pn} | {year}', f'{str(names[i])} | {years[i]}', round(float(sims[i]),4)) for i in order]

def te(pn):
    q = model.encode(pn, normalize_embeddings=True)
    for i,v in enumerate(names):
        
        if "Flood Management Area" in v:
            print(v)
            print(years[i])

#te("Flood Management Area -  Major Maintenance (Bar X, Baypoint ")

sim = compare("Flood Management Area -  Major Maintenance (Bar X, Baypoint ", 2017)
for i in sim:
    print(i)