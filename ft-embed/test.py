from sentence_transformers import SentenceTransformer
import os
import csv
import numpy as np

TRAIN_DIR = r'C:\Users\vince\Documents\GitHub\CIPBD-Embedding\ft-embed\training'

CITY = 'Dallas'
CSV_DIR = rf'C:\Users\vince\Documents\GitHub\CIPBD\{CITY}\CSV\\'
OUT_DIR = rf'C:\Users\vince\Documents\GitHub\CIPBD-Embedding\Outputs\{CITY}\\'

model = SentenceTransformer(rf"{TRAIN_DIR}\cip-model\final")
def vectorize():
    
    names = []
    for file in os.listdir(CSV_DIR):
        file_dir = os.path.join(CSV_DIR,file)
        with open(file_dir, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            for row in reader:
                if row['project_name'] != '' and row['project_name'] not in names:
                    names.append(row['project_name'])
                    
    vecs  = model.encode(names, normalize_embeddings=True)
    np.save(os.path.join(OUT_DIR,'names.npy'), np.array(names), allow_pickle=False)
    np.save(os.path.join(OUT_DIR,'vecs.npy'), np.array(vecs), allow_pickle=False)

    return vecs, names

#vectorize()

vecs, names = np.load(os.path.join(OUT_DIR,'vecs.npy')), np.load(os.path.join(OUT_DIR,'names.npy'))
print(names)
k = 10 # k best embeddings
emb_mat = np.stack([emb_1, emb_2, ..., emb_100000])
scores = np.dot(emb_mat, embedding_new)
best_k_ind = np.argpartition(scores, k)[-k:] 
top_k_emb = emb_mat[best_k_ind]   