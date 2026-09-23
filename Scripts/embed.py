# OpenAI embedding model

from openai import OpenAI
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc
import os
from dotenv import load_dotenv

CITY = "San-Diego"

IN_PATH = rf"C:\Users\vince\Documents\GitHub\CIPBD\{CITY}\CSV\\"
OUT_PATH = rf"C:\Users\vince\Documents\GitHub\CIPBD-Embedding\Outputs\{CITY}\dump.npy"

load_dotenv()
API_KEY = os.getenv('API_KEY')
client = OpenAI(api_key = API_KEY)

vecs = []
names = []
years = []

def embed(year):
    with open(f'{IN_PATH}{year}.csv', newline='', encoding='utf-8-sig') as f:
        new = [row['project_name'] for row in csv.DictReader(f)]

    for i in range(0, len(new), 1000):
        chunk = new[i:i + 1000]
        response = client.embeddings.create(
            input=chunk, model="text-embedding-3-small", dimensions=512
        )
        vecs.extend(d.embedding for d in response.data)

    names.extend(new)
    years.extend([year] * len(new))

def compare(a, b):
    vecs  = np.load(OUT_PATH, mmap_mode='r')
    names = list(np.load(OUT_PATH.replace('.npy', '_names.npy')))
    idx   = {n: i for i, n in enumerate(names)}

    i, j = idx[a], idx[b]
    sim = float(np.asarray(vecs[i]) @ np.asarray(vecs[j]))
    print(f"{names[i]!r} vs {names[j]!r}: {sim:.4f}")

def cluster():
    vecs = np.load(OUT_PATH, mmap_mode='r')
    pca = PCA(n_components = 2)
    _principal = pca.fit_transform(vecs)
    _principal = pd.DataFrame(_principal)

    plt.figure(figsize =(8, 8))
    plt.title('Visualising the data')
    Dendrogram = shc.dendrogram((shc.linkage(_principal, method ='average')))
    plt.show()

fc = len(os.listdir(IN_PATH))

for year in range(2026-fc,2026):
    print(year)
    embed(year)

assert len(vecs) == len(names) == len(years), (len(vecs), len(names), len(years))
np.save(OUT_PATH, np.array(vecs), allow_pickle=False)
np.save(OUT_PATH.replace('.npy', '_names.npy'), np.array(names), allow_pickle=False)

print("Embedding Completed")

