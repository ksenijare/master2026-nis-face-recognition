import csv
import numpy as np
from face_pipeline import cosine_similarity
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", required=True, help="CSV sa parovima")
ap.add_argument("--embeddings", default="all_embeddings.npy", help="Precomputed embeddings")
ap.add_argument("--min_thr", type=float, default=0.2)
ap.add_argument("--max_thr", type=float, default=0.9)
ap.add_argument("--step", type=float, default=0.01)
args = ap.parse_args()

# Učitaj embedding-e
embeddings = np.load(args.embeddings, allow_pickle=True).item()
print(f"Loaded {len(embeddings)} embeddings")

# Učitaj parove
rows = []
with open(args.pairs, "r") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

sims, labels = [], []

for i, r in enumerate(rows, start=1):
    emb1 = embeddings.get(r["img1_path"])
    emb2 = embeddings.get(r["img2_path"])
    if emb1 is None or emb2 is None:
        print(f"Skipping pair {i} because embedding missing")
        continue
    sim = cosine_similarity(emb1, emb2)
    sims.append(sim)
    labels.append(int(r["label"]))

sims = np.array(sims)
labels = np.array(labels)

# Best threshold
best_thr, best_acc = None, -1
thr = args.min_thr
while thr <= args.max_thr + 1e-9:
    preds = (sims >= thr).astype(int)
    acc = float((preds == labels).mean())
    if acc > best_acc:
        best_acc, best_thr = acc, thr
    thr += args.step

if best_thr is not None:
    print(f"\nBest threshold: {best_thr:.2f}   accuracy: {best_acc:.3f}")
else:
    print("No valid pairs found.")