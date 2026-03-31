import csv
import argparse
import numpy as np
from PIL import Image
import os

from face_pipeline import analyze_face, cosine_similarity

def load_rgb(csv_path: str) -> np.ndarray:
    if csv_path is None or csv_path == "" or str(csv_path).lower() == "nan":
        return None

    csv_path = str(csv_path).replace("\\", os.sep).replace("/", os.sep)

    if not os.path.isfile(csv_path):
        print(f"Warning: File not found: {csv_path}")
        return None

    return np.array(Image.open(csv_path).convert("RGB"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="CSV: img1_path,img2_path,label")
    ap.add_argument("--model", default="ArcFace")
    ap.add_argument("--detector", default="retinaface")
    ap.add_argument("--min_thr", type=float, default=0.20)
    ap.add_argument("--max_thr", type=float, default=0.90)
    ap.add_argument("--step", type=float, default=0.01)
    args = ap.parse_args()

    rows = []
    with open(args.pairs, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    sims, labels = [], []

    print(f"Loaded pairs: {len(rows)}")
    for i, r in enumerate(rows, start=1):
        img1 = load_rgb(r["img1_path"])
        img2 = load_rgb(r["img2_path"])
        if img1 is None or img2 is None:
            print(f"Skipping pair {i} because one image is missing")
            continue

        res1 = analyze_face(img1, model_name=args.model, detector_backend=args.detector, enforce_detection=True)
        res2 = analyze_face(img2, model_name=args.model, detector_backend=args.detector, enforce_detection=True)

        sim = cosine_similarity(res1.embedding, res2.embedding)
        sims.append(sim)
        labels.append(int(r["label"]))

        print(f"{i:03d}: sim={sim:.4f} label={r['label']}")

    sims = np.array(sims, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    best_thr, best_acc = None, -1.0
    thr = args.min_thr
    while thr <= args.max_thr + 1e-9:
        preds = (sims >= thr).astype(np.int32)
        acc = float((preds == labels).mean())
        if acc > best_acc:
            best_acc, best_thr = acc, thr
        thr += args.step

    if best_thr is not None:
        print(f"\nBest threshold: {best_thr:.2f}   accuracy: {best_acc:.3f}")
    else:
        print("No valid pairs found. Cannot compute threshold.")

if __name__ == "__main__":
    main()