import os
import time
import random
import numpy as np
from PIL import Image
from face_pipeline import analyze_face

IMAGES_DIR = "images"
OUTPUT_FILE = "all_embeddings.npy"
MAX_IMAGES = 500

embeddings = {}
start_time = time.time()
count = 0
skipped = 0

#Rekurzivno skupljanje svih slika u folderu i podfolderima
all_images = []
for root, dirs, files in os.walk(IMAGES_DIR):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            all_images.append(os.path.join(root, file))
        else:
            skipped += 1
            print(f"Skipping (not an image): {os.path.join(root, file)}")

#Uzmi random subset za ubrzavanje računanja
all_images = random.sample(all_images, min(MAX_IMAGES, len(all_images)))
total_images = len(all_images)
print(f"Total images to process: {total_images}\n")

#Obrada svake slike
for idx, img_path in enumerate(all_images, start=1):
    try:
        img = np.array(Image.open(img_path).convert("RGB"))

        res = analyze_face(
            img,
            model_name="Facenet",
            detector_backend="mtcnn",
            enforce_detection=True
        )

        embeddings[img_path] = res.embedding
        count += 1

    except PermissionError:
        skipped += 1
        print(f"Skipping (permission denied): {img_path}")
        continue
    except Exception as e:
        skipped += 1
        print(f"Skipping {img_path}: {e}")
        continue

    #Logging svakih 10 slika
    if idx % 10 == 0 or idx == total_images:
        elapsed = time.time() - start_time
        speed = count / elapsed if elapsed > 0 else 0
        progress = (idx / total_images) * 100
        print(f"[{idx}/{total_images} | {progress:.1f}%] "
              f"Processed: {count} | Skipped: {skipped} | Speed: {speed:.2f} img/sec")

#Čuvanje embeddings
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)  # obriši stari fajl ako postoji

np.save(OUTPUT_FILE, embeddings)

#Final stats
total_time = time.time() - start_time
print("\n======================")
print("DONE")
print("======================")
print(f"Total images: {total_images}")
print(f"Processed: {count}")
print(f"Skipped: {skipped}")
print(f"Total time: {total_time:.2f} sec")
print(f"Average speed: {count / total_time:.2f} img/sec")
print(f"Saved to: {OUTPUT_FILE}")
