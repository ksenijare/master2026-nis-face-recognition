import csv
import os

IMAGES_DIR = "images"
MATCH_CSV = "matchpairsDevTrain.csv"
MISMATCH_CSV = "mismatchpairsDevTrain.csv"
OUTPUT_CSV = "pairsTrain.csv"

def construct_path(person_name: str, img_num: str) -> str:
    """Kreira punu putanju sa 4-cifrenim brojem."""
    try:
        img_num_int = int(img_num)
    except ValueError:
        return None
    img_num_str = f"{img_num_int:04d}"  # 0001, 0017 ...
    filename = f"{person_name}_{img_num_str}.jpg"
    full_path = os.path.join(IMAGES_DIR, person_name, filename)
    if not os.path.isfile(full_path):
        print(f"Warning: File not found: {full_path}")
        return None
    return full_path

def process_match_csv(input_file: str):
    rows = []
    with open(input_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            person = r["name"]
            img1 = construct_path(person, r["imagenum1"])
            img2 = construct_path(person, r["imagenum2"])
            if img1 and img2:
                rows.append({"img1_path": img1, "img2_path": img2, "label": 1})
    return rows

def process_mismatch_csv(input_file: str):
    rows = []
    with open(input_file, "r", newline="", encoding="utf-8") as f:
        first_line = f.readline().strip()
        columns = first_line.split(",")
        f.seek(0)

        if "name1" not in columns:
            fieldnames = ["name1","imagenum1","name2","imagenum2"]
            reader = csv.DictReader(f, fieldnames=fieldnames)
            next(reader)  # preskoči originalni prvi red
        else:
            reader = csv.DictReader(f)

        for r in reader:
            img1 = construct_path(r["name1"], r["imagenum1"])
            img2 = construct_path(r["name2"], r["imagenum2"])
            if img1 and img2:
                rows.append({"img1_path": img1, "img2_path": img2, "label": 0})
    return rows

def main():
    all_rows = []
    all_rows += process_match_csv(MATCH_CSV)
    all_rows += process_mismatch_csv(MISMATCH_CSV)

    if not all_rows:
        print("No valid pairs found.")
        return

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["img1_path","img2_path","label"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"Successfully wrote {len(all_rows)} pairs to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()