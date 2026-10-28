# STEP 1: Take your raw downloaded photocards (in two folders: fake and real),
# give each one a neutral name (pc_0001.jpg ...), shuffle them, remove
# duplicates / broken images, and write dataset/metadata.csv with the labels.
#
# Why neutral names? If the file name says "fake", a model could cheat.
# The label lives only in metadata.csv.

import os
import csv
import shutil
import random
import hashlib
from PIL import Image

RAW_FOLDER = "raw_photocards"          # contains the folders: fake and real
OUT_IMAGES = os.path.join("dataset", "images")
OUT_CSV = os.path.join("dataset", "metadata.csv")
LABELS = ["fake", "real"]
ALLOWED = [".jpg", ".jpeg", ".png", ".webp"]


def file_hash(path):
    md5 = hashlib.md5()
    f = open(path, "rb")
    while True:
        chunk = f.read(65536)
        if not chunk:
            break
        md5.update(chunk)
    f.close()
    return md5.hexdigest()


os.makedirs(OUT_IMAGES, exist_ok=True)

items = []
seen_hashes = {}
bad_count = 0
dup_count = 0

for label in LABELS:
    folder = os.path.join(RAW_FOLDER, label)
    if not os.path.isdir(folder):
        print("MISSING FOLDER:", folder)
        continue
    names = sorted(os.listdir(folder))
    for name in names:
        extension = os.path.splitext(name)[1].lower()
        if extension not in ALLOWED:
            continue
        source_path = os.path.join(folder, name)

        # 1) is the image readable?
        try:
            img = Image.open(source_path)
            img.verify()
        except Exception:
            print("Unreadable, skipped:", source_path)
            bad_count = bad_count + 1
            continue

        # 2) is it an exact duplicate of an earlier file?
        digest = file_hash(source_path)
        if digest in seen_hashes:
            print("Duplicate, skipped:", source_path, "(same as", seen_hashes[digest] + ")")
            dup_count = dup_count + 1
            continue
        seen_hashes[digest] = source_path

        items.append({"label": label, "source_path": source_path,
                      "original": name, "extension": extension})

# Shuffle so the numbering does not reveal the label
random.seed(42)
random.shuffle(items)

rows = []
counter = 1
for item in items:
    new_name = "pc_" + str(counter).zfill(4) + item["extension"]
    shutil.copy2(item["source_path"], os.path.join(OUT_IMAGES, new_name))
    rows.append({"id": "pc_" + str(counter).zfill(4), "filename": new_name,
                 "label": item["label"], "original_filename": item["original"]})
    counter = counter + 1

f = open(OUT_CSV, "w", encoding="utf-8-sig", newline="")
writer = csv.DictWriter(f, fieldnames=["id", "filename", "label", "original_filename"])
writer.writeheader()
for row in rows:
    writer.writerow(row)
f.close()

fake_total = 0
real_total = 0
for row in rows:
    if row["label"] == "fake":
        fake_total = fake_total + 1
    else:
        real_total = real_total + 1

print("")
print("Done.")
print("Fake images:", fake_total)
print("Real images:", real_total)
print("Broken skipped:", bad_count)
print("Duplicates skipped:", dup_count)
print("Saved:", OUT_CSV)