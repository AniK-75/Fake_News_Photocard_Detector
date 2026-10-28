# STEP 3: Smoke test. Reads every image in dataset/metadata.csv, runs OCR,
# predicts fake/real, and writes results.csv.
# - OCR text is saved in ocr_text.csv, so if you stop, run again and it RESUMES.
# - Predictions are recomputed every run (so you can retrain and re-run quickly).

import os
import csv
import joblib

METADATA_CSV = os.path.join("dataset", "metadata.csv")
IMAGE_FOLDER = os.path.join("dataset", "images")
OCR_CSV = "ocr_text.csv"
RESULTS_CSV = "results.csv"
MODEL_PATH = os.path.join("models", "classifier.joblib")
MIN_CONFIDENCE = 0.4
LIMIT = 0        # 10 = quick test.  Change to 0 to run ALL images.


def read_rows(path):
    rows = []
    if not os.path.exists(path):
        return rows
    f = open(path, "r", encoding="utf-8-sig", newline="")
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)
    f.close()
    return rows


def fake_probability(model, text):
    fake_index = 0
    classes = list(model.classes_)
    for i in range(len(classes)):
        if classes[i] == "fake":
            fake_index = i
    probabilities = model.predict_proba([text])[0]
    return float(probabilities[fake_index])


meta_rows = read_rows(METADATA_CSV)
if len(meta_rows) == 0:
    print("dataset/metadata.csv not found or empty. Run prepare_dataset.py first.")
    raise SystemExit

if LIMIT > 0:
    meta_rows = meta_rows[:LIMIT]
print("Images to process:", len(meta_rows))

# ---------- Part 1: OCR (resumable) ----------
done_text = {}
old_rows = read_rows(OCR_CSV)
for row in old_rows:
    done_text[row["filename"]] = row["ocr_text"]

todo = []
for row in meta_rows:
    if row["filename"] not in done_text:
        todo.append(row)
print("Already OCR'd:", len(done_text), "| still to do:", len(todo))

if len(todo) > 0:
    import easyocr
    print("Loading OCR (first time downloads Bangla + English models)...")
    reader = easyocr.Reader(["bn", "en"], gpu=False)

    new_file = not os.path.exists(OCR_CSV)
    f = open(OCR_CSV, "a", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(f, fieldnames=["filename", "ocr_text"])
    if new_file:
        writer.writeheader()

    number = 0
    for row in todo:
        number = number + 1
        path = os.path.join(IMAGE_FOLDER, row["filename"])
        pieces = []
        try:
            results = reader.readtext(path)
            for item in results:
                piece_text = item[1]
                piece_conf = item[2]
                if piece_conf >= MIN_CONFIDENCE:
                    pieces.append(piece_text)
        except Exception as error:
            print("OCR failed for", row["filename"], error)
        text = " ".join(pieces)
        writer.writerow({"filename": row["filename"], "ocr_text": text})
        f.flush()
        done_text[row["filename"]] = text
        print(number, "/", len(todo), row["filename"])
    f.close()

# ---------- Part 2: predict ----------
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    print("No model found (run train_classifier.py). Saving OCR only.")

out = open(RESULTS_CSV, "w", encoding="utf-8-sig", newline="")
writer = csv.DictWriter(out, fieldnames=["filename", "label", "ocr_text", "fake_probability", "prediction"])
writer.writeheader()

fake_fake = 0
fake_real = 0
real_fake = 0
real_real = 0
no_text = 0

for row in meta_rows:
    text = done_text.get(row["filename"], "")
    probability = ""
    prediction = ""
    if text.strip() == "":
        prediction = "no_text"
        no_text = no_text + 1
    elif model is not None:
        value = fake_probability(model, text)
        probability = round(value, 4)
        if value >= 0.5:
            prediction = "fake"
        else:
            prediction = "real"
        if row["label"] == "fake" and prediction == "fake":
            fake_fake = fake_fake + 1
        if row["label"] == "fake" and prediction == "real":
            fake_real = fake_real + 1
        if row["label"] == "real" and prediction == "fake":
            real_fake = real_fake + 1
        if row["label"] == "real" and prediction == "real":
            real_real = real_real + 1
    writer.writerow({"filename": row["filename"], "label": row["label"], "ocr_text": text,
                     "fake_probability": probability, "prediction": prediction})
out.close()

print("")
print("Saved", RESULTS_CSV)
print("Images with no readable text:", no_text)
if model is not None:
    total = fake_fake + fake_real + real_fake + real_real
    print("Confusion counts (actual -> predicted):")
    print("  fake -> fake:", fake_fake, "   fake -> real:", fake_real)
    print("  real -> fake:", real_fake, "   real -> real:", real_real)
    if total > 0:
        print("Accuracy (just for information):", round((fake_fake + real_real) / total, 3))