# STEP 2: Train the text classifier from data/train.csv
# data/train.csv must have two columns:  text,label
# label must be fake or real (use news HEADLINES, because photocards are short).

import os
import csv
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

TRAIN_CSV = os.path.join("data", "train.csv")
MODEL_PATH = os.path.join("models", "classifier.joblib")

texts = []
labels = []
f = open(TRAIN_CSV, "r", encoding="utf-8-sig", newline="")
reader = csv.DictReader(f)
for row in reader:
    text = (row.get("text") or "").strip()
    label = (row.get("label") or "").strip().lower()
    if text == "":
        continue
    if label != "fake" and label != "real":
        continue
    texts.append(text)
    labels.append(label)
f.close()

fake_count = 0
real_count = 0
for label in labels:
    if label == "fake":
        fake_count = fake_count + 1
    else:
        real_count = real_count + 1
print("Rows loaded:", len(texts), "| fake:", fake_count, "| real:", real_count)

if len(texts) < 100:
    print("WARNING: very few rows. This is only OK for testing the code.")

x_train, x_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels)

# Character n-grams cope better with OCR spelling noise than whole words.
model = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=60000)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
model.fit(x_train, y_train)

predictions = model.predict(x_test)
print(classification_report(y_test, predictions))

os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)
print("Saved model to", MODEL_PATH)