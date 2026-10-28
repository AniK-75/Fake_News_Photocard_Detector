# Streamlit website: Bangla Fake News Photocard Detector
# Run with:  .\venv\Scripts\python.exe -m streamlit run app.py

import os
import csv
import numpy as np
import requests
import joblib
import streamlit as st
from PIL import Image

MODEL_PATH = os.path.join("models", "classifier.joblib")
RESULTS_CSV = "results.csv"
IMAGE_FOLDER = os.path.join("dataset", "images")
FALSE_WORDS = ["false", "fake", "misleading", "incorrect", "mislead",
               "মিথ্যা", "ভুল", "বিভ্রান্তিকর", "ভুয়া"]

st.set_page_config(page_title="Bangla Fake News Photocard Detector", layout="wide")


@st.cache_resource
def load_reader():
    import easyocr
    return easyocr.Reader(["bn", "en"], gpu=False)


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def run_ocr(pil_image, min_confidence):
    reader = load_reader()
    array = np.array(pil_image.convert("RGB"))
    results = reader.readtext(array)
    pieces = []
    for item in results:
        if item[2] >= min_confidence:
            pieces.append(item[1])
    return " ".join(pieces)


def get_fake_probability(model, text):
    fake_index = 0
    classes = list(model.classes_)
    for i in range(len(classes)):
        if classes[i] == "fake":
            fake_index = i
    probabilities = model.predict_proba([text])[0]
    return float(probabilities[fake_index])


def search_fact_checks(text, api_key):
    matches = []
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": text[:200], "languageCode": "bn", "key": api_key}
    try:
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
    except Exception as error:
        return matches, "Fact Check search failed: " + str(error)
    claims = data.get("claims", [])
    for claim in claims:
        reviews = claim.get("claimReview", [])
        for review in reviews:
            rating = review.get("textualRating", "")
            is_false = False
            for word in FALSE_WORDS:
                if word in rating.lower():
                    is_false = True
            matches.append({
                "claim": claim.get("text", ""),
                "publisher": review.get("publisher", {}).get("name", ""),
                "rating": rating,
                "url": review.get("url", ""),
                "is_false": is_false,
            })
    return matches, ""


def make_verdict(probability, matches):
    for match in matches:
        if match["is_false"]:
            return "Likely Fake", "A fact-checker rated a similar claim as false."
    if probability is None:
        return "Uncertain", "No classifier is loaded and no fact-check match was found."
    if probability >= 0.75 and len(matches) == 0:
        return "Likely Fake", "The text looks like fake news (high probability), no fact-check found."
    if probability <= 0.25:
        return "Probably Real (low confidence)", "The text looks like real news. This is not proof."
    return "Uncertain", "The model is not sure. Check trusted sources manually."


st.title("Bangla Fake News Photocard Detector")
st.caption("Prototype for a Machine Learning Lab project. The result is a hint, not proof.")

tab_check, tab_smoke = st.tabs(["Check a photocard", "Smoke test results (400 cards)"])

# ------------------------------------------------------------------ TAB 1
with tab_check:
    model = load_model()
    st.sidebar.header("Settings")
    min_conf = st.sidebar.slider("OCR confidence filter", 0.0, 1.0, 0.4, 0.05)
    api_key = st.sidebar.text_input("Google Fact Check API key (optional)",
                                    value=os.environ.get("FACTCHECK_API_KEY", ""),
                                    type="password")
    if model is None:
        st.sidebar.warning("No classifier found. Run train_classifier.py first.")

    uploaded = st.file_uploader("Upload a news photocard", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is not None:
        image = Image.open(uploaded)
        left, right = st.columns(2)
        with left:
            st.image(image, caption="Uploaded photocard", use_container_width=True)
        with right:
            with st.spinner("Reading text from the image (OCR)..."):
                ocr_text = run_ocr(image, min_conf)
            st.subheader("1. Text found (you can fix mistakes)")
            text = st.text_area("OCR text", value=ocr_text, height=150)

            probability = None
            if model is not None and text.strip() != "":
                probability = get_fake_probability(model, text)
                st.subheader("2. Classifier")
                st.write("Fake probability:", round(probability, 3))
                st.progress(min(max(probability, 0.0), 1.0))

            matches = []
            if api_key.strip() != "" and text.strip() != "":
                st.subheader("3. Fact-check search")
                matches, error_message = search_fact_checks(text, api_key.strip())
                if error_message != "":
                    st.warning(error_message)
                elif len(matches) == 0:
                    st.write("No matching fact-check found.")
                for match in matches:
                    st.write("- **" + match["publisher"] + "**: " + match["rating"]
                             + " — " + match["claim"])
                    st.write(match["url"])

            st.subheader("Verdict")
            verdict, reason = make_verdict(probability, matches)
            st.markdown("### " + verdict)
            st.write(reason)

# ------------------------------------------------------------------ TAB 2
with tab_smoke:
    if not os.path.exists(RESULTS_CSV):
        st.info("results.csv not found. Run run_batch.py first.")
    else:
        rows = []
        f = open(RESULTS_CSV, "r", encoding="utf-8-sig", newline="")
        for row in csv.DictReader(f):
            rows.append(row)
        f.close()

        fake_fake = 0
        fake_real = 0
        real_fake = 0
        real_real = 0
        no_text = 0
        actual_fake = 0
        actual_real = 0
        for row in rows:
            if row["label"] == "fake":
                actual_fake = actual_fake + 1
            else:
                actual_real = actual_real + 1
            if row["prediction"] == "no_text":
                no_text = no_text + 1
            if row["label"] == "fake" and row["prediction"] == "fake":
                fake_fake = fake_fake + 1
            if row["label"] == "fake" and row["prediction"] == "real":
                fake_real = fake_real + 1
            if row["label"] == "real" and row["prediction"] == "fake":
                real_fake = real_fake + 1
            if row["label"] == "real" and row["prediction"] == "real":
                real_real = real_real + 1

        st.write("Total photocards:", len(rows), "| fake:", actual_fake,
                 "| real:", actual_real, "| no readable text:", no_text)
        st.write("This is a prototype smoke test. The model was trained on article "
                 "headlines, not on photocards, so accuracy is only for information.")

        st.subheader("Confusion table (rows = actual, columns = predicted)")
        st.table({"predicted fake": {"actual fake": fake_fake, "actual real": real_fake},
                  "predicted real": {"actual fake": fake_real, "actual real": real_real}})

        scored = fake_fake + fake_real + real_fake + real_real
        if scored > 0:
            st.write("Accuracy:", round((fake_fake + real_real) / scored, 3))

        st.subheader("Look at one photocard")
        names = []
        for row in rows:
            names.append(row["filename"])
        chosen = st.selectbox("Choose a file", names)
        for row in rows:
            if row["filename"] == chosen:
                image_path = os.path.join(IMAGE_FOLDER, chosen)
                col_a, col_b = st.columns(2)
                with col_a:
                    if os.path.exists(image_path):
                        st.image(Image.open(image_path), use_container_width=True)
                with col_b:
                    st.write("Actual label:", row["label"])
                    st.write("Predicted:", row["prediction"])
                    st.write("Fake probability:", row["fake_probability"])
                    st.text_area("OCR text", value=row["ocr_text"], height=150)

        st.subheader("All results")
        st.dataframe(rows)