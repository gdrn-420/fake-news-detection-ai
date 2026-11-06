"""
Streamlit interface for the educational Fake News Detection prototype.
Run: streamlit run app.py
"""
from pathlib import Path
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "fake_news_pipeline.joblib"

st.set_page_config(
    page_title="Fake News Detection AI",
    page_icon="📰",
    layout="wide"
)
st.title("📰 Fake News Detection")
st.write("Paste a headline or short news text to see the model's predicted writing-pattern label.")

if not MODEL_PATH.exists():
    st.warning("The trained model was not found. First run: python train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)
news_text = st.text_area("Enter news text", height=180, placeholder="Paste a headline or a short news paragraph here...")

if st.button("Analyse text", type="primary"):
    if len(news_text.strip()) < 15:
        st.error("Please enter at least 15 characters.")
    else:
        prediction = model.predict([news_text])[0]
        probabilities = model.predict_proba([news_text])[0]
        classes = list(model.classes_)
        confidence = float(probabilities[classes.index(prediction)]) * 100
        if prediction == "FAKE":
            st.error(f"Model label: {prediction}")
        else:
            st.success(f"Model label: {prediction}")
        st.metric("Model confidence", f"{confidence:.1f}%")
        st.caption("Confidence is the model's estimate for its learned labels, not proof that the story is true or false.")

st.divider()
st.subheader("Important limitations")
st.write(
    "- This classroom prototype was trained on fictional teaching examples, not a verified news archive.\n"
    "- It learns word patterns; it does not check facts, sources, dates, images or live websites.\n"
    "- A REAL label does not prove a story is true, and a FAKE label does not prove it is false.\n"
    "- Verify important claims with multiple reliable sources and official statements."
)
st.markdown("""
<style>
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: #080808 !important;
    color: #ffffff !important;
}

[data-testid="stHeader"] {
    background: transparent;
}

h1, h2, h3 {
    color: #FFD700 !important;
}

[data-testid="stMarkdownContainer"] p {
    color: #eeeeee;
}

[data-testid="stMetric"] {
    background: #201b0b;
    border: 1px solid #D4AF37;
    padding: 20px;
    border-radius: 15px;
}

.stTextArea textarea,
.stTextInput input {
    background: #171717 !important;
    color: white !important;
    border: 1px solid #D4AF37 !important;
}

.stButton button {
    background: linear-gradient(90deg, #B8860B, #FFD700);
    color: #000000 !important;
    border: none;
    border-radius: 12px;
    font-weight: bold;
}

.stButton button:hover {
    background: #FFE55C;
    color: black !important;
    border: 1px solid #ffffff;
}
</style>
""", unsafe_allow_html=True)
