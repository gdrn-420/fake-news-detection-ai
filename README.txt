FAKE NEWS DETECTION — CLASS XII AI PROJECT (843)

FILES
- data/news_samples.csv: 120 invented teaching examples (60 REAL-style, 60 FAKE-style).
- generate_data.py: recreates the fictional CSV dataset.
- train_model.py: trains TF-IDF + Logistic Regression and prints evaluation metrics.
- app.py: Streamlit interface.
- models/fake_news_pipeline.joblib: trained model included for convenience.
- requirements.txt: Python dependencies.

RUN
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Run: pip install -r requirements.txt
4. Run: python train_model.py
5. Run: streamlit run app.py

IMPORTANT
The examples are fictional and their labels are illustrative teaching labels. This prototype learns
language patterns from those examples; it is NOT a reliable real-world fact checker. Do not use it
to make decisions about people or to share accusations. For a stronger project, replace the sample
CSV with a properly sourced, licensed, fact-checked dataset, document its source and limitations,
and test on a separate representative dataset. Never invent interview feedback or test results.
