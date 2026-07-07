"""
Temporary script to create dummy models for testing.
Replace these with real trained models from your notebooks later.
"""

import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import sys

# Add src to path to import dummy models
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from models.dummy_models import DummyRecommender, DummyFitPredictor, DummyClassifier

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
MODELS_DIR.mkdir(exist_ok=True)
INTERIM_DIR.mkdir(exist_ok=True)

print("Creating dummy models...")

# ---- Create dummy TF-IDF Vectorizer for Resume Classification ----
tfidf = TfidfVectorizer(max_features=100, stop_words='english')
dummy_texts = [
    "python developer software engineer",
    "data scientist machine learning",
    "product manager business analyst",
]
tfidf.fit(dummy_texts)

with open(MODELS_DIR / "tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
print("✓ Created tfidf_vectorizer.pkl")

# ---- Create dummy Classifier ----
X_dummy = tfidf.transform(dummy_texts)
y_dummy = np.array(['Software Engineer', 'Data Scientist', 'Product Manager'])
clf = MultinomialNB()
clf.fit(X_dummy, y_dummy)

with open(MODELS_DIR / "classifier.pkl", "wb") as f:
    pickle.dump(clf, f)
print("✓ Created classifier.pkl")

# ---- Create dummy Job TF-IDF Vectorizer ----
job_tfidf = TfidfVectorizer(max_features=200, stop_words='english')
dummy_job_descriptions = [
    "python django rest api backend development",
    "machine learning tensorflow pytorch data science",
    "product strategy business analysis requirements",
]
job_tfidf.fit(dummy_job_descriptions)

with open(MODELS_DIR / "job_tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(job_tfidf, f)
print("✓ Created job_tfidf_vectorizer.pkl")

# ---- Create dummy Job Vectors ----
job_vectors = job_tfidf.transform(dummy_job_descriptions).toarray()
with open(MODELS_DIR / "job_vectors.pkl", "wb") as f:
    pickle.dump(job_vectors, f)
print("✓ Created job_vectors.pkl")

# ---- Create dummy Job Corpus CSV ----
job_corpus_df = pd.DataFrame({
    "title": ["Backend Developer", "Data Scientist", "Product Manager"],
    "company": ["Tech Corp", "AI Inc", "StartupXYZ"],
    "location": ["San Francisco", "New York", "Remote"],
    "skills": ["Python, Django, REST API", "Python, TensorFlow, ML", "Strategy, Analytics"],
    "description": dummy_job_descriptions,
    "experience": ["3-5 years", "2-4 years", "5+ years"],
    "source": ["linkedin", "naukri", "linkedin"]
})
job_corpus_df.to_csv(INTERIM_DIR / "job_corpus.csv", index=False)
print("✓ Created job_corpus.csv")

# ---- Create dummy Recommender ----
recommender = DummyRecommender()
with open(MODELS_DIR / "recommender.pkl", "wb") as f:
    pickle.dump(recommender, f)
print("✓ Created recommender.pkl")

# ---- Create dummy Fit Predictor ----
predictor = DummyClassifier()
with open(MODELS_DIR / "fit_predictor.pkl", "wb") as f:
    pickle.dump(predictor, f)
print("✓ Created fit_predictor.pkl")

print("\n✓ All dummy models created!")
print("⚠️  IMPORTANT: These are placeholder models. Run your training notebooks to create real models.")
