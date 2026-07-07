# 🎯 SmartHire AI/ML — Resume-to-Job Matching Platform

An intelligent job recommendation system that matches resumes to job postings using machine learning. Upload a resume → get matching jobs, a predicted career category, a fit score, and a skill-gap report.

## ✨ Features

- **Resume Classification**: Automatically classify resumes into career categories
- **Job Recommendations**: Personalized job matches based on resume content using TF-IDF + cosine similarity
- **Fit Score Prediction**: Predict resume-to-job match probability
- **Skill Gap Analysis**: Identify matched and missing skills
- **Multi-Format Support**: Parse PDF, DOCX, and TXT resumes
- **Web Interface**: Easy-to-use Streamlit portal for instant results
- **Multi-Source Data**: Supports LinkedIn and Naukri job postings

## 🏗️ Core Components

1. **Resume Classifier** — Supervised ML (TF-IDF + Naive Bayes)
2. **Job Recommender** — Unsupervised (TF-IDF + cosine similarity)
3. **Fit Predictor** — Supervised classification model
4. **Skill-Gap Engine** — Job skills minus resume skills analysis

Optional: Topic clustering, salary prediction, trending skills analysis.

## 📋 Requirements

- **Python 3.10+** (developed on 3.12)
- **pip** or **conda** for package management
- Virtual environment (recommended)
- Optional: **Kaggle account** (to download real datasets)

## 🚀 Quick Start (3 Steps)

### Step 1: Setup Environment
```bash
cd "Smart_hire_AI_ML_June"

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Generate Dummy Models (for testing)
```bash
python create_dummy_models.py
```

This creates placeholder models so you can test the app immediately without waiting for data downloads and training.

### Step 3: Launch the App
```bash
streamlit run app/streamlit_app.py
```

Open browser → `http://localhost:8501` → Upload a resume → Get results! 🎉

---

## 📊 Full Setup (with Real Data & Models)

### Step 1: Download Datasets

Get your **Kaggle API key**:
1. Go to kaggle.com → Account Settings → API → Create New Token
2. Place `kaggle.json` in `~/.kaggle/` (Windows: `C:\Users\<username>\.kaggle\`)

Then run:
```bash
python download_data.py
```

Downloads:
- Resume dataset (Kaggle: `datatorque/resume-cv-data`)
- Naukri jobs (Kaggle: `PromptCloud/naukri-com-job-postings`)
- LinkedIn jobs (optional: `arshkon/linkedin-job-postings`)

See [`data/DATASETS.md`](data/DATASETS.md) for manual download alternatives.

### Step 2: Preprocess Data
```bash
python -m src.data.preprocess
```

Creates model-ready datasets:
- `data/interim/job_corpus.csv` — merged jobs
- `data/processed/resumes_clean.csv` — cleaned resumes
- `data/processed/jobs_clean.csv` — model-ready jobs

### Step 3: Train Models (Jupyter Notebooks)

Open notebooks in **VS Code** (install **Python** & **Jupyter** extensions):

1. **`notebooks/01_eda.ipynb`** — Exploratory data analysis
2. **`notebooks/02_resume_classifier.ipynb`** — Train resume classifier
3. **`notebooks/03_recommender.ipynb`** — Train job recommender
4. **`notebooks/04_clustering_topics.ipynb`** — Topic modeling & clustering
5. **`notebooks/05_fit_predictor.ipynb`** — Train fit prediction model

Run each notebook, save trained models to `models/` directory as `.pkl` files.

### Step 4: Run the App with Real Models
```bash
streamlit run app/streamlit_app.py
```

The app now uses your trained models instead of dummy ones!

---

## 🗂️ Project Structure

```
Smart_hire_AI_ML_June/
├── README.md                         # This file
├── requirements.txt                  # Dependencies
├── create_dummy_models.py            # Generate test models
├── download_data.py                  # Download datasets from Kaggle
│
├── data/                             # Datasets (git-ignored)
│   ├── DATASETS.md                   # Dataset documentation
│   ├── raw/                          # Original downloads
│   │   ├── linkedin/                 # LinkedIn job postings
│   │   ├── naukri/                   # Naukri job postings
│   │   └── resumes/                  # Resume files
│   ├── interim/                      # Merged/partially cleaned
│   └── processed/                    # Final model-ready data
│
├── notebooks/                        # Jupyter notebooks (run in order)
│   ├── 01_eda.ipynb                  # Exploratory Data Analysis
│   ├── 02_resume_classifier.ipynb    # Resume classification model
│   ├── 03_recommender.ipynb          # Job recommendation system
│   ├── 04_clustering_topics.ipynb    # Topic modeling & clustering
│   └── 05_fit_predictor.ipynb        # Fit prediction model
│
├── src/                              # Reusable Python modules
│   ├── __init__.py
│   ├── config.py                     # Paths & constants
│   ├── evaluate.py                   # Evaluation metrics
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py              # Load raw datasets
│   │   └── preprocess.py             # Clean & preprocess data
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── text_features.py          # Text vectorization
│   │   └── match_features.py         # Job-resume matching features
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classifier.py             # Resume classifier
│   │   ├── recommender.py            # Job recommender
│   │   ├── fit_predictor.py          # Fit score predictor
│   │   ├── clustering.py             # Topic clustering
│   │   └── dummy_models.py           # Test models
│   │
│   └── parsing/
│       ├── __init__.py
│       └── resume_parser.py          # PDF/DOCX/TXT parsing
│
├── models/                           # Trained .pkl models (git-ignored)
│   ├── classifier.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── job_tfidf_vectorizer.pkl
│   ├── job_vectors.pkl
│   ├── fit_predictor.pkl
│   └── recommender.pkl
│
├── app/
│   └── streamlit_app.py              # Web UI portal
│
├── reports/
│   └── figures/                      # Visualizations & plots
│
└── tests/
    └── test_features.py              # Unit tests

---

## 📈 Model Architecture

| Model | Type | Algorithm | Input | Output |
|-------|------|-----------|-------|--------|
| **Classifier** | Supervised | TF-IDF + Naive Bayes | Resume text | Career category + confidence |
| **Recommender** | Unsupervised | TF-IDF + Cosine Similarity | Resume text | Top-N jobs ranked |
| **Fit Predictor** | Supervised | Engineered features + LogReg | Resume + Job | Fit probability (0–1) |
| **Clustering** | Unsupervised | KMeans / LDA | Job descriptions | Topic assignments |

---






---

## 📚 Technology Stack

- **Data Processing**: pandas, scikit-learn
- **NLP**: TF-IDF, cosine similarity
- **ML Models**: Naive Bayes, Logistic Regression, KMeans
- **Web Framework**: Streamlit
- **Text Parsing**: pdfplumber, python-docx
- **Utilities**: numpy, pathlib

---

## 📝 Workflow

```
Raw Data → Preprocess → Feature Engineering → Model Training → App Portal
           (Step 2)     (Notebooks)          (Notebooks)       (Streamlit)
```

---

## 🎯 Next Steps

1. **Immediate**: Run `python create_dummy_models.py` → Test app
2. **Download Data**: Get real datasets via `python download_data.py`
3. **Preprocess**: Clean data with `python -m src.data.preprocess`
4. **Train Models**: Run notebooks 01→05 to train real models
5. **Evaluate**: Check model performance & iterate
6. **Deploy**: Push to cloud (Streamlit Cloud, Heroku, AWS, etc.)

---




