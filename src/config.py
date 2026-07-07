"""Paths and constants. Import from here instead of hard-coding filenames."""

from pathlib import Path

# ---- Folders ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# ---- Raw dataset paths (what you downloaded into data/raw/) ----
RESUME_CSV = RAW_DIR / "UpdatedResumeDataSet.csv"
NAUKRI_CSV = RAW_DIR / "naukri_com-job_sample.csv"

# Optional LinkedIn dataset (arshkon/linkedin-job-postings), unzipped into
# data/raw/linkedin/. Merged in only if these files exist.
LINKEDIN_DIR = RAW_DIR / "linkedin"
LINKEDIN_POSTINGS_CSV = LINKEDIN_DIR / "postings.csv"
LINKEDIN_JOB_SKILLS_CSV = LINKEDIN_DIR / "jobs" / "job_skills.csv"
LINKEDIN_SKILLS_MAP_CSV = LINKEDIN_DIR / "mappings" / "skills.csv"

# ---- Outputs (created by preprocess.py) ----
JOB_CORPUS_CSV = INTERIM_DIR / "job_corpus.csv"          # merged jobs (readable)
RESUMES_CLEAN_CSV = PROCESSED_DIR / "resumes_clean.csv"  # cleaned resumes
JOBS_CLEAN_CSV = PROCESSED_DIR / "jobs_clean.csv"        # model-ready jobs (+ text col)

# ---- Model files ----
CLASSIFIER_PATH = MODELS_DIR / "classifier.pkl"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
RECOMMENDER_PATH = MODELS_DIR / "recommender.pkl"
JOB_TFIDF_VECTORIZER_PATH = MODELS_DIR / "job_tfidf_vectorizer.pkl"
JOB_VECTORS_PATH = MODELS_DIR / "job_vectors.pkl"
FIT_PREDICTOR_PATH = MODELS_DIR / "fit_predictor.pkl"
JOB_CORPUS_PATH = JOB_CORPUS_CSV

# ---- Common schema every job source is mapped to before merging ----
COMMON_JOB_COLUMNS = ["title", "company", "location", "skills", "description", "experience"]

# ---- Constants ----
RANDOM_STATE = 42   # seed for reproducible train/test splits
TOP_N = 5           # jobs returned by the recommender

# Generic words to exclude from skill extraction
GENERIC_WORDS = {
    'and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'can', 'will', 'would', 'could', 'should', 'must', 'may', 'might',
    'experience', 'work', 'working', 'job', 'role', 'position', 'company',
    'team', 'project', 'development', 'software', 'application', 'system',
    'management', 'manager', 'lead', 'senior', 'junior', 'years', 'strong'
}
