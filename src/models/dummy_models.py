"""Dummy model classes for testing before real models are trained."""

import numpy as np
from sklearn.base import BaseEstimator


class DummyRecommender:
    def __init__(self):
        self.jobs = [
            {"title": "Python Developer", "company": "Tech Corp", "skills": ["Python", "Django"]},
            {"title": "Data Scientist", "company": "AI Inc", "skills": ["Python", "ML"]},
        ]
    
    def recommend(self, resume_text, top_n=5):
        return self.jobs[:top_n]


class DummyClassifier:
    """Dummy classifier that mimics sklearn's interface."""
    def predict(self, X):
        return np.array([0.75] * len(X))
    
    def predict_proba(self, X):
        # Return dummy probabilities: [[negative_prob, positive_prob], ...]
        n_samples = len(X) if hasattr(X, '__len__') else 1
        probs = np.random.rand(n_samples, 2)
        probs = probs / probs.sum(axis=1, keepdims=True)  # Normalize to sum to 1
        return probs


class DummyFitPredictor:
    """Dummy fit predictor that mimics sklearn's classifier interface."""
    def predict(self, X):
        n_samples = len(X) if hasattr(X, '__len__') else 1
        return np.array([1] * n_samples)  # Dummy prediction
    
    def predict_proba(self, X):
        # Return dummy probabilities: [[negative_prob, positive_prob], ...]
        n_samples = len(X) if hasattr(X, '__len__') else 1
        probs = np.array([[0.3, 0.7]] * n_samples)  # 70% fit score
        return probs

