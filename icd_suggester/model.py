import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from .data import CODES, synth_notes
from .phi import redact_phi


class IcdSuggester:
    """TF-IDF + logistic regression with per-prediction evidence terms."""

    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000, C=5.0)),
        ])
        self.metrics = {}

    def fit(self, texts=None, labels=None):
        if texts is None:
            texts, labels = synth_notes()
        X_tr, X_te, y_tr, y_te = train_test_split(texts, labels, test_size=0.2, stratify=labels, random_state=42)
        self.pipeline.fit(X_tr, y_tr)
        pred = self.pipeline.predict(X_te)
        self.metrics = {"accuracy": round(accuracy_score(y_te, pred), 4),
                        "macro_f1": round(f1_score(y_te, pred, average="macro"), 4),
                        "n_test": len(y_te)}
        return self

    def suggest(self, note: str, top_k: int = 3):
        clean, phi_found = redact_phi(note)
        tfidf, clf = self.pipeline.named_steps["tfidf"], self.pipeline.named_steps["clf"]
        vec = tfidf.transform([clean])
        probs = clf.predict_proba(vec)[0]
        terms = np.array(tfidf.get_feature_names_out())
        results = []
        for i in np.argsort(probs)[::-1][:top_k]:
            code = clf.classes_[i]
            contrib = vec.multiply(clf.coef_[i]).toarray()[0]
            evidence = [terms[j] for j in np.argsort(contrib)[::-1][:4] if contrib[j] > 0]
            results.append({"code": code, "description": CODES[code][0],
                            "confidence": round(float(probs[i]), 4), "evidence": evidence})
        return {"redacted_note": clean, "phi_removed": phi_found, "suggestions": results,
                "disclaimer": "Decision support only. A certified coder must review before billing."}

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
