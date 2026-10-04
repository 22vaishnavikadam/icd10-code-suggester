"""Retrieval-augmented explanation layer.

Flow: redacted note -> retrieve guideline passages -> cross-check with the classifier ->
explain (LLM if OPENAI_API_KEY is set, otherwise a deterministic template).

HIPAA notes:
* Only the REDACTED note is ever sent to an LLM, and only when a key is configured.
* In production use a vendor with a signed BAA (e.g. Azure OpenAI) and add audit logging.
* LLM output is validated: it may only mention codes from the retrieved/candidate set.
"""
import json
import os
import re
import urllib.error
import urllib.request

from sklearn.feature_extraction.text import TfidfVectorizer

from .kb import KB

_CODE_RE = re.compile(r"\b[A-Z]\d{2}(?:\.\w{1,4})?\b")

_SYSTEM = (
    "You are a medical coding assistant. Use ONLY the candidate codes and guideline "
    "passages provided. Cite code numbers. Never invent codes. In 3-4 sentences explain "
    "which candidate best fits the note and why, and say so if documentation is insufficient."
)


class Retriever:
    def __init__(self, kb=KB):
        self.kb = kb
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.matrix = self.vec.fit_transform([d["text"] for d in kb])

    def retrieve(self, text: str, k: int = 3):
        sims = (self.matrix @ self.vec.transform([text]).T).toarray().ravel()
        order = sims.argsort()[::-1][:k]
        return [
            {"code": self.kb[i]["code"], "description": self.kb[i]["description"],
             "passage": self.kb[i]["note"], "score": round(float(sims[i]), 4)}
            for i in order if sims[i] > 0
        ]


class RagExplainer:
    def __init__(self, retriever=None, timeout=20):
        self.retriever = retriever or Retriever()
        self.timeout = timeout

    # read config at call time so tests / deployments can change env vars
    @property
    def api_key(self):
        return os.getenv("OPENAI_API_KEY")

    @property
    def llm_enabled(self):
        return bool(self.api_key)

    def analyse(self, redacted_note: str, suggestions: list, review_threshold: float = 0.6):
        passages = self.retriever.retrieve(redacted_note, k=3)
        retrieved = [p["code"] for p in passages]
        checked = [{**s, "code": str(s["code"]), "retrieval_agrees": str(s["code"]) in retrieved}
                   for s in suggestions]
        top = checked[0] if checked else None
        needs_review = (top is None or top["confidence"] < review_threshold
                        or not top["retrieval_agrees"])
        explanation, mode = self._explain(redacted_note, checked, passages)
        return {"passages": passages, "checked_suggestions": checked,
                "needs_review": needs_review, "explanation": explanation, "mode": mode}

    # ---- explanation -------------------------------------------------
    def _template(self, checked, passages):
        if not checked:
            return "No candidate codes were produced."
        top = checked[0]
        note = next((p["passage"] for p in passages if p["code"] == top["code"]), None)
        parts = [f"Top candidate is {top['code']} ({top['description']}) "
                 f"with {top['confidence']:.0%} model confidence."]
        if top["evidence"]:
            parts.append(f"Supporting terms in the note: {', '.join(top['evidence'])}.")
        parts.append(f"Guideline: {note}" if note else
                     "No guideline passage matched this code, so manual review is advised.")
        return " ".join(parts)

    def _explain(self, note, checked, passages):
        if not self.llm_enabled:
            return self._template(checked, passages), "template"
        try:
            text = self._call_llm(note, checked, passages)
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError, OSError):
            return self._template(checked, passages), "template_fallback"
        allowed = {c["code"] for c in checked} | {p["code"] for p in passages}
        for found in _CODE_RE.findall(text):
            if not any(a.startswith(found) for a in allowed):
                return self._template(checked, passages), "template_fallback"
        return text, "llm"

    def _call_llm(self, note, checked, passages):
        cands = "\n".join(f"- {c['code']}: {c['description']} (confidence {c['confidence']:.2f})"
                          for c in checked)
        refs = "\n".join(f"[{p['code']}] {p['passage']}" for p in passages) or "(none retrieved)"
        payload = {
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "temperature": 0, "max_tokens": 250,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"Note (PHI redacted):\n{note}\n\nCandidates:\n{cands}\n\nGuidelines:\n{refs}"},
            ],
        }
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        req = urllib.request.Request(
            f"{base}/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"].strip()
