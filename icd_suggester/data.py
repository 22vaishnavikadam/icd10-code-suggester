"""Small curated ICD-10-CM code set + phrase banks used to synthesise training notes.

Synthetic data keeps the repo free of real PHI. Swap in de-identified notes
(e.g. MIMIC-IV with a DUA) for a real model.
"""
import random

CODES = {
    "E11.9": ("Type 2 diabetes mellitus without complications",
              ["type 2 diabetes", "elevated hba1c", "metformin", "polyuria and polydipsia", "fasting glucose high", "diabetic diet counselling"]),
    "I10": ("Essential (primary) hypertension",
            ["elevated blood pressure", "hypertension", "lisinopril", "bp 150/95", "headache with high bp", "low sodium diet"]),
    "J45.909": ("Unspecified asthma, uncomplicated",
                ["wheezing", "asthma", "albuterol inhaler", "shortness of breath at night", "peak flow reduced", "triggered by cold air"]),
    "J18.9": ("Pneumonia, unspecified organism",
              ["productive cough", "fever and chills", "chest x-ray infiltrate", "crackles on auscultation", "pneumonia", "azithromycin"]),
    "N39.0": ("Urinary tract infection, site not specified",
              ["dysuria", "urinary frequency", "positive urinalysis nitrites", "suprapubic pain", "urinary tract infection", "nitrofurantoin"]),
    "M54.50": ("Low back pain, unspecified",
               ["low back pain", "lumbar stiffness", "pain radiating to buttock", "muscle spasm", "ibuprofen and physiotherapy", "pain on bending"]),
    "J02.9": ("Acute pharyngitis, unspecified",
              ["sore throat", "pharyngeal erythema", "painful swallowing", "rapid strep negative", "acute pharyngitis", "throat lozenges"]),
    "K21.9": ("Gastro-esophageal reflux disease without esophagitis",
              ["heartburn", "acid reflux after meals", "gerd", "omeprazole", "regurgitation", "worse when lying down"]),
    "E78.5": ("Hyperlipidemia, unspecified",
              ["high ldl cholesterol", "hyperlipidemia", "atorvastatin", "elevated triglycerides", "lipid panel abnormal", "dietary fat reduction"]),
    "F32.9": ("Major depressive disorder, single episode, unspecified",
              ["persistent low mood", "loss of interest", "depression", "sertraline", "poor sleep and fatigue", "phq-9 elevated"]),
    "I48.91": ("Unspecified atrial fibrillation",
               ["irregular heartbeat", "atrial fibrillation", "palpitations", "apixaban", "ecg irregularly irregular", "rate control with metoprolol"]),
    "J44.9": ("Chronic obstructive pulmonary disease, unspecified",
              ["chronic cough", "copd", "long smoking history", "tiotropium", "dyspnea on exertion", "reduced fev1"]),
    "G43.909": ("Migraine, unspecified, not intractable",
                ["throbbing unilateral headache", "migraine", "photophobia and nausea", "sumatriptan", "visual aura", "relieved by dark quiet room"]),
    "R07.9": ("Chest pain, unspecified",
              ["chest pain", "chest tightness", "ecg normal", "troponin negative", "pain not related to exertion", "observation advised"]),
}

_OPENERS = ["Patient presents with", "Seen today for", "Complaints include", "Follow-up visit for", "History notable for"]
_FILLERS = ["Vitals stable.", "No known drug allergies.", "Advised follow-up in 2 weeks.", "Exam otherwise unremarkable.",
            "Denies recent travel.", "Plan discussed with patient."]


def synth_notes(n_per_code: int = 60, seed: int = 7):
    rng = random.Random(seed)
    texts, labels = [], []
    for code, (_, phrases) in CODES.items():
        for _ in range(n_per_code):
            picked = rng.sample(phrases, k=rng.randint(2, 4))
            note = f"{rng.choice(_OPENERS)} {', '.join(picked)}. {' '.join(rng.sample(_FILLERS, 2))}"
            texts.append(note)
            labels.append(code)
    return texts, labels
