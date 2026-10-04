import unittest
from icd_suggester import IcdSuggester, redact_phi


class TestPHI(unittest.TestCase):
    def test_redacts_common_identifiers(self):
        text = "Patient: John Smith, SSN 123-45-6789, call 555-123-4567 on 03/14/2024, a@b.com"
        out, counts = redact_phi(text)
        for secret in ["John Smith", "123-45-6789", "555-123-4567", "03/14/2024", "a@b.com"]:
            self.assertNotIn(secret, out)
        self.assertEqual(set(counts), {"NAME", "SSN", "PHONE", "DATE", "EMAIL"})

    def test_clean_text_untouched(self):
        out, counts = redact_phi("Elevated blood pressure noted.")
        self.assertEqual(out, "Elevated blood pressure noted.")
        self.assertEqual(counts, {})


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = IcdSuggester().fit()

    def test_quality_floor(self):
        self.assertGreater(self.m.metrics["macro_f1"], 0.9)

    def test_known_note(self):
        r = self.m.suggest("Patient: Jane Roe seen 01/02/2024 for heartburn, acid reflux after meals, omeprazole.")
        self.assertEqual(r["suggestions"][0]["code"], "K21.9")
        self.assertNotIn("Jane Roe", r["redacted_note"])
        self.assertTrue(r["suggestions"][0]["evidence"])

    def test_top_k_and_confidence_sorted(self):
        s = self.m.suggest("wheezing, albuterol inhaler, shortness of breath at night", top_k=3)["suggestions"]
        self.assertEqual(len(s), 3)
        self.assertEqual(s[0]["code"], "J45.909")
        self.assertGreaterEqual(s[0]["confidence"], s[1]["confidence"])


class TestAPI(unittest.TestCase):
    def test_endpoint(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi/httpx not installed")
        from icd_suggester.api import app
        c = TestClient(app)
        self.assertEqual(c.get("/health").status_code, 200)
        r = c.post("/suggest", json={"note": "dysuria, urinary frequency, positive urinalysis nitrites"})
        self.assertEqual(r.json()["suggestions"][0]["code"], "N39.0")
        self.assertEqual(c.post("/suggest", json={"note": "short"}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
