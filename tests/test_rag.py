import io
import json
import os
import unittest
import urllib.error
from unittest import mock

from icd_suggester import IcdSuggester
from icd_suggester.rag import RagExplainer, Retriever

RAW = "Patient: Jane Roe, 01/02/2024, phone 555-123-4567. Heartburn, acid reflux after meals, omeprazole."


def fake_response(text):
    body = json.dumps({"choices": [{"message": {"content": text}}]}).encode()
    cm = mock.MagicMock()
    cm.__enter__.return_value = io.BytesIO(body)
    return cm


class TestRag(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = IcdSuggester().fit()
        cls.rag = RagExplainer()

    def setUp(self):
        os.environ.pop("OPENAI_API_KEY", None)

    def _run(self, note=RAW):
        r = self.model.suggest(note)
        return r, self.rag.analyse(r["redacted_note"], r["suggestions"])

    def test_retrieval_finds_right_passage(self):
        top = Retriever().retrieve("heartburn and acid reflux, started omeprazole")[0]
        self.assertEqual(top["code"], "K21.9")

    def test_offline_template_mode(self):
        r, out = self._run()
        self.assertEqual(out["mode"], "template")
        self.assertIn("K21.9", out["explanation"])
        self.assertFalse(out["needs_review"])
        self.assertTrue(out["checked_suggestions"][0]["retrieval_agrees"])

    def test_gibberish_needs_review(self):
        _, out = self._run("zzzz qqqq xxxx wwww")
        self.assertTrue(out["needs_review"])

    @mock.patch("icd_suggester.rag.urllib.request.urlopen")
    def test_llm_mode_sends_only_redacted_text(self, urlopen):
        os.environ["OPENAI_API_KEY"] = "test-key"
        urlopen.return_value = fake_response("K21.9 fits because of heartburn and reflux.")
        _, out = self._run()
        self.assertEqual(out["mode"], "llm")
        req = urlopen.call_args[0][0]
        sent = req.data.decode()
        for secret in ["Jane Roe", "555-123-4567", "01/02/2024"]:
            self.assertNotIn(secret, sent)
        self.assertEqual(req.get_header("Authorization"), "Bearer test-key")

    @mock.patch("icd_suggester.rag.urllib.request.urlopen")
    def test_guardrail_rejects_invented_code(self, urlopen):
        os.environ["OPENAI_API_KEY"] = "test-key"
        urlopen.return_value = fake_response("This is clearly Z99.9.")
        _, out = self._run()
        self.assertEqual(out["mode"], "template_fallback")
        self.assertIn("K21.9", out["explanation"])

    @mock.patch("icd_suggester.rag.urllib.request.urlopen", side_effect=urllib.error.URLError("down"))
    def test_llm_failure_falls_back(self, _):
        os.environ["OPENAI_API_KEY"] = "test-key"
        _, out = self._run()
        self.assertEqual(out["mode"], "template_fallback")


if __name__ == "__main__":
    unittest.main()
