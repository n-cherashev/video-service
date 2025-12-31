import unittest
from unittest.mock import Mock

from features.nlp.humor_detection_handler import HumorDetectionHandler
from features.nlp.topic_segmentation_handler import TopicSegmentationHandler
from llm.ollama_client import _normalize_language, _normalize_slug


class LLMHandlerTests(unittest.TestCase):
    def test_humor_empty_segments(self):
        llm = Mock()
        llm.score_humor.return_value = 0.7

        h = HumorDetectionHandler(llm_client=llm, min_chars=5)
        ctx = {"transcript_segments": [{"start": 0, "end": 1, "text": ""}], "language": "Russian"}
        out = h.handle(ctx)

        self.assertEqual(out["humor_scores"], [])
        self.assertEqual(out["humor_summary"]["mean"], 0.0)

    def test_humor_llm_error_fallback(self):
        llm = Mock()
        llm.score_humor.side_effect = Exception("LLM failed")

        h = HumorDetectionHandler(llm_client=llm, min_chars=5)
        ctx = {"transcript_segments": [{"start": 0, "end": 1, "text": "hello world"}], "language": "en"}
        out = h.handle(ctx)

        self.assertEqual(len(out["humor_scores"]), 1)
        self.assertEqual(out["humor_scores"][0]["score"], 0.0)

    def test_topics_length_normalization(self):
        llm = Mock()
        llm.summarize_topics.return_value = ["a"]  # меньше, чем блоков

        h = TopicSegmentationHandler(llm_client=llm, block_duration=10, min_block_chars=0)
        ctx = {
            "duration_seconds": 25,
            "language": "ru",
            "transcript_segments": [{"start": 0, "end": 5, "text": "hello"}],
        }
        out = h.handle(ctx)

        self.assertEqual(len(out["topic_segments"]), 3)
        self.assertEqual(out["topic_segments"][0]["topic"], "a")
        self.assertEqual(out["topic_segments"][1]["topic"], "general")
        self.assertEqual(out["topic_segments"][2]["topic"], "general")

    def test_topics_llm_error_fallback(self):
        llm = Mock()
        llm.summarize_topics.side_effect = Exception("LLM failed")

        h = TopicSegmentationHandler(llm_client=llm, block_duration=10, min_block_chars=0)
        ctx = {
            "duration_seconds": 15,
            "language": "en",
            "transcript_segments": [{"start": 0, "end": 5, "text": "hello world"}],
        }
        out = h.handle(ctx)

        # Должно быть 2 блока (0-10, 10-15), все с fallback "general"
        self.assertEqual(len(out["topic_segments"]), 2)
        self.assertTrue(all(seg["topic"] == "general" for seg in out["topic_segments"]))

    def test_normalize_language(self):
        self.assertEqual(_normalize_language("Russian"), "ru")
        self.assertEqual(_normalize_language("English"), "en")
        self.assertEqual(_normalize_language("ru"), "ru")
        self.assertEqual(_normalize_language("en"), "en")
        self.assertEqual(_normalize_language("french"), "french")
        self.assertIsNone(_normalize_language(""))
        self.assertIsNone(_normalize_language(None))

    def test_normalize_slug(self):
        self.assertEqual(_normalize_slug("Hello World"), "hello_world")
        self.assertEqual(_normalize_slug("test-slug"), "test_slug")
        self.assertEqual(_normalize_slug(""), "general")
        self.assertEqual(_normalize_slug("русский_текст"), "русский_текст")
        self.assertEqual(_normalize_slug("test@#$%slug"), "testslug")


if __name__ == "__main__":
    unittest.main()