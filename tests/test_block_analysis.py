import unittest
from unittest.mock import Mock

from features.nlp.block_analysis_handler import BlockAnalysisHandler
from features.nlp.humor_detection_handler import HumorDetectionHandler


class BlockAnalysisTests(unittest.TestCase):
    def test_block_analysis_handler(self):
        llm = Mock()
        llm.analyze_blocks.return_value = [
            {
                "topic_slug": "intro",
                "topic_title": "Introduction", 
                "topic_confidence": 0.8,
                "humor_score": 0.3,
                "humor_label": "light"
            },
            {
                "topic_slug": "main_content",
                "topic_title": "Main Content",
                "topic_confidence": 0.9, 
                "humor_score": 0.7,
                "humor_label": "clear"
            }
        ]

        handler = BlockAnalysisHandler(llm_client=llm, block_duration=30)
        context = {
            "transcript_segments": [
                {"start": 0, "end": 15, "text": "Hello everyone"},
                {"start": 15, "end": 45, "text": "This is funny content"}
            ],
            "duration_seconds": 60,
            "language": "en"
        }
        
        result = handler.handle(context)
        
        self.assertEqual(len(result["block_analysis"]), 2)
        self.assertEqual(len(result["topic_segments"]), 2)
        self.assertEqual(result["topic_segments"][0]["topic"], "intro")
        self.assertEqual(result["block_analysis"][1]["humor_score"], 0.7)

    def test_humor_projection(self):
        handler = HumorDetectionHandler(threshold=0.5)
        context = {
            "transcript_segments": [
                {"start": 5, "end": 10, "text": "segment 1"},
                {"start": 35, "end": 40, "text": "segment 2"}
            ],
            "block_analysis": [
                {"start": 0, "end": 30, "humor_score": 0.3},
                {"start": 30, "end": 60, "humor_score": 0.8}
            ]
        }
        
        result = handler.handle(context)
        
        self.assertEqual(len(result["humor_scores"]), 2)
        self.assertEqual(result["humor_scores"][0]["score"], 0.3)  # первый сегмент в первом блоке
        self.assertEqual(result["humor_scores"][1]["score"], 0.8)  # второй сегмент во втором блоке
        self.assertEqual(result["humor_summary"]["count_positive"], 1)  # только один >= 0.5

    def test_humor_projection_fallback(self):
        handler = HumorDetectionHandler()
        context = {
            "transcript_segments": [{"start": 0, "end": 5, "text": "test"}],
            "block_analysis": []  # пустой block_analysis
        }
        
        result = handler.handle(context)
        
        self.assertEqual(len(result["humor_scores"]), 1)
        self.assertEqual(result["humor_scores"][0]["score"], 0.0)  # fallback


if __name__ == "__main__":
    unittest.main()