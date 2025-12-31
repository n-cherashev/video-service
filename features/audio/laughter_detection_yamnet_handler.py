from typing import Any, Dict
from core.base_handler import BaseHandler
from ml.yamnet_client import YAMNetClient


class LaughterDetectionYamnetHandler(BaseHandler):
    """Детектор смеха/аплодисментов через YAMNet.

    Сниженный threshold для лучшего обнаружения в русском контенте.
    """

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        self.yamnet_client = YAMNetClient()

    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("[YAMNet] LaughterDetectionYamnetHandler")

        audio_path = context.get("audio_path")
        if not audio_path:
            print("✓ Laughter detection: no audio path")
            context["laughter_timeline"] = []
            context["laughter_summary"] = {"max_prob": 0.0, "mean_prob": 0.0, "count_positive": 0, "threshold": self.threshold}
            return context

        try:
            result = self.yamnet_client.predict_laughter(audio_path)

            context["laughter_timeline"] = result["timeline"]
            context["laughter_summary"] = result["summary"]
            context["laughter_summary"]["threshold"] = self.threshold

            count_positive = sum(1 for item in result["timeline"] if item["prob"] >= self.threshold)
            print(f"✓ Laughter detection: {len(result['timeline'])} seconds, {count_positive} with laughter")

        except Exception as e:
            print(f"Laughter detection failed: {e}")
            context["laughter_timeline"] = []
            context["laughter_summary"] = {"max_prob": 0.0, "mean_prob": 0.0, "count_positive": 0, "threshold": self.threshold}

        return context
