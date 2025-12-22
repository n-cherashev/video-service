from __future__ import annotations

from typing import Any, Dict

from handlers.base_handler import BaseHandler


class FinalizeAnalysisHandler(BaseHandler):
    """Собирает финальный результат анализа."""

    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        analysis_result = {
            "video_info": self._extract_video_info(context),
            "motion": self._extract_motion_data(context),
            "audio": self._extract_audio_data(context),
            "speech_to_text": self._extract_stt_data(context),
            "sentiment": self._extract_sentiment_data(context),
            "humor": self._extract_humor_data(context),
            "topics": self._extract_topics_data(context),
            "audio_events": context.get("audio_events", []),
            "scene_boundaries": context.get("scene_boundaries", []),
            "timeline": context.get("timeline", []),
            "highlights": context.get("highlights", []),
            "chapters": context.get("chapters", [])
        }
        
        context["analysis_result"] = analysis_result
        
        print("✓ Analysis finalized")
        return context

    def _extract_video_info(self, context: Dict) -> Dict:
        """Извлекает основную информацию о видео."""
        return {
            "duration_seconds": context.get("duration_seconds", 0),
            "fps": context.get("fps", 0),
            "file_size_bytes": context.get("file_size", 0),
            "processing_time_seconds": context.get("processing_time_seconds", 0)
        }

    def _extract_motion_data(self, context: Dict) -> Dict:
        """Извлекает данные о движении."""
        return {
            "heatmap": context.get("motion_heatmap", []),
            "summary": context.get("motion_summary", {})
        }

    def _extract_audio_data(self, context: Dict) -> Dict:
        """Извлекает аудио данные."""
        audio_features = context.get("audio_features", {})
        return {
            "features": audio_features,
            "summary": {
                "mean_loudness": self._safe_mean(audio_features.get("loudness", [])),
                "mean_energy": self._safe_mean(audio_features.get("energy", [])),
                "mean_speech_probability": self._safe_mean(audio_features.get("speech_probability", []))
            }
        }

    def _extract_stt_data(self, context: Dict) -> Dict:
        """Извлекает данные распознавания речи."""
        segments = context.get("transcript_segments", [])
        
        # Безопасное вычисление общей длительности речи
        total_speech_duration = 0.0
        for seg in segments:
            if isinstance(seg, dict) and "start" in seg and "end" in seg:
                try:
                    duration = float(seg["end"]) - float(seg["start"])
                    if duration > 0:
                        total_speech_duration += duration
                except (ValueError, TypeError):
                    continue
        
        return {
            "segments": segments,
            "full_transcript": context.get("full_transcript", ""),
            "summary": {
                "total_segments": len(segments),
                "total_characters": len(context.get("full_transcript", "")),
                "total_speech_duration": total_speech_duration
            }
        }

    def _extract_sentiment_data(self, context: Dict) -> Dict:
        """Извлекает данные сентимент-анализа."""
        return {
            "timeline": context.get("sentiment_timeline", []),
            "summary": context.get("sentiment_summary", {})
        }

    def _extract_humor_data(self, context: Dict) -> Dict:
        """Извлекает данные анализа юмора."""
        return {
            "scores": context.get("humor_scores", []),
            "summary": context.get("humor_summary", {})
        }

    def _extract_topics_data(self, context: Dict) -> Dict:
        """Извлекает данные сегментации по топикам."""
        segments = context.get("topic_segments", [])
        
        # Безопасное извлечение уникальных топиков
        unique_topics = set()
        for seg in segments:
            if isinstance(seg, dict) and "topic" in seg:
                unique_topics.add(seg["topic"])
        
        return {
            "segments": segments,
            "summary": {
                "total_segments": len(segments),
                "unique_topics": len(unique_topics)
            }
        }

    def _safe_mean(self, values: list) -> float:
        """Безопасно вычисляет среднее значение."""
        if not values:
            return 0.0
        
        numeric_values = []
        for val in values:
            if isinstance(val, (int, float)):
                numeric_values.append(float(val))
            elif hasattr(val, '__float__'):
                try:
                    numeric_values.append(float(val))
                except (ValueError, TypeError):
                    continue
        
        return sum(numeric_values) / len(numeric_values) if numeric_values else 0.0