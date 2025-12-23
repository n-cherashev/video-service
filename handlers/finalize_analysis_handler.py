from __future__ import annotations

from typing import Any, Dict

from handlers.base_handler import BaseHandler
from handlers.timeseries import safe_mean_points


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
            "chapters": context.get("chapters", []),
        }

        context["analysis_result"] = analysis_result
        print("✓ Analysis finalized")
        return context

    def _extract_video_info(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "duration_seconds": float(context.get("duration_seconds", 0.0) or 0.0),
            "fps": float(context.get("fps", 0.0) or 0.0),
            "file_size_bytes": int(context.get("video_size_bytes", 0) or 0),
            "processing_time_seconds": float(context.get("processing_time_seconds", 0.0) or 0.0),
        }

    def _extract_motion_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"heatmap": context.get("motion_heatmap", []), "summary": context.get("motion_summary", {})}

    def _extract_audio_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        audio_features = context.get("audio_features", {}) or {}
        return {
            "features": audio_features,
            "meta": context.get("audio_features_meta", {}),
            "summary": {
                "mean_loudness": safe_mean_points(audio_features.get("loudness", [])),
                "mean_energy": safe_mean_points(audio_features.get("energy", [])),
                "mean_speech_probability": safe_mean_points(audio_features.get("speech_probability", [])),
            },
        }

    def _extract_stt_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        segments = context.get("transcript_segments", []) or []
        total_speech_duration = 0.0
        for seg in segments:
            if isinstance(seg, dict) and "start" in seg and "end" in seg:
                try:
                    d = float(seg["end"]) - float(seg["start"])
                except (TypeError, ValueError):
                    continue
                if d > 0:
                    total_speech_duration += d

        full = str(context.get("full_transcript", "") or "")
        return {
            "segments": segments,
            "full_transcript": full,
            "summary": {
                "total_segments": int(len(segments)),
                "total_characters": int(len(full)),
                "total_speech_duration": float(total_speech_duration),
            },
        }

    def _extract_sentiment_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"timeline": context.get("sentiment_timeline", []), "summary": context.get("sentiment_summary", {})}

    def _extract_humor_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"scores": context.get("humor_scores", []), "summary": context.get("humor_summary", {})}

    def _extract_topics_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        segments = context.get("topic_segments", []) or []
        unique_topics = {seg.get("topic") for seg in segments if isinstance(seg, dict) and "topic" in seg}
        unique_topics.discard(None)
        return {
            "segments": segments,
            "summary": {"total_segments": int(len(segments)), "unique_topics": int(len(unique_topics))},
        }
