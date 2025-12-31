"""
FinalizeAnalysisHandler - собирает финальный результат анализа.

Формирует:
- AnalysisResultPublic - для API ответа
- AnalysisResultFull - для сохранения на диск
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar, Dict, FrozenSet, List
import uuid

from core.base_handler import AnalyzerHandler
from models.keys import Key
from models.results import (
    AnalysisResultPublic,
    AnalysisResultFull,
    AnalysisSummary,
    ChapterResult,
    NodeTimingResult,
    ScoreBreakdown,
    TimelinePointResult,
    TranscriptSegment,
    ViralClipResult,
)
from utils.timeseries import safe_mean_points


class FinalizeAnalysisHandler(AnalyzerHandler):
    """Собирает финальный результат анализа.

    Формирует два типа результатов:
    - public: компактный для API
    - full: полный для сохранения

    Provides:
    - ANALYSIS_RESULT: dict с public и full результатами
    """

    requires: ClassVar[FrozenSet[Key]] = frozenset()
    provides: ClassVar[FrozenSet[Key]] = frozenset({Key.ANALYSIS_RESULT})

    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("[15] FinalizeAnalysisHandler")

        # Собираем данные
        task_id = context.get("task_id") or str(uuid.uuid4())[:8]
        video_name = context.get("video_name", "video")
        duration = float(context.get("duration_seconds", 0.0) or 0.0)
        processing_time = float(context.get("processing_time_seconds", 0.0) or 0.0)

        # Viral clips
        viral_clips = self._extract_viral_clips(context)

        # Chapters
        chapters = self._extract_chapters(context)

        # Summary
        summary = self._extract_summary(context)

        # Public result
        public = AnalysisResultPublic(
            task_id=task_id,
            video_name=video_name,
            duration_seconds=duration,
            processing_time_seconds=processing_time,
            viral_clips=viral_clips,
            chapters=chapters,
            summary=summary,
            created_at=datetime.utcnow(),
        )

        # Full result with additional data
        full = self._build_full_result(public, context)

        # Legacy format for backward compatibility
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
            # New typed results
            "public": public.to_dict(),
            "full": full.to_dict(),
        }

        context["analysis_result"] = analysis_result
        context["analysis_result_public"] = public
        context["analysis_result_full"] = full

        print(f"✓ Analysis finalized: {len(viral_clips)} clips, {len(chapters)} chapters")
        return context

    def _extract_viral_clips(self, context: Dict[str, Any]) -> List[ViralClipResult]:
        """Извлекает viral clips из контекста."""
        clips_raw = context.get("viral_clips") or context.get("highlights", [])
        results = []

        for i, clip in enumerate(clips_raw):
            if not isinstance(clip, dict):
                continue

            breakdown = clip.get("score_breakdown", {})
            results.append(ViralClipResult(
                id=clip.get("id", f"clip_{i}"),
                start=float(clip.get("start", 0)),
                end=float(clip.get("end", 0)),
                score=float(clip.get("score", 0)),
                score_breakdown=ScoreBreakdown.from_dict(breakdown) if isinstance(breakdown, dict) else ScoreBreakdown(),
                anchor_type=str(clip.get("anchor_type", "unknown")),
                reasons=clip.get("reasons", []),
            ))

        return results

    def _extract_chapters(self, context: Dict[str, Any]) -> List[ChapterResult]:
        """Извлекает chapters из контекста."""
        chapters_raw = context.get("chapters", [])
        results = []

        for i, ch in enumerate(chapters_raw):
            if not isinstance(ch, dict):
                continue

            results.append(ChapterResult(
                id=ch.get("id", f"chapter_{i}"),
                start=float(ch.get("start", 0)),
                end=float(ch.get("end", 0)),
                title=str(ch.get("title", f"Chapter {i + 1}")),
                description=str(ch.get("description", "")),
            ))

        return results

    def _extract_summary(self, context: Dict[str, Any]) -> AnalysisSummary:
        """Извлекает сводку анализа."""
        audio_features = context.get("audio_features", {}) or {}
        motion_summary = context.get("motion_summary", {}) or {}

        # Speech ratio from speech_quality or calculate
        speech_quality = context.get("speech_quality", {}) or {}
        speech_ratio = speech_quality.get("speech_ratio", 0.0)

        # Total speech duration from STT
        segments = context.get("transcript_segments", []) or []
        total_speech = sum(
            max(0, float(s.get("end", 0)) - float(s.get("start", 0)))
            for s in segments
            if isinstance(s, dict)
        )

        return AnalysisSummary(
            total_scenes=len(context.get("scene_boundaries", [])),
            total_speech_duration=total_speech,
            speech_ratio=speech_ratio,
            mean_motion=motion_summary.get("mean", 0.0),
            mean_loudness=safe_mean_points(audio_features.get("loudness", [])),
            mean_interest=safe_mean_points(context.get("timeline", [])),
            detected_language=context.get("detected_language"),
        )

    def _build_full_result(
        self,
        public: AnalysisResultPublic,
        context: Dict[str, Any],
    ) -> AnalysisResultFull:
        """Строит полный результат с дополнительными данными."""

        # Timeline preview (sampled)
        timeline_raw = context.get("timeline", []) or context.get("timeline_points", [])
        timeline_preview = self._sample_timeline(timeline_raw, max_points=100)

        # Transcript segments
        segments_raw = context.get("transcript_segments", [])
        transcript_segments = [
            TranscriptSegment(
                start=float(s.get("start", 0)),
                end=float(s.get("end", 0)),
                text=str(s.get("text", "")),
            )
            for s in segments_raw
            if isinstance(s, dict)
        ]

        # Scene boundaries
        scene_boundaries = context.get("scene_boundaries", [])

        # Node timings
        node_timings_raw = context.get("node_timings", {})
        node_timings = [
            NodeTimingResult(name=name, execution_time_seconds=time)
            for name, time in node_timings_raw.items()
        ] if isinstance(node_timings_raw, dict) else []

        # Settings snapshot
        settings = context.get("settings")
        settings_snapshot = {}
        if settings:
            try:
                settings_snapshot = {
                    k: v for k, v in vars(settings).items()
                    if not k.startswith("_") and isinstance(v, (str, int, float, bool, list, dict))
                }
            except Exception:
                pass

        return AnalysisResultFull(
            public=public,
            artifacts={},  # TODO: fill from context artifacts
            timeline_preview=timeline_preview,
            transcript_segments=transcript_segments,
            scene_boundaries=[float(b) for b in scene_boundaries],
            node_timings=node_timings,
            warnings=context.get("warnings", []),
            settings_snapshot=settings_snapshot,
            run_id=context.get("run_id"),
        )

    def _sample_timeline(
        self,
        timeline: List[Dict[str, Any]],
        max_points: int = 100,
    ) -> List[TimelinePointResult]:
        """Сэмплирует timeline до max_points."""
        if not timeline:
            return []

        # Определяем шаг сэмплирования
        step = max(1, len(timeline) // max_points)

        results = []
        for i in range(0, len(timeline), step):
            p = timeline[i]
            if not isinstance(p, dict):
                continue

            results.append(TimelinePointResult(
                time=float(p.get("time", 0)),
                interest=float(p.get("interest", 0)),
                motion=float(p.get("motion", 0)),
                audio_loudness=float(p.get("audio_loudness", p.get("loudness", 0))),
                clarity=float(p.get("clarity", 0.5)),
                sentiment=float(p.get("sentiment", 0)),
                has_laughter=bool(p.get("has_laughter", False)),
                is_scene_boundary=bool(p.get("is_scene_boundary", False)),
            ))

        return results

    # Legacy methods for backward compatibility

    def _extract_video_info(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "duration_seconds": float(context.get("duration_seconds", 0.0) or 0.0),
            "fps": float(context.get("fps", 0.0) or 0.0),
            "file_size_bytes": int(context.get("video_size_bytes", 0) or 0),
            "processing_time_seconds": float(context.get("processing_time_seconds", 0.0) or 0.0),
        }

    def _extract_motion_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "heatmap": context.get("motion_heatmap", []),
            "summary": context.get("motion_summary", {}),
        }

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
        total_speech_duration = sum(
            max(0, float(seg.get("end", 0)) - float(seg.get("start", 0)))
            for seg in segments
            if isinstance(seg, dict) and "start" in seg and "end" in seg
        )

        full = str(context.get("full_transcript", "") or "")
        return {
            "segments": segments,
            "full_transcript": full,
            "summary": {
                "total_segments": len(segments),
                "total_characters": len(full),
                "total_speech_duration": total_speech_duration,
            },
        }

    def _extract_sentiment_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "timeline": context.get("sentiment_timeline", []),
            "summary": context.get("sentiment_summary", {}),
        }

    def _extract_humor_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "scores": context.get("humor_scores", []),
            "summary": context.get("humor_summary", {}),
        }

    def _extract_topics_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        segments = context.get("topic_segments", []) or []
        unique_topics = {
            seg.get("topic")
            for seg in segments
            if isinstance(seg, dict) and "topic" in seg
        }
        unique_topics.discard(None)
        return {
            "segments": segments,
            "summary": {
                "total_segments": len(segments),
                "unique_topics": len(unique_topics),
            },
        }
