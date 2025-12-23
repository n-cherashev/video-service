from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from handlers.base_handler import BaseHandler


SentimentStrategy = Literal["model", "rule"]


@dataclass(frozen=True)
class SentimentPoint:
    time: float
    score: float


class SentimentAnalysisHandler(BaseHandler):
    """
    Sentiment analysis for transcript segments.

    Output contract:
      - context["sentiment_timeline"] = [{"time": float, "score": float}, ...]
      - context["sentiment_summary"] = {"mean": float, "max_abs": float}

    Strategy:
      - "model": uses HuggingFace Transformers pipeline("sentiment-analysis")
      - "rule": lightweight rule-based fallback (always available)

    Device handling:
      - If device is a non-negative int (e.g. 0) but CUDA is not available,
        we automatically switch to CPU (device=-1) to avoid runtime crashes.
      - If pipeline initialization fails for any reason, we fallback to rule-based.
    """

    def __init__(
        self,
        strategy: SentimentStrategy = "model",
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        device: int | str | None = 0,
        batch_size: int = 8,
        max_length: int = 256,
    ) -> None:
        self.strategy: SentimentStrategy = strategy
        self.model_name = model_name
        self.device = device
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)

        self._pipe = None
        self._pipe_ready: bool = False

        if self.strategy == "model":
            self._init_model_safely()

    def handle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        segments = context.get("transcript_segments", []) or []
        if not segments:
            context["sentiment_timeline"] = []
            context["sentiment_summary"] = {"mean": 0.0, "max_abs": 0.0}
            return context

        points: List[SentimentPoint] = []

        if self.strategy == "model" and self._pipe_ready and self._pipe is not None:
            points = self._analyze_with_model(segments)
        else:
            # Always available fallback (and also used if init failed)
            points = self._analyze_rule_based(segments)

        timeline = [{"time": p.time, "score": p.score} for p in points]
        scores = [p.score for p in points]
        summary = {
            "mean": (sum(scores) / len(scores)) if scores else 0.0,
            "max_abs": max((abs(s) for s in scores), default=0.0),
        }

        context["sentiment_timeline"] = timeline
        context["sentiment_summary"] = summary
        print(f"✓ Sentiment: {len(timeline)} points")
        return context

    # -------------------------
    # Model init / device logic
    # -------------------------

    def _init_model_safely(self) -> None:
        """
        Initialize HF pipeline with robust device fallback.
        If anything goes wrong -> mark pipe not ready, and handler will fallback to rules.
        """
        try:
            from transformers import pipeline  # lazy import
        except Exception:
            self._pipe_ready = False
            self._pipe = None
            self.strategy = "rule"
            return

        device_arg = self._resolve_device(self.device)

        try:
            # task alias: "sentiment-analysis" maps to text-classification pipeline [page:0]
            self._pipe = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=device_arg,
            )
            self._pipe_ready = True
        except Exception:
            # Any runtime init errors (wrong device, missing torch backend, etc.) -> fallback
            self._pipe_ready = False
            self._pipe = None
            self.strategy = "rule"

    def _resolve_device(self, device: int | str | None) -> int | str | None:
        """
        Transformers pipelines accept:
          - int: -1 for CPU, >=0 for CUDA device index
          - str: "cpu", "cuda:0", "mps", ...
          - torch.device

        We support int/str/None and implement:
          - if int>=0 but CUDA not available => return -1
          - if str startswith "cuda" but CUDA not available => return "cpu"
          - allow "mps" on mac if available, otherwise fallback to "cpu"
        """
        # Default: if None, let transformers decide; but for reliability on machines without GPU
        # it's often better to force CPU.
        if device is None:
            return -1

        # Handle int device
        if isinstance(device, int):
            if device < 0:
                return -1
            # device >= 0 implies CUDA ordinal; verify CUDA
            if self._has_cuda():
                return device
            return -1

        # Handle string device
        dev = device.strip().lower()

        if dev.startswith("cuda"):
            return dev if self._has_cuda() else "cpu"

        if dev == "mps":
            return "mps" if self._has_mps() else "cpu"

        if dev in ("cpu",):
            return "cpu"

        # Unknown string: let transformers try, but it's safer to CPU
        return "cpu"

    def _has_cuda(self) -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _has_mps(self) -> bool:
        try:
            import torch
            # MPS available on macOS with Apple Silicon when PyTorch built with it
            return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        except Exception:
            return False

    # -------------------------
    # Inference implementations
    # -------------------------

    def _analyze_with_model(self, segments: List[Dict[str, Any]]) -> List[SentimentPoint]:
        assert self._pipe is not None

        texts: List[str] = []
        times: List[float] = []

        for seg in segments:
            if not isinstance(seg, dict):
                continue
            text = str(seg.get("text", "") or "").strip()
            if not text:
                continue
            try:
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", start))
            except (TypeError, ValueError):
                continue
            mid = (start + end) / 2.0
            texts.append(text)
            times.append(mid)

        if not texts:
            return []

        points: List[SentimentPoint] = []

        # Pipeline batching: pass list of strings, pipeline will handle batching internally. [page:0]
        # We also enforce truncation via kwargs compatible with TextClassificationPipeline.
        outputs = self._pipe(
            texts,
            batch_size=self.batch_size,
            truncation=True,
            max_length=self.max_length,
        )

        # outputs: list[{"label": "...", "score": float}, ...]
        # Map to [-1, 1]. If model labels are unknown, fallback to 0.
        for t, out in zip(times, outputs, strict=False):
            score = self._map_model_output_to_score(out)
            points.append(SentimentPoint(time=float(t), score=float(score)))

        return points

    def _map_model_output_to_score(self, out: Any) -> float:
        if not isinstance(out, dict):
            return 0.0

        label = str(out.get("label", "")).upper()
        try:
            conf = float(out.get("score", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(conf, 1.0))

        # Common label conventions:
        # - POSITIVE/NEGATIVE
        # - LABEL_0/LABEL_1/LABEL_2 (varies by model)
        if "POS" in label:
            return conf
        if "NEG" in label:
            return -conf

        # Heuristic for 3-class roberta sentiment (often: LABEL_0=NEG, LABEL_1=NEU, LABEL_2=POS)
        if label == "LABEL_2":
            return conf
        if label == "LABEL_0":
            return -conf
        if label == "LABEL_1":
            return 0.0

        return 0.0

    def _analyze_rule_based(self, segments: List[Dict[str, Any]]) -> List[SentimentPoint]:
        """
        Нейтральный fallback: не пытается угадать эмоции, только размечает
        временные точки наличия речи с score = 0.0.
        """
        points: List[SentimentPoint] = []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            text = str(seg.get("text", "") or "").strip()
            if not text:
                continue
            try:
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", start))
            except (TypeError, ValueError):
                continue
            mid = (start + end) / 2.0
            points.append(SentimentPoint(time=float(mid), score=0.0))
        return points

