import csv
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from scipy.io import wavfile
from scipy import signal
from typing import List, Dict, Any, Optional


class YAMNetClient:
    """YAMNet client for audio event detection (laughter, applause, cheer)."""
    
    MODEL_URL = "https://tfhub.dev/google/yamnet/1"
    TARGET_SUBSTRINGS = ("laughter", "applause", "cheer")
    FRAME_HOP_SECONDS = 0.48
    
    def __init__(self):
        self._model = None
        self._class_names = None
        self._target_indices = None
    
    def _load_model(self):
        """Lazy load model and class names."""
        if self._model is None:
            self._model = hub.load(self.MODEL_URL)
            class_map_path = self._model.class_map_path().numpy()
            self._class_names = self._load_class_names(class_map_path)
            self._target_indices = self._get_target_indices()
    
    def _load_class_names(self, class_map_csv_path: str) -> List[str]:
        """Load class names from CSV."""
        class_names = []
        with tf.io.gfile.GFile(class_map_csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_names.append(row["display_name"])
        return class_names
    
    def _get_target_indices(self) -> List[int]:
        """Get indices for target classes (laughter, applause, cheer)."""
        indices = []
        for i, name in enumerate(self._class_names):
            name_lower = name.lower()
            if any(substr in name_lower for substr in self.TARGET_SUBSTRINGS):
                indices.append(i)
        return indices
    
    def _ensure_sample_rate(self, original_sr: int, waveform: np.ndarray, desired_sr: int = 16000):
        """Resample audio to desired sample rate."""
        if original_sr == desired_sr:
            return original_sr, waveform
        desired_length = int(round(float(len(waveform)) / original_sr * desired_sr))
        waveform = signal.resample(waveform, desired_length)
        return desired_sr, waveform
    
    def predict_laughter(self, audio_path: str) -> Dict[str, Any]:
        """Predict laughter timeline from audio file."""
        try:
            self._load_model()
            
            # Load and preprocess audio
            sr, wav_data = wavfile.read(audio_path)
            sr, wav_data = self._ensure_sample_rate(sr, wav_data, 16000)
            
            # Normalize to [-1, 1]
            waveform = (wav_data / tf.int16.max).astype(np.float32)
            
            # Run inference
            scores, _, _ = self._model(waveform)
            scores_np = scores.numpy()
            
            # Extract target class probabilities
            if self._target_indices:
                frame_prob = scores_np[:, self._target_indices].max(axis=1)
            else:
                frame_prob = np.zeros(scores_np.shape[0], dtype=np.float32)
            
            # Convert to 1s timeline
            duration_seconds = len(waveform) / 16000.0
            timeline = self._to_1s_timeline(frame_prob, duration_seconds)
            
            # Calculate summary stats
            summary = self._calculate_summary(frame_prob)
            
            return {
                "timeline": timeline,
                "summary": summary,
                "classes_used": [self._class_names[i] for i in self._target_indices]
            }
            
        except Exception as e:
            print(f"YAMNet prediction failed: {e}")
            return {
                "timeline": [],
                "summary": {"max_prob": 0.0, "mean_prob": 0.0, "count_positive": 0, "threshold": 0.6},
                "classes_used": []
            }
    
    def _to_1s_timeline(self, frame_prob: np.ndarray, duration_seconds: float) -> List[Dict[str, float]]:
        """Convert frame probabilities to 1-second timeline."""
        n_steps = int(np.ceil(duration_seconds))
        timeline = []
        
        frame_times = np.arange(len(frame_prob)) * self.FRAME_HOP_SECONDS
        
        for i in range(n_steps):
            t0 = float(i)
            t1 = min(float(i + 1), duration_seconds)
            
            # Find frames in this second
            mask = (frame_times >= t0) & (frame_times < t1)
            prob = float(frame_prob[mask].max()) if np.any(mask) else 0.0
            
            timeline.append({"time": t0, "prob": prob})
        
        return timeline
    
    def _calculate_summary(self, frame_prob: np.ndarray, threshold: float = 0.6) -> Dict[str, Any]:
        """Calculate summary statistics."""
        return {
            "max_prob": float(frame_prob.max()) if len(frame_prob) > 0 else 0.0,
            "mean_prob": float(frame_prob.mean()) if len(frame_prob) > 0 else 0.0,
            "count_positive": int((frame_prob >= threshold).sum()),
            "threshold": threshold
        }