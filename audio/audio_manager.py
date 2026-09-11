import os
import wave
import threading
from typing import Dict, Optional
import numpy as np


class AudioManager:
    """
    Frame-accurate, state-synchronized audio manager for DomainVision.
    Plays sound effects strictly on state machine transitions without blocking rendering.
    """

    SAMPLE_RATE = 22050

    def __init__(self, sounds_dir: str = "assets/sounds"):
        self.sounds_dir = sounds_dir
        os.makedirs(self.sounds_dir, exist_ok=True)
        self.has_audio = False
        try:
            import sounddevice as sd
            self.sd = sd
            self.has_audio = True
        except Exception:
            self.sd = None

        self.audio_data: Dict[str, np.ndarray] = {}
        self._ensure_sound_files()

    def _ensure_sound_files(self):
        """Generate and cache WAV sound assets."""
        sounds = {
            "charge.wav": self._gen_charge(),
            "activation.wav": self._gen_activation(),
            "impact.wav": self._gen_impact(),
            "slash.wav": self._gen_slash(),
            "collapse.wav": self._gen_collapse(),
        }

        for fname, samples in sounds.items():
            path = os.path.join(self.sounds_dir, fname)
            if not os.path.exists(path):
                self._write_wav(path, samples)
            self.audio_data[fname.replace(".wav", "")] = samples

    def _write_wav(self, filepath: str, samples: np.ndarray):
        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(samples.tobytes())

    def _gen_charge(self, duration: float = 0.9) -> np.ndarray:
        t = np.linspace(0, duration, int(self.SAMPLE_RATE * duration), endpoint=False)
        freq = np.exp(np.linspace(np.log(70), np.log(420), len(t)))
        phase = 2 * np.pi * np.cumsum(freq) / self.SAMPLE_RATE
        sweep = 0.6 * np.sin(phase)
        sub = 0.35 * np.sin(2 * np.pi * 55 * t)
        mod = 0.5 + 0.5 * np.sin(2 * np.pi * 10 * t)
        wave_data = (sweep * mod + sub) * np.linspace(0.1, 1.0, len(t))
        return (wave_data * 32767 * 0.75).astype(np.int16)

    def _gen_activation(self, duration: float = 1.2) -> np.ndarray:
        t = np.linspace(0, duration, int(self.SAMPLE_RATE * duration), endpoint=False)
        freq = np.exp(np.linspace(np.log(140), np.log(35), len(t)))
        phase = 2 * np.pi * np.cumsum(freq) / self.SAMPLE_RATE
        bass = np.sin(phase) * np.exp(-2.2 * t)
        snap = np.random.uniform(-1, 1, len(t)) * np.exp(-30.0 * t)
        return (np.clip(bass * 0.8 + snap * 0.4, -1, 1) * 32767 * 0.85).astype(np.int16)

    def _gen_impact(self, duration: float = 0.7) -> np.ndarray:
        t = np.linspace(0, duration, int(self.SAMPLE_RATE * duration), endpoint=False)
        noise = np.random.uniform(-1, 1, len(t)) * np.exp(-12.0 * t)
        boom = np.sin(2 * np.pi * 65 * t) * np.exp(-4.0 * t)
        return (np.clip(boom * 0.7 + noise * 0.4, -1, 1) * 32767 * 0.8).astype(np.int16)

    def _gen_slash(self, duration: float = 0.45) -> np.ndarray:
        t = np.linspace(0, duration, int(self.SAMPLE_RATE * duration), endpoint=False)
        freq = np.exp(np.linspace(np.log(800), np.log(120), len(t)))
        phase = 2 * np.pi * np.cumsum(freq) / self.SAMPLE_RATE
        whistle = np.sin(phase) * np.exp(-6.0 * t)
        snap = np.random.uniform(-1, 1, len(t)) * np.exp(-18.0 * t)
        return (np.clip(whistle * 0.6 + snap * 0.5, -1, 1) * 32767 * 0.8).astype(np.int16)

    def _gen_collapse(self, duration: float = 0.8) -> np.ndarray:
        t = np.linspace(0, duration, int(self.SAMPLE_RATE * duration), endpoint=False)
        shatter = np.random.uniform(-1, 1, len(t)) * np.exp(-5.0 * t)
        hum = 0.4 * np.sin(2 * np.pi * 180 * t) * np.exp(-2.5 * t)
        return (np.clip(shatter * 0.5 + hum * 0.4, -1, 1) * 32767 * 0.7).astype(np.int16)

    def play(self, sound_name: str):
        """Asynchronously play a sound clip without blocking."""
        if not self.has_audio or sound_name not in self.audio_data:
            return

        def _worker():
            try:
                data = self.audio_data[sound_name].astype(np.float32) / 32767.0
                self.sd.play(data, self.SAMPLE_RATE)
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True).start()
