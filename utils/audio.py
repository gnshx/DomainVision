import os
import threading
import wave
import numpy as np


class CursedSoundSynthesizer:
    """Procedurally synthesizes and plays JJK-style cursed energy sound effects."""

    SAMPLE_RATE = 22050

    def __init__(self, sfx_dir: str = "assets/sounds"):
        self.sfx_dir = sfx_dir
        os.makedirs(self.sfx_dir, exist_ok=True)
        self.has_sounddevice = False
        try:
            import sounddevice as sd
            self.sd = sd
            self.has_sounddevice = True
        except Exception:
            self.sd = None

        # Pre-synthesize core audio clips
        self.sounds = {
            "charge": self._generate_charge_sound(),
            "flash": self._generate_flash_sound(),
            "expansion": self._generate_expansion_sound(),
            "collapse": self._generate_collapse_sound()
        }

    def _generate_charge_sound(self, duration: float = 1.0) -> np.ndarray:
        """Rising frequency sweep + rumbling low frequency."""
        t = np.linspace(0, duration, int(self.SAMPLE_RATE * duration), endpoint=False)
        # 60Hz to 350Hz exponential pitch rise
        freq = np.exp(np.linspace(np.log(60), np.log(350), len(t)))
        phase = 2 * np.pi * np.cumsum(freq) / self.SAMPLE_RATE
        sweep = 0.5 * np.sin(phase)

        # Low resonant sub-bass
        sub = 0.4 * np.sin(2 * np.pi * 50 * t)

        # Pulsing modulation
        mod = 0.5 + 0.5 * np.sin(2 * np.pi * 8 * t)
        wave_data = (sweep * mod + sub) * np.linspace(0.1, 1.0, len(t))
        return (wave_data * 32767 * 0.7).astype(np.int16)

    def _generate_flash_sound(self, duration: float = 0.8) -> np.ndarray:
        """Sub-bass drop + sharp high-frequency transient."""
        t = np.linspace(0, duration, int(self.SAMPLE_RATE * duration), endpoint=False)
        # Deep bass drop: 120Hz down to 35Hz
        freq = np.exp(np.linspace(np.log(120), np.log(35), len(t)))
        phase = 2 * np.pi * np.cumsum(freq) / self.SAMPLE_RATE
        bass = np.sin(phase) * np.exp(-3.0 * t)

        # Sharp snap
        noise = np.random.uniform(-1, 1, len(t)) * np.exp(-25.0 * t)
        wave_data = 0.8 * bass + 0.4 * noise
        return (np.clip(wave_data, -1, 1) * 32767 * 0.8).astype(np.int16)

    def _generate_expansion_sound(self, duration: float = 2.0) -> np.ndarray:
        """Ominous celestial resonance with metallic ringing chords."""
        t = np.linspace(0, duration, int(self.SAMPLE_RATE * duration), endpoint=False)
        f1, f2, f3 = 55.0, 110.0, 220.0
        wave_data = (
            0.5 * np.sin(2 * np.pi * f1 * t) +
            0.3 * np.sin(2 * np.pi * f2 * t) +
            0.2 * np.sin(2 * np.pi * f3 * t) +
            0.1 * np.sin(2 * np.pi * 440 * t) * np.sin(2 * np.pi * 3 * t)
        )
        envelope = np.exp(-1.2 * t)
        return (wave_data * envelope * 32767 * 0.7).astype(np.int16)

    def _generate_collapse_sound(self, duration: float = 0.8) -> np.ndarray:
        """Glassy shatter and dissipate sound."""
        t = np.linspace(0, duration, int(self.SAMPLE_RATE * duration), endpoint=False)
        noise = np.random.uniform(-1, 1, len(t))
        envelope = np.exp(-4.0 * t)
        sine = 0.4 * np.sin(2 * np.pi * 880 * t) * envelope
        wave_data = (0.5 * noise * envelope + sine)
        return (np.clip(wave_data, -1, 1) * 32767 * 0.7).astype(np.int16)

    def play(self, sound_name: str):
        """Asynchronously play sound effect if audio output is available."""
        if not self.has_sounddevice or sound_name not in self.sounds:
            return

        def _play_worker():
            try:
                data = self.sounds[sound_name].astype(np.float32) / 32767.0
                self.sd.play(data, self.SAMPLE_RATE)
            except Exception:
                pass

        threading.Thread(target=_play_worker, daemon=True).start()

    def export_wav(self, sound_name: str, filepath: str):
        """Export sound to WAV file."""
        if sound_name not in self.sounds:
            return
        data = self.sounds[sound_name]
        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(data.tobytes())
