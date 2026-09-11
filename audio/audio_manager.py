import os
import time
import wave
import shutil
import subprocess
import threading
from typing import Dict, Optional, List, Tuple
import numpy as np

from config import AUDIO_TIMELINE


class AudioManager:
    """
    High-performance audio manager and timeline sequencer for DomainVision.
    - Synchronizes exact Crunchyroll Sukuna activation timing (0.00s -> 1.50s)
    - Multi-backend playback: aplay (ALSA), pw-play (PipeWire), sounddevice, or silent fallback
    - Non-blocking, frame-accurate execution without blocking video rendering
    - Procedural generation of original/royalty-free cursed energy soundscape
    """

    SAMPLE_RATE = 22050

    def __init__(self, sounds_dir: str = "assets/sounds"):
        self.sounds_dir = sounds_dir
        os.makedirs(self.sounds_dir, exist_ok=True)

        # Detect audio backend
        self.backend = self._detect_backend()
        print(f"[AudioManager] Audio playback engine: {self.backend}")

        self.audio_data: Dict[str, np.ndarray] = {}
        self.sound_paths: Dict[str, str] = {}
        self._ensure_all_sound_assets()

        # Looping ambience state
        self._ambience_process: Optional[subprocess.Popen] = None
        self._ambience_stop_event = threading.Event()

        # Sequencer state
        self._sequence_active = False
        self._sequence_start_time = 0.0
        self._scheduled_timers: List[threading.Timer] = []

    def _detect_backend(self) -> str:
        if shutil.which("aplay"):
            return "aplay"
        if shutil.which("pw-play"):
            return "pw-play"
        try:
            import sounddevice as sd
            self.sd = sd
            return "sounddevice"
        except Exception:
            return "dummy"

    def _ensure_all_sound_assets(self):
        """Generate and verify all required WAV files."""
        generators = {
            "charge.wav": self._gen_charge,
            "energy_build.wav": self._gen_energy_build,
            "voice_domain.wav": self._gen_voice_domain,
            "flash.wav": self._gen_flash,
            "shockwave.wav": self._gen_shockwave,
            "activation.wav": self._gen_flash,
            "impact.wav": self._gen_shockwave,
            "shrine_ambience.wav": self._gen_shrine_ambience,
            "void_ambience.wav": self._gen_void_ambience,
            "collapse.wav": self._gen_collapse,
        }

        for fname, gen_fn in generators.items():
            path = os.path.join(self.sounds_dir, fname)
            key = fname.replace(".wav", "")
            self.sound_paths[key] = path

            if not os.path.exists(path):
                samples = gen_fn()
                self._write_wav(path, samples)
                self.audio_data[key] = samples
            else:
                # Load existing WAV
                try:
                    with wave.open(path, "rb") as wf:
                        n_frames = wf.getnframes()
                        raw_data = wf.readframes(n_frames)
                        self.audio_data[key] = np.frombuffer(raw_data, dtype=np.int16)
                except Exception:
                    samples = gen_fn()
                    self._write_wav(path, samples)
                    self.audio_data[key] = samples

    def _write_wav(self, filepath: str, samples: np.ndarray):
        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(samples.tobytes())

    # --- Procedural Audio Synthesizers (Royalty-Free Original JJK Soundscapes) ---

    def _gen_charge(self, duration: float = 0.85) -> np.ndarray:
        """Rising frequency sweep + rumbling low frequency."""
        t = np.linspace(0, duration, int(self.SAMPLE_RATE * duration), endpoint=False)
        freq = np.exp(np.linspace(np.log(60), np.log(380), len(t)))
        phase = 2 * np.pi * np.cumsum(freq) / self.SAMPLE_RATE
        sweep = 0.55 * np.sin(phase)
        sub = 0.4 * np.sin(2 * np.pi * 52 * t)
        mod = 0.5 + 0.5 * np.sin(2 * np.pi * 12 * t)
        envelope = np.linspace(0.15, 1.0, len(t))
        wave_data = (sweep * mod + sub) * envelope
        return (wave_data * 32767 * 0.75).astype(np.int16)

    def _gen_energy_build(self, duration: float = 0.7) -> np.ndarray:
        """Cursed energy crackling + accelerating electrical harmonics."""
        t = np.linspace(0, duration, int(self.SAMPLE_RATE * duration), endpoint=False)
        f0 = np.linspace(120, 520, len(t))
        phase = 2 * np.pi * np.cumsum(f0) / self.SAMPLE_RATE
        harmonic1 = 0.4 * np.sin(phase)
        harmonic2 = 0.3 * np.sin(phase * 2.0)
        crackle = np.random.uniform(-1, 1, len(t)) * (np.random.random(len(t)) > 0.85) * 0.5
        pulsing = 0.5 + 0.5 * np.sin(2 * np.pi * 24 * t)
        wave_data = (harmonic1 + harmonic2 + crackle) * pulsing * np.linspace(0.3, 1.0, len(t))
        return (np.clip(wave_data, -1, 1) * 32767 * 0.8).astype(np.int16)

    def _gen_voice_domain(self, duration: float = 1.3) -> np.ndarray:
        """
        Original synthesized voice line: Resonant deep robotic/demonic Japanese formant articulation
        representing 'Ryōiki Tenkai' ('Domain Expansion').
        Utilizes triple-formant filter banks, deep fundamental pitch (72Hz), and subtle cursed distortion.
        """
        t = np.linspace(0, duration, int(self.SAMPLE_RATE * duration), endpoint=False)
        # Deep menacing fundamental pitch with slight pitch drift
        f0 = 74.0 + 8.0 * np.sin(2 * np.pi * 1.5 * t)
        phase_f0 = 2 * np.pi * np.cumsum(f0) / self.SAMPLE_RATE
        # Formant frequencies for resonant Japanese vowels (o-i-i-e-a-i)
        f1 = 550.0 + 120.0 * np.sin(2 * np.pi * 2.2 * t)
        f2 = 1350.0 + 200.0 * np.cos(2 * np.pi * 2.8 * t)
        f3 = 2600.0 + 150.0 * np.sin(2 * np.pi * 3.4 * t)

        vocal_source = np.sin(phase_f0) + 0.6 * np.sin(phase_f0 * 2) + 0.35 * np.sin(phase_f0 * 3)
        res1 = np.sin(2 * np.pi * f1 * t) * np.exp(-t * 0.8)
        res2 = np.sin(2 * np.pi * f2 * t) * np.exp(-t * 1.1)
        res3 = np.sin(2 * np.pi * f3 * t) * np.exp(-t * 1.4)

        voice = vocal_source * (0.45 * res1 + 0.35 * res2 + 0.2 * res3)
        # Add subtle dark saturation
        voice = np.tanh(voice * 2.2)
        # Envelope: distinct 2-part syllable emphasis (Ryō-iki / Ten-kai)
        syl1 = np.exp(-((t - 0.25) / 0.18) ** 2)
        syl2 = np.exp(-((t - 0.75) / 0.28) ** 2)
        env = 0.35 + 0.65 * (syl1 + syl2)
        out = voice * env
        return (np.clip(out, -1, 1) * 32767 * 0.9).astype(np.int16)

    def _gen_flash(self, duration: float = 0.65) -> np.ndarray:
        """Sudden intense flash transient + deep bass drop."""
        t = np.linspace(0, duration, int(self.SAMPLE_RATE * duration), endpoint=False)
        freq = np.exp(np.linspace(np.log(160), np.log(38), len(t)))
        phase = 2 * np.pi * np.cumsum(freq) / self.SAMPLE_RATE
        bass = np.sin(phase) * np.exp(-3.5 * t)
        snap = np.random.uniform(-1, 1, len(t)) * np.exp(-35.0 * t)
        out = bass * 0.85 + snap * 0.5
        return (np.clip(out, -1, 1) * 32767 * 0.88).astype(np.int16)

    def _gen_shockwave(self, duration: float = 0.85) -> np.ndarray:
        """Explosive spatial barrier shockwave boom with rumble."""
        t = np.linspace(0, duration, int(self.SAMPLE_RATE * duration), endpoint=False)
        boom = np.sin(2 * np.pi * 50 * t) * np.exp(-2.5 * t)
        sub_rumble = np.sin(2 * np.pi * 32 * t) * np.exp(-1.5 * t)
        blast_noise = np.random.uniform(-1, 1, len(t)) * np.exp(-14.0 * t)
        out = 0.5 * boom + 0.4 * sub_rumble + 0.4 * blast_noise
        return (np.clip(out, -1, 1) * 32767 * 0.85).astype(np.int16)

    def _gen_shrine_ambience(self, duration: float = 3.5) -> np.ndarray:
        """Malevolent Shrine ominous dark drone + sinister Buddhist bell chime."""
        t = np.linspace(0, duration, int(self.SAMPLE_RATE * duration), endpoint=False)
        f_drone = 55.0
        drone = (
            0.4 * np.sin(2 * np.pi * f_drone * t) +
            0.25 * np.sin(2 * np.pi * (f_drone * 2) * t) +
            0.15 * np.sin(2 * np.pi * (f_drone * 3.01) * t)
        )
        # Deep temple bell resonant strike at t = 0.2s
        bell_t = np.maximum(0.0, t - 0.2)
        f_bell = 216.0
        bell = (
            0.4 * np.sin(2 * np.pi * f_bell * bell_t) +
            0.2 * np.sin(2 * np.pi * (f_bell * 1.48) * bell_t) +
            0.15 * np.sin(2 * np.pi * (f_bell * 2.05) * bell_t)
        ) * np.exp(-1.2 * bell_t)

        out = drone + bell
        return (np.clip(out, -1, 1) * 32767 * 0.7).astype(np.int16)

    def _gen_void_ambience(self, duration: float = 3.5) -> np.ndarray:
        """Infinite Void ethereal celestial harmonic drone."""
        t = np.linspace(0, duration, int(self.SAMPLE_RATE * duration), endpoint=False)
        f1, f2, f3 = 110.0, 220.0, 330.0
        shimmer = np.sin(2 * np.pi * 4.0 * t) * 0.15
        drone = (
            0.35 * np.sin(2 * np.pi * f1 * t) +
            0.25 * np.sin(2 * np.pi * f2 * t + shimmer) +
            0.20 * np.sin(2 * np.pi * f3 * t)
        ) * (0.85 + 0.15 * np.sin(2 * np.pi * 0.8 * t))
        return (np.clip(drone, -1, 1) * 32767 * 0.65).astype(np.int16)

    def _gen_collapse(self, duration: float = 0.9) -> np.ndarray:
        """Glassy domain barrier shatter and dissipation."""
        t = np.linspace(0, duration, int(self.SAMPLE_RATE * duration), endpoint=False)
        shatter = np.random.uniform(-1, 1, len(t)) * np.exp(-6.0 * t)
        hum = 0.4 * np.sin(2 * np.pi * 175 * t) * np.exp(-3.0 * t)
        out = shatter * 0.6 + hum * 0.35
        return (np.clip(out, -1, 1) * 32767 * 0.75).astype(np.int16)

    # --- Non-Blocking Playback API ---

    def play(self, sound_name: str):
        """Play a sound effect immediately without blocking video loop."""
        if sound_name not in self.sound_paths:
            return

        filepath = self.sound_paths[sound_name]

        def _worker():
            try:
                if self.backend == "aplay":
                    subprocess.run(
                        ["aplay", "-q", filepath],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False
                    )
                elif self.backend == "pw-play":
                    subprocess.run(
                        ["pw-play", filepath],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False
                    )
                elif self.backend == "sounddevice" and hasattr(self, "sd") and self.sd:
                    data = self.audio_data[sound_name].astype(np.float32) / 32767.0
                    self.sd.play(data, self.SAMPLE_RATE)
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True).start()

    def start_domain_sequence(self, theme: str = "malevolent_shrine"):
        """
        Executes the canonical Crunchyroll Sukuna activation timing sequence:
        0.00s   hand sign recognized
        0.05s   charge sound
        0.70s   energy builds
        1.20s   voice/activation ('Domain Expansion')
        1.35s   flash
        1.40s   shockwave
        1.50s   domain environment & background ambience
        """
        self.stop_domain_sequence()
        self._sequence_active = True
        self._sequence_start_time = time.time()

        ambience_name = "shrine_ambience" if theme == "malevolent_shrine" else "void_ambience"

        schedule_plan = [
            (AUDIO_TIMELINE.get("charge", 0.05), "charge"),
            (AUDIO_TIMELINE.get("energy_build", 0.70), "energy_build"),
            (AUDIO_TIMELINE.get("voice", 1.20), "voice_domain"),
            (AUDIO_TIMELINE.get("flash", 1.35), "flash"),
            (AUDIO_TIMELINE.get("shockwave", 1.40), "shockwave"),
            (AUDIO_TIMELINE.get("ambience", 1.50), ambience_name),
        ]

        for delay_sec, sname in schedule_plan:
            timer = threading.Timer(delay_sec, self._timed_play, args=[sname])
            timer.daemon = True
            timer.start()
            self._scheduled_timers.append(timer)

    def _timed_play(self, sound_name: str):
        if self._sequence_active:
            self.play(sound_name)

    def stop_domain_sequence(self):
        """Cancel pending sequence timers and stop ambience."""
        self._sequence_active = False
        for timer in self._scheduled_timers:
            try:
                timer.cancel()
            except Exception:
                pass
        self._scheduled_timers.clear()

    def play_collapse(self):
        """Play domain barrier collapse effect."""
        self.stop_domain_sequence()
        self.play("collapse")
