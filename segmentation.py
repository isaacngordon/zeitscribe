#!/usr/bin/env python3
"""
segmentation.py

Energy-based speech segmentation with live transcription attempts.

Usage:
  python segmentation.py /path/to/audio.(wav|mp3|m4a|flac|...)

Features:
- Computes short-term energy to detect speech segments.
- Prints start/end timestamps for each segment in real time.
- For each segment, attempts transcription twice:
    1) Force Hebrew ("he")
    2) Force English ("en")
  Keeps the first successful, non-empty result.
- Uses mlx-whisper if available, else faster-whisper, else openai-whisper.

Notes:
- This script prefers local Whisper models; install one of:
    pip install mlx-whisper
  or
    pip install faster-whisper
  or
    pip install openai-whisper

- Audio decoding uses either:
    - soundfile (libsndfile) + resampy (optional) or
    - librosa
  If neither is available, it falls back to ffmpeg via subprocess to read PCM.

"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ------------------------------
# Audio loading utilities
# ------------------------------

def _have(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def load_audio_mono_float32(path: str, target_sr: int = 16000) -> Tuple[List[float], int]:
    """
    Load audio as mono float32 at target_sr.
    Tries: librosa -> soundfile -> ffmpeg fallback.
    Returns (samples, sample_rate).
    """
    # 1) librosa path (convenient, handles resample)
    if _have("librosa"):
        import librosa

        y, sr = librosa.load(path, sr=target_sr, mono=True)
        return y.astype("float32").tolist(), target_sr

    # 2) soundfile path
    if _have("soundfile"):
        import soundfile as sf
        import numpy as np

        data, sr = sf.read(path, always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        # Resample if required
        if sr != target_sr:
            if _have("resampy"):
                import resampy

                data = resampy.resample(data, sr, target_sr)
                sr = target_sr
            else:
                # Naive linear resample fallback to keep dependencies light
                factor = target_sr / float(sr)
                new_len = int(round(len(data) * factor))
                if new_len <= 1:
                    new_len = 1
                x_old = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
                x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
                data = np.interp(x_new, x_old, data)
                sr = target_sr
        data = data.astype("float32")
        return data.tolist(), sr

    # 3) ffmpeg fallback (requires ffmpeg in PATH)
    import subprocess
    import numpy as np

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        path,
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "-f",
        "f32le",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("Error: Neither librosa/soundfile are installed, and ffmpeg is not available.", file=sys.stderr)
        print("Please install one audio backend: `pip install librosa` or `pip install soundfile resampy`,", file=sys.stderr)
        print("or install ffmpeg.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print("ffmpeg failed to decode the audio:", e.stderr.decode("utf-8", errors="ignore"), file=sys.stderr)
        sys.exit(1)

    data = np.frombuffer(proc.stdout, dtype=np.float32)
    return data.tolist(), target_sr


# ------------------------------
# Energy-based segmentation (simple VAD)
# ------------------------------

@dataclass
class Segment:
    start: float
    end: float


def energy_vad(
    y: List[float],
    sr: int,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0,
    threshold_db: float = -35.0,
    min_speech_ms: float = 300.0,
    min_silence_ms: float = 250.0,
    pad_ms: float = 100.0,
) -> List[Segment]:
    """
    Simple energy-based VAD to produce speech segments.
    - Computes log-energy per frame.
    - Thresholds by a global floor (median + offset) and absolute dB floor.
    - Applies min speech duration and min silence merging.
    - Adds small padding around segments.
    """
    import numpy as np

    x = np.asarray(y, dtype=np.float32)
    if len(x) == 0:
        return []

    frame_len = int(sr * frame_ms / 1000.0)
    hop_len = int(sr * hop_ms / 1000.0)
    if frame_len <= 0:
        frame_len = 1
    if hop_len <= 0:
        hop_len = 1

    # Frame the signal
    num_frames = 1 + max(0, (len(x) - frame_len) // hop_len)
    if num_frames <= 0:
        num_frames = 1
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(num_frames, frame_len),
        strides=(x.strides[0] * hop_len, x.strides[0]),
        writeable=False,
    )
    # Compute log-energy
    energy = np.log10((frames ** 2).mean(axis=1) + 1e-10) * 10.0  # dB-ish scale

    # Adaptive threshold using median energy
    med = np.median(energy)
    thr = max(threshold_db, med - 3.0)  # ensure not too low
    voiced = energy > thr

    # Merge into contiguous segments in frame units
    segments_f = []
    in_seg = False
    seg_start = 0
    for i, v in enumerate(voiced):
        if v and not in_seg:
            in_seg = True
            seg_start = i
        elif not v and in_seg:
            in_seg = False
            segments_f.append((seg_start, i))
    if in_seg:
        segments_f.append((seg_start, len(voiced)))

    # Filter by min speech duration
    min_speech_frames = max(1, int(min_speech_ms / hop_ms))
    segments_f = [s for s in segments_f if (s[1] - s[0]) >= min_speech_frames]

    # Merge segments separated by short silence
    merged = []
    min_silence_frames = max(1, int(min_silence_ms / hop_ms))
    for s in segments_f:
        if not merged:
            merged.append(list(s))
            continue
        prev = merged[-1]
        if s[0] - prev[1] <= min_silence_frames:
            prev[1] = s[1]
        else:
            merged.append(list(s))

    # Convert to seconds with padding
    pad_frames = int(pad_ms / hop_ms)
    out: List[Segment] = []
    for a, b in merged:
        start_s = max(0, (a - pad_frames) * hop_ms / 1000.0)
        end_s = (b + pad_frames) * hop_ms / 1000.0
        out.append(Segment(start=start_s, end=end_s))

    # Clamp to audio length
    duration = len(x) / float(sr)
    for seg in out:
        seg.end = min(duration, seg.end)
    return out


# ------------------------------
# Transcription backends (Whisper)
# ------------------------------

class Transcriber:
    def __init__(self, model_size: str = "small") -> None:
        self.backend = None
        self.model = None
        self.model_size = model_size
        self._init_backend()

    def _init_backend(self) -> None:
        # Try mlx-whisper first (optimized for Apple Silicon)
        if _have("mlx_whisper"):
            try:
                import mlx_whisper  # type: ignore

                self._mlx_load_model = getattr(mlx_whisper, "load_model", None)
                self._mlx_transcribe = getattr(mlx_whisper, "transcribe", None)
                if self._mlx_load_model is None or self._mlx_transcribe is None:
                    raise RuntimeError("mlx_whisper missing load_model/transcribe APIs")
                try:
                    self.model = self._mlx_load_model(self.model_size)
                except Exception as e1:
                    # Try common repo prefix used by MLX community weights
                    alt_name = f"mlx-community/whisper-{self.model_size}"
                    try:
                        self.model = self._mlx_load_model(alt_name)
                    except Exception as e2:
                        raise RuntimeError(f"mlx-whisper load_model failed for '{self.model_size}' and '{alt_name}': {e1} | {e2}")
                self.backend = "mlx-whisper"
                return
            except Exception as e:
                warnings.warn(f"Failed to init mlx-whisper: {e}")

        # Try faster-whisper next
        if _have("faster_whisper"):
            from faster_whisper import WhisperModel  # type: ignore

            try:
                # Run on CPU by default; user can set device via env
                device = os.environ.get("WHISPER_DEVICE", "cpu")
                compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
                self.model = WhisperModel(self.model_size, device=device, compute_type=compute_type)
                self.backend = "faster-whisper"
                return
            except Exception as e:
                warnings.warn(f"Failed to init faster-whisper: {e}")

        # Fallback to openai-whisper
        if _have("whisper"):
            import whisper  # type: ignore

            try:
                device = os.environ.get("WHISPER_DEVICE", "cpu")
                # Device control for openai-whisper is implicit; we’ll just load.
                self.model = whisper.load_model(self.model_size)
                self.backend = "openai-whisper"
                return
            except Exception as e:
                warnings.warn(f"Failed to init openai-whisper: {e}")

        self.backend = None
        self.model = None

    def available(self) -> bool:
        return self.backend is not None and self.model is not None

    def transcribe_wav(self, wav_path: str, language: str) -> str:
        """
        Return transcript text (may be empty). language like 'he' or 'en'.
        """
        if not self.available():
            raise RuntimeError("No transcription backend available. Install mlx-whisper, faster-whisper, or openai-whisper.")

        if self.backend == "mlx-whisper":
            try:
                # mlx-whisper may expose transcribe as either a bound method on the model
                # instance (model.transcribe(...)) or as a module-level function that
                # expects the model as the first argument (mlx_whisper.transcribe(model, ...)).
                # Prefer the bound method to avoid passing the model twice.
                if hasattr(self.model, "transcribe") and callable(getattr(self.model, "transcribe")):
                    result = self.model.transcribe(wav_path, language=language, task="transcribe")
                elif callable(self._mlx_transcribe):
                    # Try common module-level signatures
                    try:
                        result = self._mlx_transcribe(self.model, audio=wav_path, language=language, task="transcribe")
                    except TypeError:
                        result = self._mlx_transcribe(self.model, wav_path, language, "transcribe")
                else:
                    raise RuntimeError("mlx_whisper transcribe API not found")

            except Exception as e:
                # Let caller handle the warning; re-raise so upstream warning shows the real error
                raise

            text = ""
            if isinstance(result, dict):
                text = (result.get("text") or "").strip()
                if not text and isinstance(result.get("segments"), list):
                    parts = []
                    for seg in result["segments"]:
                        t = (seg.get("text") or "").strip()
                        if t:
                            parts.append(t)
                    text = " ".join(parts).strip()
            return text

        if self.backend == "faster-whisper":
            from faster_whisper.transcribe import Segment as FW_Segment  # type: ignore

            segments, info = self.model.transcribe(wav_path, language=language, beam_size=1, vad_filter=False)
            out_parts = []
            for seg in segments:
                # seg may be a generator of Segment objects
                if hasattr(seg, "text"):
                    out_parts.append(seg.text)
            return " ".join(s.strip() for s in out_parts if s and s.strip())

        elif self.backend == "openai-whisper":
            # openai-whisper returns a dict with 'text'
            import whisper  # type: ignore

            result = self.model.transcribe(
                wav_path,
                language=language,
                task="transcribe",
                verbose=False,
                condition_on_previous_text=False,
            )
            return (result.get("text") or "").strip()

        else:
            raise RuntimeError("Unsupported transcription backend state.")


# ------------------------------
# Segment extraction utility
# ------------------------------

def write_wav_segment(
    y: List[float], sr: int, seg: Segment, out_path: str
) -> None:
    import numpy as np
    import wave

    start = int(max(0, math.floor(seg.start * sr)))
    end = int(min(len(y), math.ceil(seg.end * sr)))
    chunk = np.asarray(y[start:end], dtype=np.float32)
    if chunk.size == 0:
        # write minimal silence to keep downstream stable
        chunk = np.zeros(1, dtype=np.float32)

    # Convert float32 [-1,1] to int16 PCM
    pcm = np.clip(chunk, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)

    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


# ------------------------------
# CLI
# ------------------------------

def fmt_ts(t: float) -> str:
    mm, ss = divmod(int(t), 60)
    ms = int((t - int(t)) * 1000)
    return f"{mm:02d}:{ss:02d}.{ms:03d}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Energy-based segmentation with live he/en transcription attempts")
    parser.add_argument("audio", help="Path to audio file (wav/mp3/m4a/flac/...)")
    parser.add_argument("--model", default="small", help="Whisper model size (e.g., tiny, base, small, medium)")
    parser.add_argument("--backend", default="auto", choices=["auto", "mlx", "faster", "openai", "none"], help="Transcription backend preference or disable with 'none'")
    parser.add_argument("--no-transcribe", action="store_true", help="Skip transcription; only print segments")
    parser.add_argument("--workdir", default=None, help="Working directory for temp segments (default: system temp)")
    parser.add_argument("--frame-ms", type=float, default=30.0, help="Frame size in ms for energy")
    parser.add_argument("--hop-ms", type=float, default=10.0, help="Hop size in ms for energy")
    parser.add_argument("--threshold-db", type=float, default=-35.0, help="Energy threshold in dB-ish scale")
    parser.add_argument("--min-speech-ms", type=float, default=300.0, help="Minimum speech duration in ms")
    parser.add_argument("--min-silence-ms", type=float, default=250.0, help="Minimum silence between segments to split, in ms")
    parser.add_argument("--pad-ms", type=float, default=100.0, help="Padding around segments in ms")
    args = parser.parse_args()

    audio_path = args.audio
    if not os.path.exists(audio_path):
        print(f"Error: File not found: {audio_path}", file=sys.stderr)
        return 2

    print("Loading audio…", flush=True)
    y, sr = load_audio_mono_float32(audio_path, target_sr=16000)
    duration = len(y) / float(sr)
    print(f"Loaded: {os.path.basename(audio_path)} | {sr} Hz | {duration:.2f}s")

    print("Detecting segments (energy-based)…", flush=True)
    segments = energy_vad(
        y,
        sr,
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        threshold_db=args.threshold_db,
        min_speech_ms=args.min_speech_ms,
        min_silence_ms=args.min_silence_ms,
        pad_ms=args.pad_ms,
    )

    if not segments:
        print("No speech-like segments detected.")
        return 0

    # Display all segments early for visibility
    print(f"Found {len(segments)} segment(s):")
    for i, s in enumerate(segments, 1):
        print(f"  [{i:02d}] {fmt_ts(s.start)} -> {fmt_ts(s.end)}")

    transcriber = None
    if not args.no_transcribe and args.backend != "none":
        # If user requested a specific backend, try to initialize that backend explicitly.
        def _try_init_mlx(model_size: str):
            try:
                if not _have("mlx_whisper"):
                    return None, None
                import mlx_whisper  # type: ignore

                load_model = getattr(mlx_whisper, "load_model", None)
                transcribe_fn = getattr(mlx_whisper, "transcribe", None)
                if callable(load_model):
                    try:
                        model = load_model(model_size)
                    except Exception:
                        alt_name = f"mlx-community/whisper-{model_size}"
                        model = load_model(alt_name)
                    return model, transcribe_fn

                # Some versions may expose a model class on the module
                ModelCls = getattr(mlx_whisper, "WhisperModel", None) or getattr(mlx_whisper, "Whisper", None)
                if ModelCls:
                    model = ModelCls(model_size)
                    return model, getattr(model, "transcribe", None)

            except Exception as e:
                warnings.warn(f"Failed to init mlx-whisper explicitly: {e}")
            return None, None

        if args.backend == "auto":
            transcriber = Transcriber(model_size=args.model)
        else:
            # Try to honour the requested backend
            if args.backend == "mlx":
                model_obj, trans_fn = _try_init_mlx(args.model)
                if model_obj is not None:
                    transcriber = Transcriber(model_size=args.model)
                    transcriber.backend = "mlx-whisper"
                    transcriber.model = model_obj
                    if trans_fn is not None:
                        transcriber._mlx_transcribe = trans_fn
                else:
                    # Fall back to generic initializer which will try other backends
                    transcriber = Transcriber(model_size=args.model)

            elif args.backend == "faster":
                if _have("faster_whisper"):
                    try:
                        from faster_whisper import WhisperModel  # type: ignore
                        device = os.environ.get("WHISPER_DEVICE", "cpu")
                        compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
                        model = WhisperModel(args.model, device=device, compute_type=compute_type)
                        transcriber = Transcriber(model_size=args.model)
                        transcriber.backend = "faster-whisper"
                        transcriber.model = model
                    except Exception as e:
                        warnings.warn(f"Failed to init faster-whisper explicitly: {e}")
                        transcriber = Transcriber(model_size=args.model)
                else:
                    transcriber = Transcriber(model_size=args.model)

            elif args.backend == "openai":
                if _have("whisper"):
                    try:
                        import whisper  # type: ignore
                        model = whisper.load_model(args.model)
                        transcriber = Transcriber(model_size=args.model)
                        transcriber.backend = "openai-whisper"
                        transcriber.model = model
                    except Exception as e:
                        warnings.warn(f"Failed to init openai-whisper explicitly: {e}")
                        transcriber = Transcriber(model_size=args.model)
                else:
                    transcriber = Transcriber(model_size=args.model)

            else:
                # Unknown explicit backend; fall back to auto behavior
                transcriber = Transcriber(model_size=args.model)

    if not args.no_transcribe and args.backend != "none" and (transcriber is None or not transcriber.available()):
        print(
            "Transcriber not available. Install mlx-whisper, faster-whisper, or openai-whisper to enable transcription.",
            file=sys.stderr,
        )

    print("\nProcessing segments and transcribing in real time…\n")
    success = 0
    fail = 0
    temp_kwargs = {"prefix": "segmentation_"}
    if args.workdir:
        os.makedirs(args.workdir, exist_ok=True)
        temp_kwargs["dir"] = args.workdir
    with tempfile.TemporaryDirectory(**temp_kwargs) as td:
        for idx, seg in enumerate(segments, 1):
            seg_wav = os.path.join(td, f"seg_{idx:03d}.wav")
            write_wav_segment(y, sr, seg, seg_wav)

            header = f"[{idx:02d}/{len(segments)}] {fmt_ts(seg.start)} -> {fmt_ts(seg.end)}"
            print(header, flush=True)
            transcript = ""
            lang_used: Optional[str] = None

            if args.no_transcribe or args.backend == "none":
                print("  -> skipped (no-transcribe)", flush=True)
            elif transcriber is not None and transcriber.available():
                # Try Hebrew first, then English
                try:
                    t_he = transcriber.transcribe_wav(seg_wav, language="he")
                except Exception as e:
                    t_he = ""
                    warnings.warn(f"Hebrew transcription failed: {e}")

                if t_he and t_he.strip():
                    transcript = t_he.strip()
                    lang_used = "he"
                else:
                    try:
                        t_en = transcriber.transcribe_wav(seg_wav, language="en")
                    except Exception as e:
                        t_en = ""
                        warnings.warn(f"English transcription failed: {e}")

                    if t_en and t_en.strip():
                        transcript = t_en.strip()
                        lang_used = "en"

            if not args.no_transcribe and transcript:
                success += 1
                tag = f"ok:{lang_used}" if lang_used else "ok"
                print(f"  -> {tag}: {transcript}", flush=True)
            elif not args.no_transcribe:
                fail += 1
                print("  -> transcription unavailable (install whisper backend)", flush=True)

    print("\nDone.")
    if success + fail:
        print(f"Transcribed: {success}, Unavailable: {fail}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
