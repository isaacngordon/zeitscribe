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


# New: audio analysis utility to help pick thresholds and min-speech settings
def analyze_audio(y: List[float], sr: int, frame_ms: float = 30.0, hop_ms: float = 10.0, width: int = 120, audio_path: Optional[str] = None) -> None:
    import numpy as np
    import math

    x = np.asarray(y, dtype=np.float32)
    if x.size == 0:
        print("Empty audio.")
        return

    frame_len = int(sr * frame_ms / 1000.0)
    hop_len = int(sr * hop_ms / 1000.0)
    if frame_len <= 0:
        frame_len = 1
    if hop_len <= 0:
        hop_len = 1

    num_frames = 1 + max(0, (len(x) - frame_len) // hop_len)
    if num_frames <= 0:
        num_frames = 1
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(num_frames, frame_len),
        strides=(x.strides[0] * hop_len, x.strides[0]),
        writeable=False,
    )
    energy_db = np.log10((frames ** 2).mean(axis=1) + 1e-10) * 10.0

    # Basic stats
    mean = float(np.mean(energy_db))
    median = float(np.median(energy_db))
    std = float(np.std(energy_db))
    p10, p25, p50, p75, p90 = [float(np.percentile(energy_db, p)) for p in (10, 25, 50, 75, 90)]

    duration = len(x) / float(sr)
    print(f"Duration: {duration:.2f}s | Frames: {len(energy_db)} | frame_ms={frame_ms} hop_ms={hop_ms}")
    print("Energy (dB-ish): mean={:.2f}  median={:.2f}  std={:.2f}".format(mean, median, std))
    print("Percentiles: 10%={:.2f}, 25%={:.2f}, 50%={:.2f}, 75%={:.2f}, 90%={:.2f}".format(p10, p25, p50, p75, p90))

    # Suggested simple thresholds
    suggestions = {
        "median-3": median - 3.0,
        "median-6": median - 6.0,
        "p25": p25,
        "p10": p10,
        "default(-35)": -35.0,
    }

    # Compute an Otsu-like threshold on energy to separate background/signal if possible
    def otsu_threshold(vals, bins=256):
        hist, edges = np.histogram(vals, bins=bins)
        hist = hist.astype(float)
        total = hist.sum()
        if total == 0:
            return float(np.median(vals))
        cum = np.cumsum(hist)
        cum_mean = np.cumsum(hist * (0.5 * (edges[:-1] + edges[1:])))
        sigma_b_max = -1.0
        best = edges[0]
        for i in range(len(hist)):
            if cum[i] == 0 or (total - cum[i]) == 0:
                continue
            mean1 = cum_mean[i] / cum[i]
            mean2 = (cum_mean[-1] - cum_mean[i]) / (total - cum[i])
            sigma_b = cum[i] * (total - cum[i]) * (mean1 - mean2) ** 2
            if sigma_b > sigma_b_max:
                sigma_b_max = sigma_b
                best = edges[i]
        return float(best)

    otsu = otsu_threshold(energy_db, bins=128)
    suggestions["otsu"] = otsu

    print("\nSuggested threshold candidates (dB-ish):")
    for k, v in suggestions.items():
        print(f"  {k:12s}: {v:.2f}")

    # Histogram
    bins = 60
    hist, edges = np.histogram(energy_db, bins=bins)
    max_count = hist.max() if hist.size else 1
    print("\nEnergy histogram:")
    for i in range(bins):
        left = edges[i]
        right = edges[i + 1]
        bar = "#" * int(round((hist[i] / max_count) * 50))
        print(f" {left:7.2f} -> {right:7.2f} | {bar}")

    # Sparkline over time
    print("\nEnergy over time (sparkline):")
    N = width
    if len(energy_db) < N:
        samples = energy_db
        sample_idx = np.arange(len(energy_db))
    else:
        sample_idx = np.linspace(0, len(energy_db) - 1, N).astype(int)
        samples = energy_db[sample_idx]

    chars = ' .:-=+*#%@'
    lo = float(np.min(energy_db))
    hi = float(np.max(energy_db))
    rng = hi - lo if hi > lo else 1.0
    line = ''.join(chars[int((s - lo) / rng * (len(chars) - 1))] for s in samples)
    print(line)

    # For a few candidate thresholds, compute voiced segment duration stats to help pick min_speech_ms
    print("\nSegment duration stats for candidate thresholds (useful for --min-speech-ms):")
    def seg_durations_for_thr(thr: float):
        voiced = energy_db > thr
        durations = []
        cur_frames = 0
        starts = []
        ends = []
        for i, v in enumerate(voiced):
            if v:
                if cur_frames == 0:
                    starts.append(i)
                cur_frames += 1
            else:
                if cur_frames > 0:
                    durations.append(cur_frames * hop_ms)
                    ends.append(i - 1)
                    cur_frames = 0
        if cur_frames > 0:
            durations.append(cur_frames * hop_ms)
            ends.append(len(voiced) - 1)
        return durations, starts, ends

    def summarize_durations(durations):
        if not durations:
            return (0, 0.0, 0.0, 0.0)
        arr = np.array(durations)
        return (len(arr), float(np.median(arr)), float(np.percentile(arr, 75)), float(np.percentile(arr, 90)))

    for name, thr in suggestions.items():
        durations, starts, ends = seg_durations_for_thr(thr)
        cnt, med_ms, p75_ms, p90_ms = summarize_durations(durations)
        coverage = (sum(durations) / (duration * 1000.0)) * 100.0 if duration > 0 else 0.0
        print(f"  {name:12s}: segments={cnt:4d}  median_ms={med_ms:6.1f}  75%_ms={p75_ms:6.1f}  90%_ms={p90_ms:6.1f}  coverage={coverage:5.1f}%")

    # SNR estimate: separate below/above median
    bg = energy_db[energy_db <= median]
    sig = energy_db[energy_db > median]
    if sig.size > 0 and bg.size > 0:
        snr_est = float(np.mean(sig)) - float(np.mean(bg))
    else:
        snr_est = 0.0
    print(f"\nEstimated SNR (mean(signal)-mean(background)) ~= {snr_est:.2f} dB-ish")

    # Suggest min_speech_ms and pad_ms based on observed durations
    # Use p25/p50/p75 of voiced durations at otsu threshold
    dur_otsu, starts_otsu, ends_otsu = seg_durations_for_thr(otsu)
    cnt_otsu, med_otsu, p75_otsu, p90_otsu = summarize_durations(dur_otsu)
    # Use 50% of 75th percentile or median as a reasonable min speech length
    suggested_min_speech_ms = float(max(50.0, min(1000.0, (p75_otsu * 0.5) if p75_otsu > 0 else med_otsu)))
    # pad: half of median silence length between voiced regions at otsu
    silence_lengths = []
    if starts_otsu:
        prev_end = -1
        for s, e in zip(starts_otsu, ends_otsu):
            if prev_end >= 0:
                gap_frames = s - prev_end - 1
                if gap_frames > 0:
                    silence_lengths.append(gap_frames * hop_ms)
            prev_end = e
    median_silence = float(np.median(silence_lengths)) if silence_lengths else 0.0
    suggested_pad_ms = float(min(500.0, max(0.0, median_silence * 0.5)))

    print(f"\nSuggested min_speech_ms ~ {suggested_min_speech_ms:.0f} ms  (based on Otsu segments)")
    print(f"Suggested pad_ms        ~ {suggested_pad_ms:.0f} ms  (approx half median silence between segments)")

    # Top longest voiced segments at Otsu
    if dur_otsu:
        durations_sorted = sorted([(d, s, e) for d, s, e in zip(dur_otsu, starts_otsu, ends_otsu)], reverse=True)
        print("\nTop longest voiced regions (Otsu threshold):")
        for i, (d, s, e) in enumerate(durations_sorted[:10], 1):
            t0 = s * hop_ms / 1000.0
            t1 = (e * hop_ms + frame_ms) / 1000.0
            print(f"  {i:2d}. {t0:6.2f}s -> {t1:6.2f}s  duration={d:6.1f}ms")

    # ASCII timeline overlay: mark positions above Otsu threshold
    print("\nASCII timeline (", width, "cols) — '.' low .. '#' high; '|' marks Otsu-voiced")
    chars = ' .:-=+*#%@'
    lo = float(np.min(energy_db))
    hi = float(np.max(energy_db))
    rng = hi - lo if hi > lo else 1.0
    timeline = []
    for j in range(width):
        idx = int(j * (len(energy_db) - 1) / (width - 1)) if width > 1 else 0
        v = energy_db[idx]
        ch = chars[int((v - lo) / rng * (len(chars) - 1))]
        timeline.append(ch)
    # overlay voiced markers
    voiced_bool = energy_db > otsu
    # map voiced frames to timeline positions
    pos_voiced = set()
    for i, vb in enumerate(voiced_bool):
        if vb:
            pos = int(i * (width - 1) / max(1, len(energy_db) - 1))
            pos_voiced.add(pos)
    line = ''.join('|' if i in pos_voiced else timeline[i] for i in range(len(timeline)))
    print(line)

    # Final runnable recommendation command (last line)
    audio_arg = f'"{audio_path}"' if audio_path else '"/path/to/audio"'
    cmd = (
        f"python segmentation.py {audio_arg} --threshold-db {otsu:.2f}"
        f" --min-speech-ms {int(round(suggested_min_speech_ms))} --pad-ms {int(round(suggested_pad_ms))}"
        f" --frame-ms {int(round(frame_ms))} --hop-ms {int(round(hop_ms))}"
    )
    print(cmd)


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
    parser.add_argument("--analyze", action="store_true", help="Analyze audio for visual feedback on energy levels and segment durations")
    args = parser.parse_args()

    audio_path = args.audio
    if not os.path.exists(audio_path):
        print(f"Error: File not found: {audio_path}", file=sys.stderr)
        return 2

    print("Loading audio…", flush=True)
    y, sr = load_audio_mono_float32(audio_path, target_sr=16000)
    duration = len(y) / float(sr)
    print(f"Loaded: {os.path.basename(audio_path)} | {sr} Hz | {duration:.2f}s")

    if args.analyze:
        print("Analyzing audio…", flush=True)
        analyze_audio(y, sr, frame_ms=args.frame_ms, hop_ms=args.hop_ms)
        print("\nDone.")
        return 0

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
                # Transcribe both Hebrew and English, then choose the best result via a simple heuristic
                t_he = ""
                t_en = ""
                try:
                    t_he = transcriber.transcribe_wav(seg_wav, language="he") or ""
                except Exception as e:
                    t_he = ""
                    warnings.warn(f"Hebrew transcription failed: {e}")

                try:
                    t_en = transcriber.transcribe_wav(seg_wav, language="en") or ""
                except Exception as e:
                    t_en = ""
                    warnings.warn(f"English transcription failed: {e}")

                t_he = t_he.strip()
                t_en = t_en.strip()

                def _score_text(s: str) -> int:
                    if not s:
                        return 0
                    words = len(s.split())
                    chars = len(s)
                    return words * 1000 + chars

                # Choose the higher-scoring non-empty transcription; if tie prefer Hebrew
                score_he = _score_text(t_he)
                score_en = _score_text(t_en)
                if score_he == 0 and score_en == 0:
                    transcript = ""
                    lang_used = None
                elif score_he >= score_en:
                    transcript = t_he
                    lang_used = "he" if t_he else "en"
                else:
                    transcript = t_en
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
