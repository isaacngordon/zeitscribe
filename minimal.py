"""
Minimal Gradio app: drop in audio and detect language segments.

Outputs a simple list of (start, end, language) tuples.

Backends:
- Prefers MLX Whisper on Apple Silicon if available
- Falls back to faster-whisper otherwise
"""

from typing import List, Dict, Tuple, Optional
import os
import logging
import math
import tempfile

import gradio as gr
import langid
from pydub import AudioSegment
from pydub.silence import detect_nonsilent


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Display languages for dropdown (name, code)
DISPLAY_LANGS: List[Tuple[str, str]] = [
    ("English", "en"),
    ("Hebrew", "he"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Italian", "it"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
    ("Arabic", "ar"),
    ("Chinese (zh)", "zh"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Dutch", "nl"),
    ("Polish", "pl"),
]
NAME_TO_CODE = {name: code for name, code in DISPLAY_LANGS}


def transcribe_words_mlx(audio_path: str, mlx_model_repo: str = "mlx-community/whisper-large-v3-mlx") -> List[Dict]:
    """Use MLX Whisper (GPU-accelerated on Apple Silicon) to get word-level timestamps.
    Returns list of words: {start, end, text}
    """
    try:
        import mlx_whisper  # type: ignore
    except Exception as e:
        raise RuntimeError("MLX Whisper not available") from e

    out = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=mlx_model_repo,
        word_timestamps=True,
        task="transcribe",
        language=None,
    )

    words: List[Dict] = []
    for seg in out.get("segments", []) or []:
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))
        seg_words = seg.get("words", []) or []
        if seg_words:
            for w in seg_words:
                t = (w.get("word") or "").strip()
                if not t:
                    continue
                words.append({
                    "start": float(w.get("start", seg_start)),
                    "end": float(w.get("end", seg_end)),
                    "text": t,
                })
        else:
            # Fallback: evenly divide the segment across tokens
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            tokens = text.split()
            dur = max(0.0, seg_end - seg_start)
            step = dur / len(tokens) if tokens else dur
            for i, tok in enumerate(tokens):
                s = seg_start + i * step
                e = seg_start + (i + 1) * step
                words.append({"start": float(s), "end": float(e), "text": tok})
    return words


def transcribe_words_faster_whisper(audio_path: str, model_size: str = "small", beam_size: int = 5) -> List[Dict]:
    """Use faster-whisper to get word-level timestamps. Returns list of words: {start, end, text}"""
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception as e:
        raise RuntimeError("faster-whisper not available") from e

    # Try CUDA if available, else CPU
    device = "cuda"
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments, info = model.transcribe(
        audio_path,
        task="transcribe",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        beam_size=int(beam_size),
        word_timestamps=True,
        language=None,
    )

    words: List[Dict] = []
    for seg in segments:
        if hasattr(seg, "words") and seg.words:
            for w in seg.words:
                wt = (w.word or "").strip()
                if not wt:
                    continue
                words.append({
                    "start": float(w.start or seg.start),
                    "end": float(w.end or seg.end),
                    "text": wt,
                })
        else:
            tokens = (seg.text or "").strip().split()
            if not tokens:
                continue
            dur = max(0.0, float(seg.end - seg.start))
            step = dur / len(tokens) if tokens else dur
            for i, tok in enumerate(tokens):
                s = float(seg.start + i * step)
                e = float(seg.start + (i + 1) * step)
                words.append({"start": s, "end": e, "text": tok})
    return words


# -------------------------------
# Audio utils and VAD
# -------------------------------

def audio_duration_sec(audio_path: str) -> float:
    try:
        seg = AudioSegment.from_file(audio_path)
        return max(0.0, len(seg) / 1000.0)
    except Exception:
        return 0.0


def nonsilent_intervals_sec(
    audio_path: str,
    min_silence_len_ms: int = 400,
    silence_thresh_offset_db: float = 16.0,
) -> List[Tuple[float, float]]:
    try:
        seg = AudioSegment.from_file(audio_path)
        # dBFS can be None in some formats; default threshold to -35 dBFS
        base = seg.dBFS if seg.dBFS is not None else -35.0
        thresh = base - float(silence_thresh_offset_db)
        ns = detect_nonsilent(seg, min_silence_len=min_silence_len_ms, silence_thresh=thresh)
        return [(start/1000.0, end/1000.0) for start, end in ns]
    except Exception:
        return []


def union_boundaries(
    duration: float,
    words: List[Dict],
    vad_intervals: List[Tuple[float, float]],
) -> List[float]:
    points = {0.0, float(duration)}
    for w in words:
        s = float(w.get("start", 0.0))
        e = float(w.get("end", s))
        points.add(max(0.0, s))
        points.add(max(0.0, min(e, duration)))
    for s, e in vad_intervals:
        points.add(max(0.0, s))
        points.add(max(0.0, min(e, duration)))
    pts = sorted(p for p in points if 0.0 <= p <= duration)
    # ensure strictly increasing
    cleaned = []
    for p in pts:
        if not cleaned or abs(p - cleaned[-1]) > 1e-6:
            cleaned.append(p)
    return cleaned


def build_micro_intervals(boundaries: List[float]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for i in range(len(boundaries) - 1):
        s = float(boundaries[i])
        e = float(boundaries[i+1])
        if e - s <= 1e-6:
            continue
        out.append((s, e))
    return out


# -------------------------------
# Acoustic LID via faster-whisper tiny (cached)
# -------------------------------

_FW_LID_MODEL = None  # cached tiny model instance


def get_fw_model(model_size: str = "tiny"):
    global _FW_LID_MODEL
    if _FW_LID_MODEL is not None:
        return _FW_LID_MODEL
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception as e:
        raise RuntimeError("faster-whisper required for acoustic LID is not available") from e
    device = "cuda"
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    _FW_LID_MODEL = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _FW_LID_MODEL


def lid_language_for_window(
    audio_path: str,
    start: float,
    end: float,
    allowed_langs: List[str],
    min_window_s: float = 1.0,
    prob_accept: float = 0.6,
) -> Tuple[str, float]:
    dur = audio_duration_sec(audio_path)
    if dur <= 0.0:
        return "und", 0.0
    # Ensure at least min_window_s by expanding around center
    length = max(0.0, end - start)
    if length < min_window_s:
        center = (start + end) / 2.0
        half = min_window_s / 2.0
        start = max(0.0, center - half)
        end = min(dur, center + half)
        if end - start < min_window_s:
            # stick to end
            start = max(0.0, end - min_window_s)
    # Export window to a temp wav
    try:
        seg = AudioSegment.from_file(audio_path)[int(start*1000):int(end*1000)]
    except Exception:
        return "und", 0.0
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            seg.export(tmp.name, format="wav")
            try:
                model = get_fw_model("tiny")
            except Exception:
                return "und", 0.0
            segments, info = model.transcribe(
                tmp.name,
                task="transcribe",
                vad_filter=False,
                beam_size=1,
                word_timestamps=False,
                language=None,
            )
            # Aggregate some text if available for script vote
            txt_parts = []
            try:
                for s in segments:
                    t = (getattr(s, "text", None) or "").strip()
                    if t:
                        txt_parts.append(t)
            except Exception:
                pass
            agg = " ".join(txt_parts)
            code = getattr(info, "language", None) or "und"
            prob = float(getattr(info, "language_probability", 0.0) or 0.0)
            # Restrict to allowed langs when possible
            if allowed_langs and code not in allowed_langs:
                # try script-based from decoded text
                s_code = dominant_script_lang(agg)
                if s_code in allowed_langs:
                    code = s_code
                    prob = max(prob, 0.6)
            if prob < prob_accept and code not in allowed_langs:
                return "und", prob
            return code, prob
    except Exception:
        return "und", 0.0


def setup_langid(allowed_langs: List[str]):
    if allowed_langs:
        try:
            langid.set_languages(allowed_langs)
        except Exception:
            pass


def dominant_script_lang(text: str) -> str:
    """Simple script-based classification for common cases (Hebrew vs Latin)."""
    if not text:
        return "und"
    heb = 0
    lat = 0
    for ch in text:
        o = ord(ch)
        if 0x0590 <= o <= 0x05FF:
            heb += 1
        elif ('a' <= ch.lower() <= 'z'):
            lat += 1
    if heb == 0 and lat == 0:
        return "und"
    if heb > 0 and lat == 0:
        return "he"
    if lat > 0 and heb == 0:
        return "en"
    return "he" if heb >= lat else "en"


def classify_word_lang(word: str, allowed_langs: List[str]) -> str:
    # Prefer script-based detection for robustness with Whisper output
    s_lang = dominant_script_lang(word)
    if s_lang != "und":
        return s_lang
    # Fallback to langid for ambiguous tokens
    if not word.strip():
        return "und"
    code, _ = langid.classify(word)
    return code


def label_full_timeline(
    audio_path: str,
    words: List[Dict],
    allowed_langs: List[str],
    min_silence_len_ms: int = 400,
    silence_thresh_offset_db: float = 16.0,
    lid_window_s: float = 1.0,
    lid_prob_threshold: float = 0.6,
    bridge_gap_s: float = 0.3,
) -> Tuple[List[Dict], float]:
    """Return full-coverage segments [{start,end,language,confidence,source}] and audio duration.
    Languages include actual codes plus 'silence' and 'und'.
    """
    duration = audio_duration_sec(audio_path)
    setup_langid(allowed_langs)
    vad = nonsilent_intervals_sec(audio_path, min_silence_len_ms, silence_thresh_offset_db)
    boundaries = union_boundaries(duration, words, vad)
    micro = build_micro_intervals(boundaries)

    # Helper to test if interval overlaps VAD
    def overlaps_vad(s: float, e: float) -> bool:
        for vs, ve in vad:
            if ve <= s or vs >= e:
                continue
            return True
        return False

    # Helper: words overlapping
    def words_in(s: float, e: float) -> List[Dict]:
        out = []
        for w in words:
            ws = float(w.get("start", 0.0))
            we = float(w.get("end", ws))
            if we <= s or ws >= e:
                continue
            out.append(w)
        return out

    labeled: List[Dict] = []
    for s, e in micro:
        if not overlaps_vad(s, e):
            labeled.append({"start": s, "end": e, "language": "silence", "confidence": 1.0, "source": "vad"})
            continue
        w_here = words_in(s, e)
        if w_here:
            counts: Dict[str, int] = {}
            total = 0
            for w in w_here:
                txt = (w.get("text") or "").strip()
                if not txt:
                    continue
                lang = classify_word_lang(txt, allowed_langs)
                counts[lang] = counts.get(lang, 0) + 1
                total += 1
            if total > 0:
                # majority vote
                lang = max(counts.items(), key=lambda kv: kv[1])[0]
                conf = counts[lang] / max(1, total)
                labeled.append({"start": s, "end": e, "language": lang, "confidence": float(conf), "source": "words"})
                continue
        # No words → try acoustic LID
        lang, prob = lid_language_for_window(audio_path, s, e, allowed_langs, min_window_s=lid_window_s, prob_accept=lid_prob_threshold)
        if lang == "und":
            labeled.append({"start": s, "end": e, "language": "und", "confidence": float(prob), "source": "lid"})
        else:
            labeled.append({"start": s, "end": e, "language": lang, "confidence": float(prob or 0.6), "source": "lid"})

    # Merge contiguous same labels
    merged: List[Dict] = []
    for seg in labeled:
        if not merged:
            merged.append(dict(seg))
            continue
        last = merged[-1]
        if seg["language"] == last["language"]:
            # blend confidence (time-weighted) using previous last end
            prev_end = last["end"]
            dur_last = prev_end - last["start"]
            dur_seg = seg["end"] - seg["start"]
            total = dur_last + dur_seg
            last["end"] = seg["end"]
            if total > 0:
                last["confidence"] = (last.get("confidence", 0.0)*dur_last + seg.get("confidence", 0.0)*dur_seg) / total
        else:
            merged.append(dict(seg))

    # Bridge short gaps of silence/und between same-language neighbors
    def bridge(segments: List[Dict]) -> List[Dict]:
        out: List[Dict] = []
        i = 0
        n = len(segments)
        while i < n:
            if i+2 < n:
                a, b, c = segments[i], segments[i+1], segments[i+2]
                if b["language"] in ("silence", "und") and (b["end"] - b["start"]) <= bridge_gap_s and a["language"] == c["language"] and a["language"] not in ("silence", "und"):
                    # merge a + b + c into one of language a
                    new_conf = min(a.get("confidence", 0.8), c.get("confidence", 0.8)) * 0.9
                    out.append({"start": a["start"], "end": c["end"], "language": a["language"], "confidence": new_conf, "source": "bridge"})
                    i += 3
                    continue
            out.append(segments[i])
            i += 1
        return out

    bridged = bridge(merged)
    # Final tidy merge
    final: List[Dict] = []
    for seg in bridged:
        if not final:
            final.append(seg)
            continue
        if seg["language"] == final[-1]["language"] and abs(seg["start"] - final[-1]["end"]) < 1e-6:
            final[-1]["end"] = seg["end"]
            # keep max confidence for simplicity
            final[-1]["confidence"] = max(final[-1].get("confidence", 0.0), seg.get("confidence", 0.0))
        else:
            final.append(seg)

    # Ensure exact coverage [0, duration]
    if not final:
        final = [{"start": 0.0, "end": duration, "language": "und", "confidence": 0.0, "source": "none"}]
    else:
        if final[0]["start"] > 0.0:
            final.insert(0, {"start": 0.0, "end": final[0]["start"], "language": "silence", "confidence": 1.0, "source": "pad"})
        if final[-1]["end"] < duration:
            final.append({"start": final[-1]["end"], "end": duration, "language": "silence", "confidence": 1.0, "source": "pad"})

    return final, duration


def process_audio(audio_path: str, allowed_lang_names: List[str], backend: str = "auto"):
    if not audio_path:
        return "Please upload an audio file.", []

    # Convert names → codes
    allowed_codes = [NAME_TO_CODE.get(n, n) for n in (allowed_lang_names or [])]
    # If none selected, allow a small default set
    if not allowed_codes:
        allowed_codes = ["en", "he"]

    # Transcribe words
    words: List[Dict] = []
    err = None
    if backend == "mlx" or backend == "auto":
        try:
            words = transcribe_words_mlx(audio_path)
        except Exception as e:
            if backend == "mlx":
                err = f"MLX backend failed: {e}"
            else:
                logger.info(f"MLX backend not used: {e}")
    if not words and (backend == "faster" or backend == "auto"):
        try:
            words = transcribe_words_faster_whisper(audio_path, model_size="small", beam_size=5)
        except Exception as e:
            if err is None:
                err = f"faster-whisper backend failed: {e}"
    # Build full coverage segments (works even with empty words)
    segs_full, duration = label_full_timeline(
        audio_path,
        words,
        allowed_codes,
        min_silence_len_ms=400,
        silence_thresh_offset_db=16.0,
        lid_window_s=1.0,
        lid_prob_threshold=0.6,
        bridge_gap_s=0.3,
    )
    tuples = [(round(s["start"], 3), round(s["end"], 3), s["language"]) for s in segs_full]
    status = f"Segments: {len(tuples)} covering {duration:.2f}s. Words: {len(words)}."
    return status, tuples


def build_ui():
    with gr.Blocks(title="Language Segments") as demo:
        gr.Markdown("# 🎧 Language Segments\nDrop audio and detect (start, end, language) segments.")
        with gr.Row():
            audio = gr.Audio(label="Drop audio here", type="filepath")
            with gr.Column():
                langs = gr.Dropdown(
                    choices=[name for name, _ in DISPLAY_LANGS],
                    value=["English", "Hebrew"],
                    multiselect=True,
                    label="Languages to consider",
                )
                backend = gr.Dropdown(
                    choices=["auto", "mlx", "faster"],
                    value="auto",
                    label="Backend",
                    info="Use MLX on Apple Silicon if available; otherwise faster-whisper.",
                )
                go = gr.Button("Process segments", variant="primary")

        status = gr.Textbox(label="Status", interactive=False)
        seg_df = gr.Dataframe(headers=["start", "end", "language"], value=[], row_count=0, col_count=(3, "fixed"), interactive=False)

        def _run(audio_path, lang_names, be):
            s, tuples = process_audio(audio_path, lang_names, backend=be)
            return s, tuples

        go.click(_run, inputs=[audio, langs, backend], outputs=[status, seg_df])
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
