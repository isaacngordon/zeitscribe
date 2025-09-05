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
import html as py_html
import json
from datetime import datetime

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


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def slugify(value: str) -> str:
    value = ''.join(ch for ch in value if ch.isalnum() or ch in ('-', '_', '.', ' '))
    value = value.strip().replace(' ', '-')
    while '--' in value:
        value = value.replace('--', '-')
    return value or 'untitled'


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


def _load_audio(audio_path: str) -> Optional[AudioSegment]:
    try:
        return AudioSegment.from_file(audio_path)
    except Exception:
        return None


def segment_dbfs_from_segment(seg_all: AudioSegment, start: float, end: float) -> float:
    try:
        window = seg_all[int(start*1000):int(end*1000)]
        val = window.dBFS
        if val is None:
            return -100.0
        if val == float("inf") or val == float("-inf"):
            # treat +/- inf as extreme silence/loudness
            return -100.0 if val < 0 else 0.0
        return float(val)
    except Exception:
        return -100.0


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
_FW_TXT_MODEL = None  # cached small model for text fill


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


def get_fw_text_model(model_size: str = "small"):
    """Cached faster-whisper model for on-demand text fill (defaults to 'small')."""
    global _FW_TXT_MODEL
    if _FW_TXT_MODEL is not None:
        return _FW_TXT_MODEL
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception as e:
        raise RuntimeError("faster-whisper required for text fill is not available") from e
    device = "cuda"
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    _FW_TXT_MODEL = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _FW_TXT_MODEL


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
                    prob = max(prob, 0.75)
            else:
                # If script agrees with predicted, boost confidence a bit
                s_code = dominant_script_lang(agg)
                if s_code == code and s_code in allowed_langs:
                    prob = max(prob, 0.85)
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


def export_audio_window(audio_path: str, start: float, end: float) -> Optional[str]:
    """Export [start,end] seconds to a temporary wav file. Returns path or None."""
    try:
        seg = AudioSegment.from_file(audio_path)[int(start*1000):int(end*1000)]
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        seg.export(tmp.name, format="wav")
        tmp.close()
        return tmp.name
    except Exception:
        return None


def transcribe_window_fill(audio_path: str, start: float, end: float, lang_code: Optional[str] = None) -> Tuple[str, List[Dict]]:
    """Lightweight on-demand transcription for a short window to fill missing text.
    Returns (text, words). Uses faster-whisper small model by default.
    """
    tmp = export_audio_window(audio_path, start, end)
    if not tmp:
        return "", []
    try:
        model = get_fw_text_model("small")
    except Exception:
        try:
            model = get_fw_model("tiny")
        except Exception:
            os.unlink(tmp)
            return "", []
    try:
        segments, info = model.transcribe(
            tmp,
            task="transcribe",
            vad_filter=False,
            beam_size=3,
            word_timestamps=True,
            language=lang_code,
        )
        words: List[Dict] = []
        parts: List[str] = []
        for seg in segments:
            stxt = (getattr(seg, "text", "") or "").strip()
            if stxt:
                parts.append(stxt)
            if hasattr(seg, "words") and seg.words:
                for w in seg.words:
                    wt = (w.word or "").strip()
                    if not wt:
                        continue
                    words.append({
                        "start": float(start + (w.start or 0.0)),
                        "end": float(start + (w.end or 0.0)),
                        "text": wt,
                    })
        text = " ".join(parts).strip()
        return text, words
    except Exception:
        return "", []
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


def _counts_from_tokens(tokens: List[str], allowed_langs: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for t in tokens:
        t = (t or "").strip()
        if not t:
            continue
        code = classify_word_lang(t, allowed_langs)
        counts[code] = counts.get(code, 0) + 1
    return counts


def _counts_to_probs(counts: Dict[str, int], allowed_langs: List[str]) -> Dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        return {code: 0.0 for code in allowed_langs}
    probs = {code: counts.get(code, 0) / total for code in allowed_langs}
    # normalize any missing codes
    for code in allowed_langs:
        probs.setdefault(code, 0.0)
    return probs


def assemble_segment_texts(words: List[Dict], segments: List[Dict]) -> List[str]:
    """Build text per segment by concatenating words overlapping that segment."""
    out: List[str] = []
    # Pre-sort words by start
    ws = sorted([
        {"start": float(w.get("start", 0.0)), "end": float(w.get("end", 0.0)), "text": (w.get("text") or "").strip()}
        for w in words if (w.get("text") or "").strip()
    ], key=lambda x: (x["start"], x["end"]))
    for seg in segments:
        s = float(seg.get("start", 0.0))
        e = float(seg.get("end", s))
        toks: List[str] = []
        for w in ws:
            if w["end"] <= s:
                continue
            if w["start"] >= e:
                break
            toks.append(w["text"])
        out.append(" ".join(toks).strip())
    return out


def render_segment_transcripts_html(segments: List[Dict], texts: List[str]) -> str:
    def fmt_time(secs: float) -> str:
        s = max(0.0, float(secs))
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sec = s % 60
        return f"{h:02d}:{m:02d}:{sec:06.3f}"

    blocks: List[str] = []
    for i, seg in enumerate(segments):
        start = fmt_time(float(seg.get("start", 0.0)))
        end = fmt_time(float(seg.get("end", 0.0)))
        lang = seg.get("language", "und")
        prob = float(seg.get("confidence", 0.0))
        source = seg.get("source", "")
        text = texts[i] if i < len(texts) else ""
        esc_text = py_html.escape(text) if text else ""
        summary = f"#{i} [{start} - {end}] lang={lang} prob={prob:.2f} src={source}"
        blocks.append(
            f"<details><summary>{py_html.escape(summary)}</summary>"
            f"<div style='padding:8px 0; white-space:pre-wrap; line-height:1.6;'>{esc_text or '<em>No words in this interval</em>'}</div>"
            f"</details>"
        )
    return "\n".join(blocks)


def render_audio_player_html(audio_path: Optional[str], start: Optional[float] = None, end: Optional[float] = None, autoplay: bool = False) -> str:
    if not audio_path:
        return "<em>No audio loaded</em>"
    s = 0.0 if start is None else float(start)
    e = -1.0 if end is None else float(end)
    auto = 'true' if autoplay else 'false'
    # Use Gradio's file proxy via 'file=' prefix
    return f"""
<div>
  <audio id="seg-player" src="file={py_html.escape(audio_path)}" controls style="width:100%"></audio>
  <div style="font-size:0.9rem;color:#555;">Segment: {s:.3f}s → {('∞' if e<0 else f'{e:.3f}s')}</div>
</div>
<script>
(function(){{
  try {{
    const a = document.getElementById('seg-player');
    const s = {s:.3f};
    const e = {e:.3f};
    const autoplay = {auto};
    if (!a) return;
    if (autoplay) {{
      a.currentTime = Math.max(0, s - 0.02);
      a.play().catch(()=>{{}});
      if (e > 0 && e > s) {{
        const handler = () => {{
          if (a.currentTime >= e) {{
            a.pause();
            a.removeEventListener('timeupdate', handler);
          }}
        }};
        a.addEventListener('timeupdate', handler);
      }}
    }}
  }} catch(e) {{}}
}})();
</script>
"""


def process_audio(audio_path: str, allowed_lang_names: List[str], backend: str = "auto"):
    if not audio_path:
        return "Please upload an audio file.", [], ""

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
    # Build text per segment from words (best-effort; may be empty)
    texts = assemble_segment_texts(words, segs_full)

    # Fill missing text for speech segments via short window decode
    audio_seg = _load_audio(audio_path)
    for i, seg in enumerate(segs_full):
        if seg.get("language") in ("silence", "und"):
            continue
        if not texts[i]:
            fill_text, fill_words = transcribe_window_fill(audio_path, float(seg["start"]), float(seg["end"]), lang_code=seg.get("language"))
            if fill_text:
                texts[i] = fill_text
                # incorporate words into global word list to improve downstream counts
                words.extend(fill_words)
                # bump confidence based on script agreement
                s_code = dominant_script_lang(fill_text)
                if s_code == seg.get("language"):
                    seg["confidence"] = max(float(seg.get("confidence", 0.0)), 0.85)
                    seg["source"] = (seg.get("source", "") + "+fill").strip("+")

    # Post re-check: correct 'silence' mislabeled speech and vice-versa using RMS and text
    silence_dbfs_thresh = -45.0  # segments louder than this are likely speech
    for i, seg in enumerate(segs_full):
        s, e = float(seg["start"]), float(seg["end"])
        dbfs = segment_dbfs_from_segment(audio_seg, s, e) if audio_seg else -100.0
        has_text = bool(texts[i])
        if seg.get("language") == "silence":
            if has_text or dbfs > silence_dbfs_thresh:
                # Try assign language from text/words first, then LID
                # Build probs from available evidence
                # Reuse probs_for_segment logic by inlining minimal version here
                # words overlapping this seg
                w_here = [w for w in words if float(w.get("end", 0.0)) > s and float(w.get("start", 0.0)) < e]
                new_label = None
                if w_here:
                    counts = _counts_from_tokens([(w.get("text") or "").strip() for w in w_here], allowed_codes)
                    if counts:
                        new_label = max(counts.items(), key=lambda kv: kv[1])[0]
                if not new_label and has_text:
                    counts = _counts_from_tokens(texts[i].split(), allowed_codes)
                    if counts:
                        new_label = max(counts.items(), key=lambda kv: kv[1])[0]
                if not new_label:
                    lid_code, lid_prob = lid_language_for_window(audio_path, s, e, allowed_codes, min_window_s=1.0, prob_accept=0.6)
                    if lid_code != "und":
                        new_label = lid_code
                        seg["confidence"] = max(float(seg.get("confidence", 0.0)), float(lid_prob or 0.6))
                if new_label:
                    seg["language"] = new_label
                    seg["source"] = (seg.get("source", "") + "+relabel").strip("+")
        else:
            # Non-silence but empty and very quiet → relabel to silence
            if (not has_text) and dbfs <= (silence_dbfs_thresh - 5.0):
                seg["language"] = "silence"
                seg["confidence"] = 1.0
                seg["source"] = (seg.get("source", "") + "+relabel").strip("+")

    # Compute per-segment probability distribution across allowed languages
    def probs_for_segment(seg: Dict, idx: int) -> Dict[str, float]:
        s, e = float(seg["start"]), float(seg["end"])
        # words overlapping this seg
        w_here = [w for w in words if float(w.get("end", 0.0)) > s and float(w.get("start", 0.0)) < e]
        if w_here:
            counts = _counts_from_tokens([(w.get("text") or "").strip() for w in w_here], allowed_codes)
            return _counts_to_probs(counts, allowed_codes)
        # fallback: use filled text if any
        text = texts[idx] if idx < len(texts) else ""
        if text:
            tokens = text.split()
            counts = _counts_from_tokens(tokens, allowed_codes)
            return _counts_to_probs(counts, allowed_codes)
        # else: use single-label confidence
        label = seg.get("language", "und")
        conf = float(seg.get("confidence", 0.0) or 0.0)
        probs = {code: 0.0 for code in allowed_codes}
        if label in probs:
            probs[label] = conf
        return probs

    # Optional calibration biases from previous feedback
    def load_calibration_biases(allowed: List[str]) -> Dict[str, float]:
        path = os.path.join("outputs", "feedback", "calibration.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            bias = obj.get("lang_bias", {})
            return {k: float(bias.get(k, 1.0)) for k in allowed}
        except Exception:
            return {k: 1.0 for k in allowed}

    def apply_bias(pmap: Dict[str, float], bias: Dict[str, float]) -> Dict[str, float]:
        out = {}
        for k, v in pmap.items():
            out[k] = max(0.0, float(v)) * float(bias.get(k, 1.0))
        s = sum(out.values()) or 0.0
        if s > 0:
            for k in list(out.keys()):
                out[k] = out[k] / s
        return out

    # Table rows with index, text, predicted prob, and per-language probs as compact string
    rows = []
    bias = load_calibration_biases(allowed_codes)
    # Preload audio segment for dBFS
    audio_seg = _load_audio(audio_path)
    for i, s in enumerate(segs_full):
        pmap = probs_for_segment(s, i)
        pmap = apply_bias(pmap, bias) if bias else pmap
        # predicted language prob
        pred_code = s.get("language", "und")
        pred_prob = float(pmap.get(pred_code, s.get("confidence", 0.0) or 0.0))
        probs_str = "; ".join(f"{k}={pmap.get(k, 0.0):.2f}" for k in allowed_codes)
        # Flags and energy
        dbfs = segment_dbfs_from_segment(audio_seg, float(s["start"]), float(s["end"])) if audio_seg else -100.0
        has_text = bool(texts[i])
        # quick overlap words check
        w_here = [w for w in words if float(w.get("end", 0.0)) > float(s["start"]) and float(w.get("start", 0.0)) < float(s["end"])]
        has_words = bool(w_here)
        flags = []
        if pred_code == "silence" and (has_text or dbfs > -45.0):
            flags.append("silence->speech?")
        if pred_code != "silence" and (not has_text) and dbfs <= -50.0:
            flags.append("speech->silence?")
        rows.append([
            i,
            round(float(s["start"]), 3),
            round(float(s["end"]), 3),
            pred_code,
            round(pred_prob, 3),
            texts[i],
            probs_str,
            s.get("source", ""),
            round(dbfs, 1),
            has_words,
            has_text,
            ", ".join(flags),
        ])

    html = render_segment_transcripts_html(segs_full, texts)
    status = f"Segments: {len(rows)} covering {duration:.2f}s. Words: {len(words)}."
    return status, rows, html


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
        player = gr.HTML(label="Player", value="")
        seg_df = gr.Dataframe(
            headers=["idx", "start", "end", "language", "prob", "text", "probs", "source", "dBFS", "has_words", "has_text", "flags"],
            value=[],
            row_count=0,
            col_count=(12, "fixed"),
            interactive=False,
        )
        # States for reinforcement
        segs_state = gr.State(value=None)
        words_state = gr.State(value=None)
        texts_state = gr.State(value=None)
        allowed_state = gr.State(value=None)
        audio_path_state = gr.State(value=None)
        feedback_state = gr.State(value=[])
        bias_state = gr.State(value=None)

        with gr.Row():
            sel_idx = gr.Number(label="Segment index", value=0, precision=0)
            corrected = gr.Dropdown(label="Corrected language", choices=[code for _, code in DISPLAY_LANGS] + ["silence", "und"], value=None)
            grade = gr.Radio(label="Grade", choices=["good", "bad"], value="good")
        with gr.Row():
            apply_btn = gr.Button("Apply correction locally")
            save_btn = gr.Button("Save feedback")
            export_truth_btn = gr.Button("Export truth JSON")
            calibrate_btn = gr.Button("Calibrate biases from feedback")
            apply_calib_btn = gr.Button("Apply calibration to labels")

        def _run(audio_path, lang_names, be):
            s, rows, _html = process_audio(audio_path, lang_names, backend=be)
            player_html = render_audio_player_html(audio_path, None, None, autoplay=False)
            # Initialize states from rows; rows act as our working table
            return s, player_html, rows, rows, lang_names, audio_path, [], None

        def _on_select(evt, rows, audio_path_v):
            try:
                idx = getattr(evt, 'index', None)
                if not idx:
                    return gr.update()
                r, c = idx
                # Only act when clicking start/end columns
                if c not in (1, 2):
                    return gr.update()
                s_val = float(rows[r][1])
                e_val = float(rows[r][2])
                return render_audio_player_html(audio_path_v, s_val, e_val, autoplay=True)
            except Exception:
                return gr.update()

        def _apply(idx, corr_lang, grade_val, rows, lang_names, audio_path_v, feedback, _bias):
            try:
                i = int(idx)
            except Exception:
                return rows, rows, feedback
            if not rows or i < 0 or i >= len(rows):
                return rows, rows, feedback
            # Row format: [idx,start,end,language,prob,text,probs,source,dBFS,has_words,has_text,flags]
            row = rows[i]
            # Apply correction to language column if provided
            if corr_lang:
                row[3] = corr_lang
                row[7] = (row[7] + "+man").strip("+")
            # Append feedback entry
            fb = {
                "idx": int(row[0]),
                "start": float(row[1]),
                "end": float(row[2]),
                "pred_language": str(row[3]),
                "correct_language": str(corr_lang) if corr_lang else None,
                "grade": grade_val,
                "prob": float(row[4]),
                "text": str(row[5] or ""),
                "probs": str(row[6] or ""),
                "source": str(row[7] or ""),
                "dBFS": float(row[8] or -100.0),
                "has_words": bool(row[9]),
                "has_text": bool(row[10]),
                "flags": str(row[11] or ""),
                "audio_path": audio_path_v,
                "allowed": [NAME_TO_CODE.get(n, n) for n in (lang_names or [])],
                "ts": datetime.utcnow().isoformat() + "Z",
            }
            feedback = list(feedback or []) + [fb]
            return rows, rows, feedback

        def _save_feedback(rows, lang_names, audio_path_v, feedback, _bias):
            if not feedback:
                return "No feedback to save.", rows, rows
            out_dir = os.path.join("outputs", "feedback")
            ensure_dir(out_dir)
            base = os.path.basename(audio_path_v) if audio_path_v else "audio"
            slug = slugify(os.path.splitext(base)[0])
            ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            path = os.path.join(out_dir, f"{slug}-{ts}.json")
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({"audio": audio_path_v, "entries": feedback}, f, ensure_ascii=False, indent=2)
                return f"Saved feedback → {path}", rows, rows
            except Exception as e:
                return f"Failed to save feedback: {e}", rows, rows

        def _export_truth(rows, lang_names, audio_path_v, feedback, _bias):
            # Export current table's start/end/language as truth
            truth = [{"start": float(r[1]), "end": float(r[2]), "language": str(r[3])} for r in (rows or [])]
            out_dir = os.path.join("outputs", "truth")
            ensure_dir(out_dir)
            base = os.path.basename(audio_path_v) if audio_path_v else "audio"
            slug = slugify(os.path.splitext(base)[0])
            path = os.path.join(out_dir, f"{slug}.truth.json")
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(truth, f, ensure_ascii=False, indent=2)
                return f"Exported truth → {path}", rows, rows
            except Exception as e:
                return f"Failed to export truth: {e}", rows, rows

        def _calibrate(rows, lang_names, audio_path_v, feedback, _bias):
            # Aggregate all feedback files and compute simple per-language biases
            out_dir = os.path.join("outputs", "feedback")
            ensure_dir(out_dir)
            files = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith('.json')]
            good_counts = {}
            bad_counts = {}
            for fp in files:
                try:
                    with open(fp, 'r', encoding='utf-8') as f:
                        obj = json.load(f)
                    for e in obj.get('entries', []):
                        lang = str(e.get('correct_language') or e.get('pred_language') or 'und')
                        if str(e.get('grade')) == 'good':
                            good_counts[lang] = good_counts.get(lang, 0) + 1
                        else:
                            bad_counts[lang] = bad_counts.get(lang, 0) + 1
                except Exception:
                    continue
            allowed = [NAME_TO_CODE.get(n, n) for n in (lang_names or [])] or ["en","he"]
            bias = {}
            for code in allowed + ["silence","und"]:
                g = good_counts.get(code, 0)
                b = bad_counts.get(code, 0)
                bias[code] = float((g + 1) / (b + 1))  # Laplace-smoothed ratio
            save_path = os.path.join(out_dir, "calibration.json")
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump({"lang_bias": bias, "updated_at": datetime.utcnow().isoformat()+"Z"}, f, ensure_ascii=False, indent=2)
                return f"Saved calibration → {save_path}", bias
            except Exception as e:
                return f"Failed to save calibration: {e}", None

        def _apply_calibration(rows, lang_names, audio_path_v, feedback, bias):
            if not rows:
                return rows, rows
            if not isinstance(bias, dict):
                # try load
                try:
                    with open(os.path.join('outputs','feedback','calibration.json'),'r',encoding='utf-8') as f:
                        bias = json.load(f).get('lang_bias', {})
                except Exception:
                    bias = {}
            allowed = [NAME_TO_CODE.get(n, n) for n in (lang_names or [])]
            # Update table language if biased probs point elsewhere significantly
            updated = []
            for r in rows:
                # r structure: [idx,start,end,language,prob,text,probs,source,dBFS,has_words,has_text,flags]
                pmap = {}
                # parse probs string back
                try:
                    for part in str(r[6]).split(';'):
                        part = part.strip()
                        if not part:
                            continue
                        k, v = part.split('=')
                        pmap[k.strip()] = float(v.strip())
                except Exception:
                    pmap = {}
                # apply bias
                for k in list(pmap.keys()):
                    pmap[k] = pmap[k] * float(bias.get(k, 1.0))
                ssum = sum(pmap.values()) or 0.0
                if ssum > 0:
                    for k in list(pmap.keys()):
                        pmap[k] = pmap[k] / ssum
                # choose argmax
                if pmap:
                    new_lang = max(pmap.items(), key=lambda kv: kv[1])[0]
                    new_prob = pmap.get(new_lang, r[4])
                    if new_lang != r[3] and (new_prob - float(r[4])) >= 0.10:
                        r[3] = new_lang
                        r[4] = round(float(new_prob), 3)
                        r[7] = (str(r[7]) + "+calib").strip('+')
                updated.append(r)
            return updated, updated

        go.click(_run, inputs=[audio, langs, backend], outputs=[status, player, seg_df, segs_state, allowed_state, audio_path_state, feedback_state, bias_state])
        seg_df.select(_on_select, inputs=[seg_df, audio_path_state], outputs=[player])
        apply_btn.click(_apply, inputs=[sel_idx, corrected, grade, segs_state, allowed_state, audio_path_state, feedback_state, bias_state], outputs=[seg_df, segs_state, feedback_state])
        save_btn.click(_save_feedback, inputs=[segs_state, allowed_state, audio_path_state, feedback_state, bias_state], outputs=[status, seg_df, segs_state])
        export_truth_btn.click(_export_truth, inputs=[segs_state, allowed_state, audio_path_state, feedback_state, bias_state], outputs=[status, seg_df, segs_state])
        calibrate_btn.click(_calibrate, inputs=[segs_state, allowed_state, audio_path_state, feedback_state, bias_state], outputs=[status, bias_state])
        apply_calib_btn.click(_apply_calibration, inputs=[segs_state, allowed_state, audio_path_state, feedback_state, bias_state], outputs=[seg_df, segs_state])
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
