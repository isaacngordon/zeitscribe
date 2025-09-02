#!/usr/bin/env python3
"""
Multilingual Transcriber (GPU-aware for macOS Apple Silicon)

- Uses MLX + mlx-whisper on Apple Silicon (arm64 macOS) to leverage the Mac GPU.
- Falls back to faster-whisper elsewhere (CPU or NVIDIA CUDA if available).
- Produces N-second chunked JSON files with the requested schema.
- Gradio UI: upload audio, select languages, chunk size, and destination folder.
  - Consolidated transcript shows secondary languages in colors.
  - Audio playback highlights the active segment and supports click-to-seek.

Run:
  python main.py

Dependencies are listed in requirements.txt (see environment markers for MLX).
FFmpeg is recommended to support diverse audio formats (install system-wide).
"""

import os
import logging
import re
import json
import math
import time
import shutil
import string
import unicodedata
import platform
import html
from datetime import datetime
from typing import List, Dict

import gradio as gr
import numpy as np
from tqdm import tqdm
import langid

# Optional imports done lazily in code paths:
# - faster_whisper
# - mlx_whisper
# - torch (only to detect CUDA availability; not required)

# -------------------------------
# Logging setup
# -------------------------------
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Platform / backend detection
# -------------------------------
IS_APPLE_SILICON = (platform.system() == "Darwin" and platform.machine() == "arm64")

# -------------------------------
# Utilities
# -------------------------------

DISPLAY_LANGS = [
    ("English", "en"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Italian", "it"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
    ("Arabic", "ar"),
    ("Hebrew", "he"),
    ("Yiddish", "yi"),
    ("Chinese (zh)", "zh"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Dutch", "nl"),
    ("Polish", "pl"),
]

NAME_TO_CODE = {name: code for name, code in DISPLAY_LANGS}

def slugify(value: str, allow_unicode: bool=False) -> str:
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = ''.join(ch for ch in value if ch in f"-_.() {string.ascii_letters}{string.digits}")
    value = value.strip().replace(' ', '-')
    while '--' in value:
        value = value.replace('--', '-')
    return value.lower() or "untitled"

def fmt_time(secs: float) -> str:
    if secs is None:
        return "00:00:00.000"
    s = max(0.0, float(secs))
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}"

def words_count(text: str) -> int:
    return len(text.strip().split())

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def pick_colors(additional_lang_codes: List[str]) -> Dict[str, str]:
    """Assign readable, unique colors (HSL) for secondary languages."""
    if not additional_lang_codes:
        return {}
    colors = {}
    n = len(additional_lang_codes)
    for i, code in enumerate(additional_lang_codes):
        hue = int((360 * i) / max(1, n))
        colors[code] = f"hsl({hue}, 65%, 35%)"
    return colors

def clip_segment_to_window(seg: Dict, win_start: float, win_end: float) -> Dict:
    """Return a shallow copy of seg trimmed to [win_start, win_end]."""
    s = max(seg["start"], win_start)
    e = min(seg["end"], win_end)
    body = seg["body"]
    return {"start": s, "end": e, "body": body, "language": seg["language"]}

def html_escape(s: str) -> str:
    return html.escape(s, quote=True)

# -------------------------------
# Language detection (per small phrase)
# -------------------------------

def setup_langid(allowed_langs: List[str]):
    if allowed_langs:
        try:
            langid.set_languages(allowed_langs)
        except Exception:
            pass

def detect_lang(text: str, allowed_langs: List[str]) -> str:
    if not text.strip():
        return allowed_langs[0] if allowed_langs else "und"
    code, _ = langid.classify(text)
    if allowed_langs and code not in allowed_langs:
        return allowed_langs[0]
    return code

# -------------------------------
# Transcription paths
# -------------------------------

def transcribe_words_faster_whisper(audio_path: str, model_size: str, beam_size: int = 5):
    """Use faster-whisper to get word-level timestamps."""
    from faster_whisper import WhisperModel
    # Try CUDA if available, else CPU
    device = "cuda"
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"
    compute = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute)

    segments, info = model.transcribe(
        audio_path,
        task="transcribe",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        beam_size=beam_size,
        word_timestamps=True,
        language=None,
    )
    words = []
    for seg in segments:
        if hasattr(seg, "words") and seg.words:
            for w in seg.words:
                wt = (w.word or "").strip()
                if wt:
                    words.append({
                        "start": float(w.start or seg.start),
                        "end": float(w.end or seg.end),
                        "text": wt
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

def transcribe_words_mlx(audio_path: str, mlx_model_repo: str = "mlx-community/whisper-large-v3-mlx"):
    """Use MLX whisper on Apple Silicon (GPU-accelerated)."""
    import mlx_whisper  # type: ignore
    out = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=mlx_model_repo,
        word_timestamps=True
    )
    words = []
    for seg in out.get("segments", []):
        for w in seg.get("words", []):
            t = (w.get("word") or "").strip()
            if t:
                words.append({
                    "start": float(w.get("start", seg.get("start", 0.0))),
                    "end": float(w.get("end", seg.get("end", 0.0))),
                    "text": t
                })
    return words

# -------------------------------
# Build mono-language segments
# -------------------------------

def build_mono_language_segments(words: List[Dict], allowed_langs: List[str]) -> List[Dict]:
    """
    Group contiguous words predicted to be the same language into segments:
    Each: {"start": float, "end": float, "body": str, "language": "<code>"}
    """
    setup_langid(allowed_langs)
    segments = []
    curr = None

    phrase_words = []
    phrase_start = None
    max_phrase_span = 1.2  # seconds

    def flush_phrase():
        nonlocal phrase_words, phrase_start, curr, segments
        if not phrase_words:
            return
        text = " ".join(w["text"] for w in phrase_words).strip()
        p_start = phrase_start
        p_end = phrase_words[-1]["end"]
        lang = detect_lang(text, allowed_langs) or (allowed_langs[0] if allowed_langs else "und")

        if curr is None:
            curr = {"start": p_start, "end": p_end, "body": text, "language": lang}
        else:
            if curr["language"] == lang:
                curr["end"] = p_end
                curr["body"] = (curr["body"] + " " + text).strip()
            else:
                segments.append(curr)
                curr = {"start": p_start, "end": p_end, "body": text, "language": lang}
        phrase_words = []
        phrase_start = None

    for w in words:
        if phrase_start is None:
            phrase_start = w["start"]
        phrase_words.append(w)
        if (w["end"] - phrase_start) >= max_phrase_span:
            flush_phrase()

    flush_phrase()
    if curr is not None:
        segments.append(curr)

    # Clean small overlaps
    for i in range(1, len(segments)):
        if segments[i]["start"] < segments[i-1]["end"]:
            segments[i]["start"] = segments[i-1]["end"]

    return segments

# -------------------------------
# Chunking and saving
# -------------------------------

def split_into_chunks(segments: List[Dict], chunk_sec: float) -> List[Dict]:
    if not segments:
        return []
    end_time = max(seg["end"] for seg in segments)
    n_chunks = int(math.ceil(end_time / chunk_sec))
    chunks = []

    for idx in range(n_chunks):
        win_start = idx * chunk_sec
        win_end = min((idx + 1) * chunk_sec, end_time)
        segs_in = []
        for seg in segments:
            if seg["end"] > win_start and seg["start"] < win_end:
                s = max(seg["start"], win_start)
                e = min(seg["end"], win_end)
                segs_in.append({"start": s, "end": e, "body": seg["body"], "language": seg["language"]})

        length_words = sum(words_count(s["body"]) for s in segs_in)
        chunks.append({
            "index": idx,
            "start_time": fmt_time(win_start),
            "end_time": fmt_time(win_end),
            "length_time": round(win_end - win_start, 3),
            "length_words": length_words,
            "num_segments": len(segs_in),
            "segments": segs_in,
        })
    return chunks

def save_chunk_json(chunk: Dict, out_dir: str, base_name: str, src_path: str) -> str:
    idx = chunk["index"]
    st = chunk["start_time"].replace(":", "").replace(".", "")
    et = chunk["end_time"].replace(":", "").replace(".", "")
    fname = f"{base_name}_{idx:04d}_{st}-{et}.json"
    fpath = os.path.join(out_dir, fname)
    payload = {
        "index": idx,
        "start time": chunk["start_time"],
        "end time": chunk["end_time"],
        "length_time": chunk["length_time"],
        "length_words": chunk["length_words"],
        "num_segments": chunk["num_segments"],
        "segments": chunk["segments"],
        "source_file": src_path,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return fpath

def copy_source(audio_tmp_path: str, project_dir: str) -> str:
    ensure_dir(project_dir)
    base = os.path.basename(audio_tmp_path)
    safe = slugify(os.path.splitext(base)[0]) + os.path.splitext(base)[1]
    dst = os.path.join(project_dir, safe)
    try:
        shutil.copy2(audio_tmp_path, dst)
    except Exception:
        shutil.copy(audio_tmp_path, dst)
    return dst

# -------------------------------
# HTML rendering (transcript + player)
# -------------------------------

def render_consolidated_html(segments: List[Dict],
                             primary_code: str,
                             additional_codes: List[str],
                             audio_src_path: str) -> str:
    color_map = pick_colors(additional_codes)
    spans = []
    for i, seg in enumerate(segments):
        lang = seg["language"]
        style = ""
        if lang in color_map:
            style = f"style='color:{color_map[lang]};'"
        spans.append(
            f"<span class='seg' data-i='{i}' data-start='{seg['start']:.3f}' "
            f"data-end='{seg['end']:.3f}' data-lang='{lang}' {style}>"
            f"{html_escape(seg['body'])}</span>"
        )

    legend_bits = "".join([f"&nbsp;&nbsp;<span style='color:{c};'>● {lc}</span>" for lc, c in color_map.items()])

    html_doc = f"""
<div id="player-wrap" style="position:sticky;top:0;background:#fff;z-index:10;padding:8px 0;">
  <audio id="audio" src="file={audio_src_path}" controls style="width:100%"></audio>
</div>

<div id="legend" style="margin:6px 0 12px 0;">
  <strong>Legend:</strong>
  <span>Primary ({primary_code}) = default color</span>
  {legend_bits}
</div>

<div id="transcript" style="line-height:1.8; font-size:1rem; max-height:60vh; overflow:auto; border:1px solid #eee; padding:12px;">
  {" ".join(spans)}
</div>

<style>
  .seg {{
    cursor: pointer;
    padding: 1px 2px;
    border-radius: 3px;
  }}
  .seg.active {{
    background: rgba(255, 235, 59, 0.35);
    outline: 1px solid rgba(255, 235, 59, 0.9);
  }}
</style>

<script>
(function() {{
  const audio = document.getElementById("audio");
  const container = document.getElementById("transcript");
  const spans = Array.from(container.querySelectorAll(".seg"));
  let activeIndex = -1;
  function setActive(idx) {{
    if (idx === activeIndex) return;
    if (activeIndex >= 0) spans[activeIndex]?.classList.remove("active");
    activeIndex = idx;
    if (activeIndex >= 0) {{
      const el = spans[activeIndex];
      el.classList.add("active");
      const top = el.offsetTop - container.clientHeight*0.3;
      container.scrollTo({{top: top, behavior: 'smooth'}});
    }}
  }}
  function updateActive() {{
    const t = audio.currentTime || 0;
    for (let i = 0; i < spans.length; i++) {{
      const s = parseFloat(spans[i].dataset.start);
      const e = parseFloat(spans[i].dataset.end);
      if (t >= s && t < e) {{
        setActive(i);
        return;
      }}
    }}
  }}
  audio.addEventListener("timeupdate", updateActive);
  spans.forEach((el, i) => {{
    el.addEventListener("click", () => {{
      const s = parseFloat(el.dataset.start);
      audio.currentTime = Math.max(0, s - 0.02);
      audio.play();
      setActive(i);
    }});
  }});
}})();
</script>
"""
    return html_doc

# -------------------------------
# Gradio app logic
# -------------------------------

def run_transcription(
    audio_file,
    primary_lang_name,
    addtl_lang_names,
    chunk_length_sec,
    model_choice,
    beam_size,
    project_name,
    root_save_dir,
):
    """
    Gradio generator: yields (status, list_of_chunk_paths, consolidated_html)
    """
    logger.info("Starting transcription process.")
    if audio_file is None:
        logger.warning("No audio file uploaded.")
        yield "Please upload an audio file.", [], ""
        return

    if not project_name.strip():
        project_name = "project"
    logger.debug(f"Project name: {project_name}")
    project_slug = slugify(project_name)
    logger.debug(f"Project slug: {project_slug}")
    root_dir = root_save_dir.strip() or "./outputs"
    session_root = os.path.abspath(root_dir)
    project_dir = os.path.join(session_root, project_slug)
    chunks_dir = os.path.join(project_dir, "chunks")
    ensure_dir(chunks_dir)
    logger.info(f"Chunks directory ensured at: {chunks_dir}")

    primary_code = NAME_TO_CODE.get(primary_lang_name, "en")
    addtl_codes = [NAME_TO_CODE.get(n, None) for n in (addtl_lang_names or [])]
    addtl_codes = [c for c in addtl_codes if c]
    allowed = [primary_code] + [c for c in addtl_codes if c != primary_code]

    # Copy audio to project dir for stable serving & archive
    src_audio_path = copy_source(audio_file, project_dir)
    logger.info(f"Audio file copied to: {src_audio_path}")

    # Backend selection
    requested = (model_choice or "auto").lower()
    use_mlx = IS_APPLE_SILICON and (requested in ("auto","mlx-large-v3","mlx-large-v3-turbo"))
    if requested.startswith("mlx-"):
        use_mlx = True

    if use_mlx:
        status = "Using MLX backend on Apple Silicon GPU..."
        logger.info(status)
        yield status, [], ""
        mlx_repo = "mlx-community/whisper-large-v3-mlx"
        if requested == "mlx-large-v3-turbo":
            mlx_repo = "mlx-community/whisper-large-v3-turbo"
        words = transcribe_words_mlx(src_audio_path, mlx_model_repo=mlx_repo)
    else:
        status = "Using faster-whisper (CUDA if available, else CPU)..."
        yield status, [], ""
        fw_model = requested if requested not in ("auto","mlx-large-v3","mlx-large-v3-turbo") else "medium"
        words = transcribe_words_faster_whisper(src_audio_path, model_size=fw_model, beam_size=int(beam_size))

    status = f"Detected {len(words)} words. Building mono-language segments..."
    logger.info(status)
    logger.info(status)
    yield status, [], ""
    ml_segments = build_mono_language_segments(words, allowed)

    # Save consolidated raw segments (optional convenience file)
    with open(os.path.join(project_dir, "transcript_segments.json"), "w", encoding="utf-8") as f:
        json.dump(ml_segments, f, ensure_ascii=False, indent=2)

    status = f"Chunking into {chunk_length_sec:.1f}s windows and saving JSON..."
    yield status, [], ""
    chunks = split_into_chunks(ml_segments, float(chunk_length_sec))
    base_name = slugify(os.path.splitext(os.path.basename(src_audio_path))[0])
    saved_paths = []

    for ch in tqdm(chunks):
        fpath = save_chunk_json(ch, chunks_dir, base_name, src_audio_path)
        saved_paths.append(fpath)
        status_now = (f"Saved chunk {ch['index']+1}/{len(chunks)} → {os.path.basename(fpath)}")
        html_now = render_consolidated_html(ml_segments, primary_code, addtl_codes, src_audio_path)
        logger.debug(status_now)
        yield status_now, [[p] for p in saved_paths], html_now

    status = f"Done. {len(saved_paths)} chunk files saved in {chunks_dir}"
    full_html = render_consolidated_html(ml_segments, primary_code, addtl_codes, src_audio_path)
    logger.info(status)
    yield status, [[p] for p in saved_paths], full_html

def build_ui():
    default_primary = "English"
    with gr.Blocks(title="Multilingual Transcriber", fill_height=True) as demo:
        gr.Markdown("# 🎧 Multilingual Transcriber\nUpload an audio file with mixed languages. Get chunked JSON outputs + a colorized, clickable transcript with audio playback.")

        with gr.Row():
            audio = gr.Audio(label="Upload audio", type="filepath")
            with gr.Column():
                primary = gr.Dropdown(
                    choices=[name for name, _ in DISPLAY_LANGS],
                    value=default_primary,
                    label="Primary language",
                )
                addtl = gr.Dropdown(
                    choices=[name for name, _ in DISPLAY_LANGS],
                    value=["Hebrew", "Yiddish"],
                    multiselect=True,
                    label="Additional languages (multiselect)",
                )
                chunk_len = gr.Slider(5, 180, value=30, step=5, label="Chunk length (seconds)")
                model_choice = gr.Dropdown(
                    choices=[
                        "auto",                # smart default
                        "mlx-large-v3",        # Apple Silicon GPU via MLX
                        "mlx-large-v3-turbo",  # MLX turbo variant
                        "tiny","base","small","medium","large-v3"  # faster-whisper models
                    ],
                    value="auto",
                    label="ASR model / backend",
                    info="On Apple Silicon, 'auto' uses MLX on GPU; elsewhere, faster-whisper."
                )
                beam = gr.Slider(1, 10, value=5, step=1, label="Beam size (faster-whisper only)")
                project = gr.Textbox(label="Project name", value="demo_project")
                root_dir = gr.Textbox(label="Root save directory (server path)", value="./outputs")

        transcribe_btn = gr.Button("Transcribe", variant="primary")

        with gr.Row():
            status = gr.Textbox(label="Status", value="", interactive=False)
        with gr.Row():
            files = gr.Dataframe(
                headers=["Saved Chunk JSON Paths"],
                value=[],
                row_count=0,
                col_count=(1, "fixed"),
                wrap=True,
                interactive=False,
                label="Chunk files (updates live)"
            )

        transcript_panel = gr.HTML(value="", label="Consolidated Transcript", elem_id="consolidated")

        def _wrap_runner(audio_file, primary_lang_name, addtl_lang_names, chunk_length_sec, model_choice_v, beam_size, project_name, root_save_dir):
            gen = run_transcription(
                audio_file,
                primary_lang_name,
                addtl_lang_names,
                chunk_length_sec,
                model_choice_v,
                int(beam_size),
                project_name,
                root_save_dir,
            )
            for s, paths, html_val in gen:
                yield s, paths, html_val

        transcribe_btn.click(
            _wrap_runner,
            inputs=[audio, primary, addtl, chunk_len, model_choice, beam, project, root_dir],
            outputs=[status, files, transcript_panel],
        )
    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
