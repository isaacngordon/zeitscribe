"""
Benchmark language segmentation pipeline.

Usage examples:
  python benchmark.py --input data --ext .wav .mp3 --langs en he --backend auto
  python benchmark.py --input data/sample.wav --langs en he

Analyze historical runs:
  python benchmark.py analyze --root outputs/benchmarks [--out outputs/benchmarks/analysis] [--no-csv]

Ground truth (optional):
  If a JSON or CSV with the same stem exists next to the audio file,
  it will be used as reference. Supported:
    - JSON: list of {"start": float, "end": float, "language": str}
    - JSON: object with key "segments" of the same shape
    - CSV header: start,end,language

Outputs a summary JSON under outputs/benchmarks/<timestamp>/summary.json
and prints per-file metrics.
"""

from __future__ import annotations

import os
import sys
import json
import csv
import glob
import math
import time
import argparse
from typing import List, Dict, Tuple, Optional
from pydub import AudioSegment

from minimal import (
    transcribe_words_mlx,
    transcribe_words_faster_whisper,
    label_full_timeline,
    audio_duration_sec,
)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_truth_segments(path: str) -> Optional[List[Dict]]:
    stem, ext = os.path.splitext(path)
    # Accept <stem>.truth.json / .json / .csv
    candidates = [
        f"{stem}.truth.json",
        f"{stem}.json",
        f"{stem}.csv",
    ]
    for c in candidates:
        if os.path.exists(c):
            if c.endswith(".json"):
                try:
                    with open(c, "r", encoding="utf-8") as f:
                        obj = json.load(f)
                    if isinstance(obj, list):
                        return [
                            {"start": float(x["start"]), "end": float(x["end"]), "language": str(x["language"]).strip()}
                            for x in obj
                            if isinstance(x, dict) and {"start","end","language"}.issubset(x.keys())
                        ]
                    if isinstance(obj, dict) and "segments" in obj and isinstance(obj["segments"], list):
                        return [
                            {"start": float(x["start"]), "end": float(x["end"]), "language": str(x["language"]).strip()}
                            for x in obj["segments"]
                            if isinstance(x, dict) and {"start","end","language"}.issubset(x.keys())
                        ]
                except Exception:
                    pass
            if c.endswith(".csv"):
                try:
                    rows = []
                    with open(c, newline="", encoding="utf-8") as f:
                        r = csv.DictReader(f)
                        for row in r:
                            rows.append({
                                "start": float(row.get("start", 0.0)),
                                "end": float(row.get("end", 0.0)),
                                "language": str(row.get("language", "und")).strip()
                            })
                    return rows or None
                except Exception:
                    pass
    return None


def intersect_timelines(a: List[Dict], b: List[Dict]) -> List[Tuple[float, float, str, str]]:
    out: List[Tuple[float, float, str, str]] = []
    i = j = 0
    while i < len(a) and j < len(b):
        s1, e1, l1 = float(a[i]["start"]), float(a[i]["end"]), str(a[i]["language"]) 
        s2, e2, l2 = float(b[j]["start"]), float(b[j]["end"]), str(b[j]["language"]) 
        s = max(s1, s2)
        e = min(e1, e2)
        if e > s:
            out.append((s, e, l1, l2))
        if e1 <= e2:
            i += 1
        else:
            j += 1
    return out


def time_weighted_accuracy(pred: List[Dict], truth: List[Dict]) -> Dict:
    if not pred or not truth:
        return {"acc_all": None, "acc_speech": None, "silence_acc": None}
    inter = intersect_timelines(pred, truth)
    total = sum(e - s for s, e, _, _ in inter)
    if total <= 0:
        return {"acc_all": None, "acc_speech": None, "silence_acc": None}
    correct = sum((e - s) for s, e, lp, lt in inter if lp == lt)
    # Speech-only accuracy (ignore segments labelled silence in truth)
    speech_total = sum((e - s) for s, e, lp, lt in inter if lt != "silence")
    speech_correct = sum((e - s) for s, e, lp, lt in inter if lt != "silence" and lp == lt)
    silence_total = sum((e - s) for s, e, lp, lt in inter if lt == "silence")
    silence_correct = sum((e - s) for s, e, lp, lt in inter if lt == "silence" and lp == lt)
    return {
        "acc_all": correct / total if total > 0 else None,
        "acc_speech": speech_correct / speech_total if speech_total > 0 else None,
        "silence_acc": silence_correct / silence_total if silence_total > 0 else None,
    }


def segment_dbfs(seg_all: AudioSegment, start: float, end: float) -> float:
    try:
        w = seg_all[int(start*1000):int(end*1000)]
        val = w.dBFS
        if val is None:
            return -100.0
        if val == float('inf') or val == float('-inf'):
            return -100.0 if val < 0 else 0.0
        return float(val)
    except Exception:
        return -100.0


def run_file(audio_path: str, allowed_langs: List[str], backend: str, silence_dbfs_thresh: float = -45.0) -> Dict:
    # Words via chosen backend
    words: List[Dict] = []
    if backend in ("mlx", "auto"):
        try:
            words = transcribe_words_mlx(audio_path)
        except Exception:
            words = []
    if not words and backend in ("faster", "auto"):
        try:
            words = transcribe_words_faster_whisper(audio_path, model_size="small", beam_size=5)
        except Exception:
            words = []

    segs, dur = label_full_timeline(
        audio_path,
        words,
        allowed_langs=allowed_langs,
        min_silence_len_ms=400,
        silence_thresh_offset_db=16.0,
        lid_window_s=1.0,
        lid_prob_threshold=0.6,
        bridge_gap_s=0.3,
    )
    # Metrics
    total = sum(s["end"] - s["start"] for s in segs)
    speech_time = sum(s["end"] - s["start"] for s in segs if s["language"] not in ("silence",))
    und_time = sum(s["end"] - s["start"] for s in segs if s["language"] == "und")
    avg_conf_speech = 0.0
    if speech_time > 0:
        avg_conf_speech = sum((s["end"] - s["start"]) * float(s.get("confidence", 0.0)) for s in segs if s["language"] not in ("silence",)) / speech_time
    coverage_ok = abs(total - dur) < 1e-3 and segs and abs(segs[0]["start"]) < 1e-6 and abs(segs[-1]["end"] - dur) < 1e-6

    # Silence/speech mismatch metrics
    audio_seg = AudioSegment.from_file(audio_path)
    # Fast lookup: overlapping words
    def has_words_in(seg):
        s, e = float(seg["start"]), float(seg["end"])
        for w in words:
            ws = float(w.get("start", 0.0))
            we = float(w.get("end", ws))
            if we > s and ws < e:
                return True
        return False

    silence_but_words = 0.0
    silence_but_loud = 0.0
    speech_but_quiet_and_empty = 0.0
    for seg in segs:
        s, e = float(seg["start"]), float(seg["end"])
        dur_seg = max(0.0, e - s)
        db = segment_dbfs(audio_seg, s, e)
        is_sil = (seg.get("language") == "silence")
        has_w = has_words_in(seg)
        if is_sil:
            if has_w:
                silence_but_words += dur_seg
            if db > silence_dbfs_thresh:
                silence_but_loud += dur_seg
        else:
            if (not has_w) and db <= (silence_dbfs_thresh - 5.0):
                speech_but_quiet_and_empty += dur_seg

    sil_words_frac = (silence_but_words / dur) if dur > 0 else None
    sil_loud_frac = (silence_but_loud / dur) if dur > 0 else None
    speech_quiet_empty_frac = (speech_but_quiet_and_empty / dur) if dur > 0 else None

    metrics = {
        "file": audio_path,
        "duration": dur,
        "coverage_ok": coverage_ok,
        "num_segments": len(segs),
        "speech_fraction": (speech_time / dur) if dur > 0 else None,
        "und_fraction": (und_time / dur) if dur > 0 else None,
        "avg_conf_speech": avg_conf_speech,
        "silence_not_empty_frac": sil_words_frac,  # silence label but words present
        "silence_loud_frac": sil_loud_frac,        # silence label but high energy
        "speech_quiet_empty_frac": speech_quiet_empty_frac,  # speech label but no words and quiet
        "silence_dbfs_threshold": silence_dbfs_thresh,
    }

    truth = load_truth_segments(audio_path)
    if truth:
        accs = time_weighted_accuracy(segs, truth)
        metrics.update(accs)
    return metrics


def find_audio_files(inp: str, exts: List[str]) -> List[str]:
    paths: List[str] = []
    if os.path.isdir(inp):
        for ext in exts:
            paths.extend(glob.glob(os.path.join(inp, f"**/*{ext}"), recursive=True))
    elif os.path.isfile(inp):
        paths = [inp]
    return sorted(paths)


def run_main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark language segmentation (run)")
    parser.add_argument("--input", required=True, help="Audio file or directory")
    parser.add_argument("--ext", nargs="*", default=[".wav", ".mp3", ".m4a", ".flac"], help="Extensions when input is a directory")
    parser.add_argument("--langs", nargs="+", default=["en", "he"], help="Allowed languages (codes)")
    parser.add_argument("--backend", choices=["auto","mlx","faster"], default="auto")
    parser.add_argument("--silence-dbfs", type=float, default=-45.0, help="dBFS threshold above which a segment is considered loud/speech-like")
    args = parser.parse_args(argv)

    files = find_audio_files(args.input, args.ext)
    if not files:
        print("No audio files found.")
        return 1

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join("outputs", "benchmarks", ts)
    ensure_dir(out_dir)

    all_metrics = []
    for i, fpath in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {os.path.basename(fpath)}")
        m = run_file(fpath, allowed_langs=args.langs, backend=args.backend, silence_dbfs_thresh=args.silence_dbfs)
        all_metrics.append(m)
        # Print quick line
        msg = (
            f" dur={m['duration']:.2f}s cover={m['coverage_ok']} "
            f"segments={m['num_segments']} speech={m['speech_fraction']:.2f} "
            f"und={m['und_fraction']:.2f} conf={m['avg_conf_speech']:.2f} "
            f"sil-words={m['silence_not_empty_frac'] if m['silence_not_empty_frac'] is not None else '-'} "
            f"sil-loud={m['silence_loud_frac'] if m['silence_loud_frac'] is not None else '-'} "
            f"sp-quiet={m['speech_quiet_empty_frac'] if m['speech_quiet_empty_frac'] is not None else '-'}"
        )
        if 'acc_all' in m and m['acc_all'] is not None:
            msg += f" acc_all={m['acc_all']:.3f}"
        if 'acc_speech' in m and m['acc_speech'] is not None:
            msg += f" acc_speech={m['acc_speech']:.3f}"
        print(msg)

    # Aggregate summary
    n = len(all_metrics)
    def avg(key):
        vals = [m[key] for m in all_metrics if m.get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None
    # Weighted by duration helper
    tot_dur = sum(m['duration'] for m in all_metrics if m.get('duration')) or 0.0
    def wavg(key):
        if tot_dur <= 0:
            return None
        s = 0.0
        for m in all_metrics:
            d = m.get('duration') or 0.0
            v = m.get(key)
            if v is None:
                continue
            s += v * d
        return (s / tot_dur) if tot_dur > 0 else None

    summary = {
        "count": n,
        "coverage_ok_frac": sum(1 for m in all_metrics if m.get("coverage_ok")) / n,
        "avg_segments": avg("num_segments"),
        "avg_speech_fraction": avg("speech_fraction"),
        "avg_und_fraction": avg("und_fraction"),
        "avg_conf_speech": avg("avg_conf_speech"),
        "avg_acc_all": avg("acc_all"),
        "avg_acc_speech": avg("acc_speech"),
        # New mismatch metrics (time-weighted across files)
        "silence_not_empty_frac": wavg("silence_not_empty_frac"),
        "silence_loud_frac": wavg("silence_loud_frac"),
        "speech_quiet_empty_frac": wavg("speech_quiet_empty_frac"),
        "silence_dbfs_threshold": args.silence_dbfs,
    }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "files": all_metrics}, f, ensure_ascii=False, indent=2)
    print(f"Saved benchmark summary to {out_dir}/summary.json")
    return 0


def _safe_avg(vals: List[Optional[float]]) -> Optional[float]:
    nums = [v for v in vals if isinstance(v, (int,float))]
    return (sum(nums) / len(nums)) if nums else None


def analyze_all(root: str, outdir: Optional[str] = None, write_csv: bool = True) -> int:
    if not os.path.isdir(root):
        print(f"Root not found: {root}")
        return 1
    run_dirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    runs = []
    for rd in run_dirs:
        summ = os.path.join(root, rd, "summary.json")
        if os.path.exists(summ):
            try:
                with open(summ, "r", encoding="utf-8") as f:
                    data = json.load(f)
                runs.append({"run_id": rd, "path": summ, "data": data})
            except Exception:
                pass
    if not runs:
        print("No benchmark summaries found.")
        return 1

    # Aggregate across runs (per-file level and per-run summaries)
    all_file_rows: List[Dict] = []
    per_run_summary: List[Dict] = []
    for r in runs:
        d = r["data"] or {}
        summ = d.get("summary", {})
        per_run_summary.append({"run_id": r["run_id"], **summ})
        files = d.get("files", []) or []
        for m in files:
            all_file_rows.append({"run_id": r["run_id"], **m})

    # Overall averages across all files
    def col(key):
        return [row.get(key) for row in all_file_rows if key in row]

    grand = {
        "runs": len(runs),
        "files": len(all_file_rows),
        "coverage_ok_frac": (sum(1 for r in all_file_rows if r.get("coverage_ok")) / len(all_file_rows)) if all_file_rows else None,
        "avg_duration": _safe_avg(col("duration")),
        "avg_segments": _safe_avg(col("num_segments")),
        "avg_speech_fraction": _safe_avg(col("speech_fraction")),
        "avg_und_fraction": _safe_avg(col("und_fraction")),
        "avg_conf_speech": _safe_avg(col("avg_conf_speech")),
        "avg_acc_all": _safe_avg(col("acc_all")),
        "avg_acc_speech": _safe_avg(col("acc_speech")),
        "avg_silence_acc": _safe_avg(col("silence_acc")),
    }

    # Output
    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = outdir or os.path.join(root, "analysis")
    ensure_dir(outdir)
    out_json = os.path.join(outdir, f"{ts}-analysis.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"grand_summary": grand, "runs": per_run_summary, "files": all_file_rows}, f, ensure_ascii=False, indent=2)
    print(f"Saved analysis to {out_json}")

    if write_csv:
        # Per-run CSV
        runs_csv = os.path.join(outdir, f"{ts}-runs.csv")
        run_keys = sorted({k for r in per_run_summary for k in r.keys()})
        with open(runs_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=run_keys)
            w.writeheader()
            for r in per_run_summary:
                w.writerow(r)
        print(f"Saved {runs_csv}")

        # All files CSV
        files_csv = os.path.join(outdir, f"{ts}-files.csv")
        file_keys = sorted({k for r in all_file_rows for k in r.keys()})
        with open(files_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=file_keys)
            w.writeheader()
            for r in all_file_rows:
                w.writerow(r)
        print(f"Saved {files_csv}")

    # Quick console summary
    def fmt(v):
        return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    print("Overall across runs:")
    print(
        " files=", grand["files"],
        " coverage_ok_frac=", fmt(grand["coverage_ok_frac"]),
        " avg_acc_speech=", fmt(grand["avg_acc_speech"]),
        " avg_speech_fraction=", fmt(grand["avg_speech_fraction"]),
        " avg_und_fraction=", fmt(grand["avg_und_fraction"]),
        " sil_not_empty=", fmt(grand.get("silence_not_empty_frac")),
        " sil_loud=", fmt(grand.get("silence_loud_frac")),
        " sp_quiet_empty=", fmt(grand.get("speech_quiet_empty_frac")),
    )
    return 0


def analyze_main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze all benchmark runs")
    parser.add_argument("--root", default=os.path.join("outputs", "benchmarks"), help="Root benchmarks directory")
    parser.add_argument("--out", default=None, help="Directory to write analysis outputs (defaults to <root>/analysis)")
    parser.add_argument("--no-csv", action="store_true", help="Do not write CSV files")
    args = parser.parse_args(argv)
    # Perform analysis and also print regression (delta) between latest two runs if present
    rc = analyze_all(args.root, outdir=args.out, write_csv=(not args.no_csv))
    try:
        run_dirs = sorted([d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d))])
        if len(run_dirs) >= 2:
            latest = run_dirs[-1]
            prev = run_dirs[-2]
            def load_summary(run_id):
                p = os.path.join(args.root, run_id, 'summary.json')
                with open(p, 'r', encoding='utf-8') as f:
                    return json.load(f).get('summary', {})
            s_latest = load_summary(latest)
            s_prev = load_summary(prev)
            keys = [
                'coverage_ok_frac', 'avg_acc_speech', 'avg_conf_speech',
                'avg_speech_fraction', 'avg_und_fraction',
                'silence_not_empty_frac', 'silence_loud_frac', 'speech_quiet_empty_frac',
            ]
            def fmt(v):
                return '-' if v is None else (f"{v:.3f}" if isinstance(v, (int,float)) else str(v))
            print("\nRegression (latest vs previous):")
            for k in keys:
                v1 = s_latest.get(k)
                v0 = s_prev.get(k)
                dv = None
                if isinstance(v1, (int,float)) and isinstance(v0, (int,float)):
                    dv = v1 - v0
                print(f" {k}: latest={fmt(v1)} prev={fmt(v0)} delta={fmt(dv)}")
    except Exception:
        pass
    return rc


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0].lower() == "analyze":
        return analyze_main(argv[1:])
    # Fallback to run mode for backward compatibility
    return run_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
