#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apex Legends Knockdown Highlight Extractor
==========================================
Detects YOUR knockdowns via audio fingerprinting of the knock sound,
cuts beat-synced clips, and compiles them into one video at 137 BPM.

Usage:
  1. Calibrate (one-time) -- tell it where a known knock is:
     python apex_highlights.py --calibrate --file "some_replay.mp4" --knock-at 45

  2. Scan & extract:
     python apex_highlights.py --input-dir "Z:\\Apex_replays\\Apex Legends" --output highlights.mp4

  3. Re-merge with different BPM (uses cached detections):
     python apex_highlights.py --remerge --bpm 140 --before-beats 8 --after-beats 8
"""

import argparse
import glob
import io
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import librosa
from scipy.signal import fftconvolve, find_peaks


# --- Constants ---------------------------------------------------------------

SAMPLE_RATE = 22050
TEMPLATE_FILENAME = "knock_template.npy"
TEMPLATE_META_FILENAME = "knock_template_meta.json"
DETECTIONS_FILENAME = "detections.json"


# --- FFmpeg helper -----------------------------------------------------------

def get_ffmpeg():
    """Get FFmpeg executable path from imageio_ffmpeg bundle."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def get_video_duration(video_path, ffmpeg_exe):
    """Get video duration in seconds via FFmpeg."""
    cmd = [ffmpeg_exe, '-hide_banner', '-i', str(video_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    for line in proc.stderr.split('\n'):
        if 'Duration' in line:
            parts = line.split('Duration:')[1].split(',')[0].strip()
            h, m, s = parts.split(':')
            return int(h) * 3600 + int(m) * 60 + float(s)
    return None


# --- Audio extraction (streamed via FFmpeg pipe) ----------------------------

def extract_audio(video_path, ffmpeg_exe, sr=SAMPLE_RATE, start=None, duration=None):
    """
    Extract audio from video as a float32 numpy array via FFmpeg pipe.
    Only the requested segment is decoded — nothing else touches RAM.
    """
    cmd = [ffmpeg_exe, '-hide_banner', '-loglevel', 'error']
    if start is not None:
        cmd += ['-ss', str(start)]
    cmd += ['-i', str(video_path)]
    if duration is not None:
        cmd += ['-t', str(duration)]
    cmd += [
        '-vn',                    # no video
        '-acodec', 'pcm_s16le',   # 16-bit PCM
        '-ar', str(sr),           # sample rate
        '-ac', '1',               # mono
        '-f', 'wav',              # WAV format
        'pipe:1'                  # pipe to stdout
    ]

    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction error: {proc.stderr.decode()[:500]}")

    # Skip 44-byte WAV header
    raw = proc.stdout[44:]
    if len(raw) == 0:
        return np.array([], dtype=np.float32), sr

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sr


# --- Calibration -------------------------------------------------------------

def calibrate_template(video_path, knock_timestamp, output_dir, ffmpeg_exe):
    """
    Extract the knock sound template from a known knock moment.
    Saves knock_template.npy + metadata + optional spectrogram.
    """
    _print(f"\n[CALIBRATE] Extracting knock sound template")
    _print(f"  Video:     {Path(video_path).name}")
    _print(f"  Timestamp: {knock_timestamp:.2f}s")

    # Extract 5s window centered on the knock
    window_start = max(0, knock_timestamp - 2.5)
    window_duration = 5.0

    audio, sr = extract_audio(video_path, ffmpeg_exe, start=window_start, duration=window_duration)
    _print(f"  Audio extracted: {len(audio)} samples ({len(audio)/sr:.2f}s)")

    if len(audio) == 0:
        _print("[ERROR] No audio extracted. Check the file and timestamp.")
        sys.exit(1)

    # Expected knock position in the window
    expected_time = knock_timestamp - window_start
    expected_sample = int(expected_time * sr)

    # Find the sharpest transient via onset detection
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=512)
    onsets_times = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=512,
        backtrack=False, units='time'
    )

    if len(onsets_times) == 0:
        # Fallback: loudest moment
        peak_sample = int(np.argmax(np.abs(audio)))
        _print(f"  [NOTE] No onsets detected, using loudest peak at sample {peak_sample}")
    else:
        # Convert onset times to samples
        onset_samples = (onsets_times * sr).astype(int)
        closest_idx = int(np.argmin(np.abs(onset_samples - expected_sample)))
        peak_sample = int(onset_samples[closest_idx])
        _print(f"  Found {len(onsets_times)} onsets, closest to knock at {onsets_times[closest_idx]:.3f}s in window")

    # Clamp peak_sample to valid range
    peak_sample = max(0, min(peak_sample, len(audio) - 1))

    # Extract template: 0.05s before peak to 0.35s after (0.4s total)
    tmpl_start = max(0, peak_sample - int(0.05 * sr))
    tmpl_end = min(len(audio), peak_sample + int(0.35 * sr))
    template = audio[tmpl_start:tmpl_end]

    if len(template) == 0:
        _print(f"  [ERROR] Empty template! peak_sample={peak_sample}, audio_len={len(audio)}")
        _print(f"  Falling back to 0.4s centered on expected knock time")
        center = expected_sample
        tmpl_start = max(0, center - int(0.2 * sr))
        tmpl_end = min(len(audio), center + int(0.2 * sr))
        template = audio[tmpl_start:tmpl_end]

    # Normalize template to unit max amplitude
    max_val = np.max(np.abs(template))
    if max_val < 1e-8:
        _print("  [ERROR] Template is silent! Check the timestamp.")
        sys.exit(1)
    template = template / max_val

    # Save template
    os.makedirs(output_dir, exist_ok=True)
    template_path = os.path.join(output_dir, TEMPLATE_FILENAME)
    np.save(template_path, template)

    # Save metadata
    meta = {
        "source_video": str(video_path),
        "knock_timestamp": knock_timestamp,
        "sample_rate": sr,
        "template_samples": len(template),
        "template_duration_s": round(len(template) / sr, 4),
        "peak_sample_in_window": int(peak_sample),
    }
    meta_path = os.path.join(output_dir, TEMPLATE_META_FILENAME)
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    _print(f"\n  Template saved:  {template_path}")
    _print(f"  Duration:        {len(template)/sr:.4f}s ({len(template)} samples)")
    _print(f"  Metadata:        {meta_path}")

    # Optional: save spectrogram visualization
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 6))

        # Waveform
        t = np.arange(len(template)) / sr
        axes[0].plot(t, template, linewidth=0.5, color='#00ff88')
        axes[0].set_facecolor('#1a1a2e')
        axes[0].set_title("Knock Sound Template -- Waveform", color='white')
        axes[0].set_xlabel("Time (s)", color='white')
        axes[0].set_ylabel("Amplitude", color='white')
        axes[0].tick_params(colors='white')

        # Mel spectrogram
        S = librosa.feature.melspectrogram(y=template, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sr, ax=axes[1], x_axis='time', y_axis='mel')
        axes[1].set_title("Knock Sound Template -- Mel Spectrogram", color='white')

        fig.patch.set_facecolor('#0f0f23')
        plt.tight_layout()
        spec_path = os.path.join(output_dir, "knock_template_spectrogram.png")
        plt.savefig(spec_path, dpi=150, facecolor='#0f0f23')
        plt.close()
        _print(f"  Spectrogram:     {spec_path}")
    except ImportError:
        _print("  [NOTE] matplotlib not available, skipping spectrogram")

    return template


# --- Audio scanning ----------------------------------------------------------

def scan_audio_for_knocks(video_path, template, ffmpeg_exe,
                          threshold=0.45, min_gap_s=5.0, chunk_duration_s=30.0):
    """
    Scan a video's audio for the knock sound using normalized cross-correlation.
    Audio is processed in overlapping chunks to limit RAM usage.

    Returns a list of (timestamp_seconds, confidence) tuples.
    """
    video_name = Path(video_path).name
    duration_s = get_video_duration(video_path, ffmpeg_exe)
    if duration_s is None:
        _print(f"  [WARN] Could not determine duration for {video_name}, assuming 300s")
        duration_s = 300.0

    template_len = len(template)
    template_energy = float(np.sqrt(np.sum(template ** 2)))
    # Overlap between chunks to avoid missing knocks at boundaries
    overlap_s = (template_len / SAMPLE_RATE) + 2.0

    all_detections = []
    chunk_start = 0.0

    _print(f"  Scanning {video_name} ({duration_s:.1f}s)...", end='', flush=True)

    while chunk_start < duration_s:
        chunk_dur = min(chunk_duration_s + overlap_s, duration_s - chunk_start)

        try:
            audio_chunk, _ = extract_audio(
                video_path, ffmpeg_exe,
                start=chunk_start, duration=chunk_dur
            )
        except RuntimeError as e:
            _print(f"\n  [WARN] Error at {chunk_start:.1f}s: {e}")
            chunk_start += chunk_duration_s
            continue

        if len(audio_chunk) < template_len:
            break

        # Normalize chunk
        chunk_max = float(np.max(np.abs(audio_chunk)))
        if chunk_max < 1e-6:
            chunk_start += chunk_duration_s
            continue
        audio_norm = audio_chunk / chunk_max

        # Cross-correlate with template
        correlation = fftconvolve(audio_norm, template[::-1], mode='valid')

        # Running energy for normalization
        kernel = np.ones(template_len, dtype=np.float32)
        audio_energy = np.sqrt(fftconvolve(audio_norm ** 2, kernel, mode='valid'))

        # Normalized cross-correlation (NCC)
        ncc = correlation / (audio_energy * template_energy + 1e-8)

        # Find peaks
        min_dist_samples = int(min_gap_s * SAMPLE_RATE)
        peaks, props = find_peaks(ncc, height=threshold, distance=min_dist_samples)

        for peak in peaks:
            timestamp = chunk_start + peak / SAMPLE_RATE
            confidence = float(ncc[peak])
            # Deduplicate across chunks
            if all(abs(timestamp - t) > min_gap_s for t, _ in all_detections):
                all_detections.append((timestamp, confidence))
                _print(f"\n    >> KNOCK at {_fmt_time(timestamp)} (confidence: {confidence:.3f})", end='', flush=True)

        chunk_start += chunk_duration_s
        _print('.', end='', flush=True)

    all_detections.sort(key=lambda x: x[0])
    _print(f"\n  -> {len(all_detections)} knock(s) found in {video_name}")
    return all_detections


# --- Visual scanning (KNOCKED DOWN text detection) --------------------------

# Default path for the visual template image
VISUAL_TEMPLATE_FILENAME = "knocked_down_template.png"

def scan_visual_for_knocks(video_path, ffmpeg_exe, visual_template=None,
                           sample_fps=2, match_thresh=0.55, min_gap_s=3.0):
    """
    Detect knockdowns by template-matching the 'KNOCKED DOWN' text overlay.

    Uses quarter-resolution (480x272) for fast decoding.  Applies a white-pixel
    pre-filter to skip most frames, then runs cv2.matchTemplate on candidates.

    Returns a list of (timestamp_seconds, confidence) tuples.
    """
    if visual_template is None:
        return []

    video_name = Path(video_path).name
    duration_s = get_video_duration(video_path, ffmpeg_exe)
    if duration_s is None:
        return []

    # Quarter-resolution: 4x fewer pixels to decode & process
    W, H = 480, 272

    # Scale template to match quarter-resolution (original is from 1920x1088)
    tmpl_h, tmpl_w = visual_template.shape[:2]
    scaled_tmpl = cv2.resize(visual_template, (tmpl_w // 4, tmpl_h // 4),
                             interpolation=cv2.INTER_AREA)
    # Ensure template is at least 1px in each dimension
    if scaled_tmpl.shape[0] < 3 or scaled_tmpl.shape[1] < 3:
        return []

    cmd = [
        ffmpeg_exe,
        '-i', str(video_path),
        '-vf', f'fps={sample_fps},scale={W}:{H}',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-v', 'error',
        '-'
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_bytes = W * H * 3
    frame_idx = 0
    detections = []
    skipped = 0

    # ROI for template matching: center-bottom where "KNOCKED DOWN" appears
    roi_y1, roi_y2 = int(H * 0.68), int(H * 0.75)
    roi_x1, roi_x2 = int(W * 0.20), int(W * 0.80)

    # Narrower band for fast white-pixel pre-filter
    pre_y1, pre_y2 = int(H * 0.69), int(H * 0.73)
    pre_x1, pre_x2 = int(W * 0.30), int(W * 0.70)

    while True:
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break

        frame = np.frombuffer(raw, dtype=np.uint8).reshape(H, W, 3)
        timestamp = frame_idx / sample_fps

        # --- FAST PRE-FILTER: count bright white pixels in narrow band ---
        pre_band = frame[pre_y1:pre_y2, pre_x1:pre_x2, :]
        white_px = int(np.sum(np.min(pre_band, axis=2) > 180))

        if white_px < 15:  # lower threshold for quarter-res
            skipped += 1
            frame_idx += 1
            continue

        # --- TEMPLATE MATCHING on promising frames only ---
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2, :]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        if (scaled_tmpl.shape[0] > gray_roi.shape[0] or
                scaled_tmpl.shape[1] > gray_roi.shape[1]):
            frame_idx += 1
            continue

        result = cv2.matchTemplate(gray_roi, scaled_tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        if max_val >= match_thresh:
            if all(abs(timestamp - t) > min_gap_s for t, _ in detections):
                detections.append((timestamp, float(max_val)))

        frame_idx += 1

    proc.stdout.close()
    proc.wait()

    return detections


def _fmt_time(seconds):
    """Format seconds as M:SS.ss"""
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{s:05.2f}"


# --- Clip cutting (re-encode for frame-precise beat sync) -------------------

def cut_clip_bpm(video_path, knock_timestamp, bpm, before_beats, after_beats,
                 output_path, ffmpeg_exe):
    """
    Cut a clip aligned to the BPM beat grid.
    Re-encodes for frame-precise cuts (stream copy would snap to keyframes).
    """
    beat_interval = 60.0 / bpm
    before_s = before_beats * beat_interval
    after_s = after_beats * beat_interval

    start_time = knock_timestamp - before_s
    end_time = knock_timestamp + after_s

    # Clamp to video bounds
    vid_duration = get_video_duration(video_path, ffmpeg_exe)
    if vid_duration:
        end_time = min(end_time, vid_duration)

    if start_time < 0:
        # Not enough lead-in — skip this clip (beat alignment would be off)
        _print(f"    [SKIP] Not enough footage before knock at {_fmt_time(knock_timestamp)}")
        return False

    clip_duration = end_time - start_time

    cmd = [
        ffmpeg_exe, '-hide_banner', '-loglevel', 'warning',
        '-y',
        '-ss', f'{start_time:.4f}',
        '-i', str(video_path),
        '-t', f'{clip_duration:.4f}',
        # Re-encode for frame-precise cuts
        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
        '-c:a', 'aac', '-b:a', '192k',
        '-avoid_negative_ts', 'make_zero',
        '-video_track_timescale', '90000',
        str(output_path)
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        _print(f"    [ERROR] FFmpeg cut failed: {proc.stderr[:300]}")
        return False

    return True


# --- Clip merging ------------------------------------------------------------

def merge_clips(clip_paths, output_path, ffmpeg_exe):
    """Merge clips using FFmpeg concat demuxer."""
    if not clip_paths:
        _print("[MERGE] No clips to merge!")
        return False

    concat_file = os.path.join(os.path.dirname(output_path), "_concat_list.txt")
    with open(concat_file, 'w', encoding='utf-8') as f:
        for clip in clip_paths:
            safe = str(clip).replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{safe}'\n")

    cmd = [
        ffmpeg_exe, '-hide_banner', '-loglevel', 'warning',
        '-y',
        '-f', 'concat', '-safe', '0',
        '-i', concat_file,
        '-c', 'copy',
        str(output_path)
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)

    try:
        os.remove(concat_file)
    except OSError:
        pass

    if proc.returncode != 0:
        _print(f"[MERGE] FFmpeg error: {proc.stderr[:500]}")
        return False

    return True


# --- Safe print (handles Windows cp1252) ------------------------------------

def _print(*args, **kwargs):
    """Print that handles Windows cp1252 encoding gracefully."""
    output = io.StringIO()
    print(*args, file=output, **kwargs)
    text = output.getvalue()
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except UnicodeEncodeError:
        sys.stdout.write(text.encode('ascii', 'replace').decode('ascii'))
        sys.stdout.flush()


# --- Per-video worker (top-level for multiprocessing) -----------------------

def _scan_one_video(video_path, ffmpeg_exe, vis_tmpl_path, audio_tmpl_path,
                    audio_threshold, audio_min_gap):
    """
    Scan a single video for knockdowns using visual and/or audio detection.
    Loads templates from file paths (avoids pickling large arrays).
    Returns a list of merged detection dicts.
    """
    video_hits = []  # (timestamp, confidence, source)

    # -- Visual scan --
    if vis_tmpl_path:
        visual_template = cv2.imread(vis_tmpl_path, cv2.IMREAD_GRAYSCALE)
        if visual_template is not None:
            vis_dets = scan_visual_for_knocks(video_path, ffmpeg_exe,
                                              visual_template=visual_template)
            for ts, conf in vis_dets:
                video_hits.append((ts, conf, 'visual'))

    # -- Audio scan --
    if audio_tmpl_path:
        template = np.load(audio_tmpl_path)
        aud_dets = scan_audio_for_knocks(
            video_path, template, ffmpeg_exe,
            threshold=audio_threshold,
            min_gap_s=audio_min_gap
        )
        for ts, conf in aud_dets:
            video_hits.append((ts, conf, 'audio'))

    # -- Merge & dedup within this video (2s window) --
    video_hits.sort(key=lambda x: x[0])
    merged = []
    for ts, conf, src in video_hits:
        duplicate = False
        for existing in merged:
            if abs(ts - existing['timestamp']) < 2.0:
                if src == 'visual' and existing['source'] != 'visual':
                    existing['timestamp'] = round(ts, 4)
                    existing['confidence'] = round(conf, 4)
                    existing['source'] = src
                duplicate = True
                break
        if not duplicate:
            merged.append({
                "video": str(video_path),
                "video_name": Path(video_path).name,
                "timestamp": round(ts, 4),
                "confidence": round(conf, 4),
                "source": src,
            })

    return merged


# --- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Apex Legends Knockdown Highlight Extractor -- audio-driven, BPM-synced",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Calibrate:   python apex_highlights.py --calibrate --file replay.mp4 --knock-at 45
  Extract:     python apex_highlights.py --input-dir "Z:\\Apex_replays\\Apex Legends"
  Re-merge:    python apex_highlights.py --remerge --bpm 140 --before-beats 8 --after-beats 8
        """
    )

    # Paths
    parser.add_argument('--input-dir', type=str,
                        default=r'Z:\Apex_replays\Apex Legends',
                        help='Directory containing .mp4 files')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: <input-dir>/cut/highlights.mp4)')
    parser.add_argument('--file', type=str, default=None,
                        help='Process a single file only')

    # BPM
    parser.add_argument('--bpm', type=float, default=137.0,
                        help='Target BPM (default: 137)')
    parser.add_argument('--before-beats', type=int, default=10,
                        help='Beats of footage before the knock (default: 10)')
    parser.add_argument('--after-beats', type=int, default=6,
                        help='Beats of footage after the knock (default: 6)')

    # Detection
    parser.add_argument('--threshold', type=float, default=0.45,
                        help='NCC confidence threshold 0-1 (default: 0.45)')
    parser.add_argument('--min-gap', type=float, default=5.0,
                        help='Min seconds between detections (default: 5)')

    # Modes
    parser.add_argument('--calibrate', action='store_true',
                        help='Calibration mode -- extract knock sound template')
    parser.add_argument('--knock-at', type=float, default=None,
                        help='Approx timestamp of a known knock (for --calibrate)')
    parser.add_argument('--remerge', action='store_true',
                        help='Re-cut and merge using cached detections (skip scanning)')
    parser.add_argument('--no-visual', action='store_true',
                        help='Disable visual KNOCKED DOWN detection')
    parser.add_argument('--no-audio', action='store_true',
                        help='Disable audio template matching')
    parser.add_argument('--workers', type=int, default=6,
                        help='Parallel workers for scanning (default: 6)')

    # Template
    parser.add_argument('--template', type=str, default=None,
                        help='Path to knock_template.npy')

    args = parser.parse_args()

    ffmpeg_exe = get_ffmpeg()
    input_dir = args.input_dir
    cut_dir = os.path.join(input_dir, "cut")
    os.makedirs(cut_dir, exist_ok=True)

    _print(f"===========================================================")
    _print(f"  Apex Legends Highlight Extractor")
    _print(f"  FFmpeg: {ffmpeg_exe}")
    _print(f"===========================================================")

    # ── CALIBRATE ──
    if args.calibrate:
        if not args.file:
            _print("\n[ERROR] --calibrate requires --file <video.mp4>")
            sys.exit(1)
        if args.knock_at is None:
            _print("\n[ERROR] --calibrate requires --knock-at <seconds>")
            sys.exit(1)

        video_path = args.file
        if not os.path.isabs(video_path):
            video_path = os.path.join(input_dir, video_path)

        if not os.path.exists(video_path):
            _print(f"\n[ERROR] File not found: {video_path}")
            sys.exit(1)

        calibrate_template(video_path, args.knock_at, cut_dir, ffmpeg_exe)
        _print("\n[DONE] Template calibrated! Run without --calibrate to extract highlights.")
        return

    # ── LOAD TEMPLATES ──
    # Audio template
    template_path = args.template or os.path.join(cut_dir, TEMPLATE_FILENAME)
    template = None
    if not args.no_audio:
        if not os.path.exists(template_path):
            _print(f"\n[WARN] Knock template not found: {template_path}")
            _print("  Audio scanning disabled. Using visual detection only.")
            _print('  To enable audio: python apex_highlights.py --calibrate --file "replay.mp4" --knock-at 45')
            args.no_audio = True
        else:
            template = np.load(template_path)
            _print(f"\nAudio template: {len(template)} samples ({len(template)/SAMPLE_RATE:.4f}s)")

    # Visual template (KNOCKED DOWN text image)
    visual_template = None
    if not args.no_visual:
        vis_tmpl_path = os.path.join(cut_dir, VISUAL_TEMPLATE_FILENAME)
        if not os.path.exists(vis_tmpl_path):
            _print(f"\n[WARN] Visual template not found: {vis_tmpl_path}")
            _print("  Visual scanning disabled. Using audio detection only.")
            args.no_visual = True
        else:
            visual_template = cv2.imread(vis_tmpl_path, cv2.IMREAD_GRAYSCALE)
            _print(f"Visual template: {visual_template.shape[1]}x{visual_template.shape[0]}px")

    # BPM math
    beat_interval = 60.0 / args.bpm
    total_beats = args.before_beats + args.after_beats
    clip_duration = total_beats * beat_interval
    _print(f"BPM:      {args.bpm}")
    _print(f"Beat:     {beat_interval:.4f}s")
    _print(f"Clip:     {total_beats} beats = {clip_duration:.4f}s")
    _print(f"Knock on: beat {args.before_beats + 1} of {total_beats}")

    # ── GATHER VIDEOS ──
    if args.file:
        video_path = args.file
        if not os.path.isabs(video_path):
            video_path = os.path.join(input_dir, video_path)
        videos = [video_path]
    else:
        videos = sorted(glob.glob(os.path.join(input_dir, "*.mp4")))

    if not videos:
        _print("\n[ERROR] No .mp4 files found")
        sys.exit(1)

    # ── DETECT OR LOAD CACHED ──
    detections_path = os.path.join(cut_dir, DETECTIONS_FILENAME)

    if args.remerge:
        # Load cached detections
        if not os.path.exists(detections_path):
            _print(f"\n[ERROR] No cached detections at {detections_path}")
            _print("  Run a full scan first (without --remerge).")
            sys.exit(1)
        with open(detections_path, 'r') as f:
            all_detections = json.load(f)
        _print(f"\nLoaded {len(all_detections)} cached detections from {detections_path}")
    else:
        # Scan all videos
        run_visual = not args.no_visual
        run_audio = not args.no_audio and os.path.exists(template_path)

        scan_modes = []
        if run_visual:
            scan_modes.append('visual')
        if run_audio:
            scan_modes.append('audio')

        _print(f"\n{'-'*60}")
        _print(f"PHASE 1: Scanning {len(videos)} video(s)  [{' + '.join(scan_modes)}]")
        _print(f"{'-'*60}\n")

        all_detections = []
        t_start = time.time()

        # -- Load visual template as bytes for pickling to workers --
        vis_tmpl_path = os.path.join(cut_dir, VISUAL_TEMPLATE_FILENAME) if run_visual else None
        audio_tmpl_path = template_path if run_audio else None

        num_workers = min(args.workers, len(videos))
        _print(f"Using {num_workers} parallel workers\n")

        # Submit all videos to the pool
        futures = {}
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            for video in videos:
                fut = pool.submit(
                    _scan_one_video,
                    video, ffmpeg_exe,
                    vis_tmpl_path, audio_tmpl_path,
                    args.threshold, args.min_gap
                )
                futures[fut] = video

            done_count = 0
            for fut in as_completed(futures):
                done_count += 1
                video = futures[fut]
                video_name = Path(video).name
                try:
                    merged = fut.result()
                    all_detections.extend(merged)
                    knocks_str = ', '.join(
                        f"{_fmt_time(d['timestamp'])}({d['source'][0]}:{d['confidence']:.2f})"
                        for d in merged
                    ) if merged else 'none'
                    _print(f"  [{done_count}/{len(videos)}] {video_name}: "
                           f"{len(merged)} knock(s) [{knocks_str}]")
                except Exception as e:
                    _print(f"  [{done_count}/{len(videos)}] {video_name}: ERROR {e}")

        elapsed = time.time() - t_start
        vis_count = sum(1 for d in all_detections if d.get('source') == 'visual')
        aud_count = sum(1 for d in all_detections if d.get('source') == 'audio')
        _print(f"\nScanning complete: {len(all_detections)} knocks in {elapsed:.1f}s")
        _print(f"  Visual: {vis_count}  |  Audio: {aud_count}")

        # Cache detections for re-merge
        with open(detections_path, 'w') as f:
            json.dump(all_detections, f, indent=2)
        _print(f"Detections saved: {detections_path}")

    if not all_detections:
        _print(f"\n[DONE] No knocks found. Try lowering --threshold (currently {args.threshold}).")
        return

    # ── CUT CLIPS ──
    _print(f"\n{'-'*60}")
    _print(f"PHASE 2: Cutting {len(all_detections)} beat-synced clips")
    _print(f"{'-'*60}\n")

    temp_dir = os.path.join(cut_dir, "_temp_clips")
    os.makedirs(temp_dir, exist_ok=True)

    clip_paths = []
    for idx, det in enumerate(all_detections):
        video = det["video"]
        ts = det["timestamp"]
        clip_name = f"clip_{idx:04d}.mp4"
        clip_path = os.path.join(temp_dir, clip_name)

        _print(f"  [{idx+1}/{len(all_detections)}] {det['video_name']} @ {_fmt_time(ts)} -> {clip_name}")

        success = cut_clip_bpm(
            video, ts, args.bpm,
            args.before_beats, args.after_beats,
            clip_path, ffmpeg_exe
        )

        if success and os.path.exists(clip_path) and os.path.getsize(clip_path) > 1024:
            clip_paths.append(clip_path)
        else:
            _print(f"    [SKIP] Failed or empty clip")

    # ── MERGE ──
    output_path = args.output or os.path.join(cut_dir, "highlights.mp4")
    _print(f"\n{'-'*60}")
    _print(f"PHASE 3: Merging {len(clip_paths)} clips")
    _print(f"{'-'*60}\n")

    success = merge_clips(clip_paths, output_path, ffmpeg_exe)

    # Clean up temp clips
    if success:
        _print("Cleaning up temp clips...")
        for clip in clip_paths:
            try:
                os.remove(clip)
            except OSError:
                pass
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass

    # ── SUMMARY ──
    if success and os.path.exists(output_path):
        output_size = os.path.getsize(output_path)
        out_duration = get_video_duration(output_path, ffmpeg_exe)
        _print(f"\n{'='*60}")
        _print(f"  DONE!")
        _print(f"  Output:   {output_path}")
        _print(f"  Size:     {output_size / (1024*1024):.1f} MB")
        _print(f"  Duration: {_fmt_time(out_duration or 0)}")
        _print(f"  Clips:    {len(clip_paths)}")
        _print(f"  BPM:      {args.bpm}")
        _print(f"  Beat:     {beat_interval:.4f}s")
        _print(f"  Clip len: {total_beats} beats ({clip_duration:.4f}s)")
        _print(f"  Knock on: beat {args.before_beats + 1}")
        _print(f"{'='*60}")
        _print(f"\n  Drop a {args.bpm} BPM track on top and every down hits the beat.")
    elif not success:
        _print("\n[ERROR] Merge failed. Check the errors above.")


if __name__ == "__main__":
    main()
