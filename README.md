# Apex Legends Highlight Extractor

**Automatically detect "KNOCKED DOWN" events in your Apex Legends gameplay, cut beat-synced clips, and merge them into a montage.**

This tool uses a combination of **audio fingerprinting** (detecting the specific sound of a knockdown) and **visual scanning** (looking for the "KNOCKED DOWN" text on screen) to find your best moments with high accuracy.

## Features

-   **Dual Detection**: Uses both audio (sound of the knock) and visual (text on screen) to minimize false positives.
-   **Beat-Sync**: Automatically cuts clips to a specific BPM (default 137) so they align perfectly with music in your final edit.
-   **Smart Merging**: Concatenates all clips into a single `highlights.mp4` file.
-   **Re-merge Mode**: Change the BPM or clip length instantly without re-scanning (uses cached detections).
-   **Multiprocessing**: Utilizes multiple CPU cores to scan hours of footage in minutes.

## Prerequisites

-   **Python 3.8+**
-   **FFmpeg**: Must be installed and added to your system PATH.
    -   *Windows*: Download from [ffmpeg.org](https://ffmpeg.org/download.html), extract, and add the `bin` folder to your Environment Variables.
    -   *Mac/Linux*: `brew install ffmpeg` or `sudo apt install ffmpeg`.

## Installation

1.  **Clone or Download** this repository.
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. The Easy Way (Visual Only)
Technically, you can run the tool right away using just the visual detector (included `knocked_down_template.png`).

```bash
python apex_highlights.py --input-dir "C:\Path\To\Your\Apex_Clips"
```
*Note: Audio detection is disabled by default until you calibrate it for your specific audio setup/files, as game audio levels vary significantly between users.*

### 2. The Best Way (Visual + Audio)
For the best accuracy, you should **calibrate** the tool with one of your own clips where a clear knockdown sound can be heard.

1.  Find a clip with a clean knockdown sound (minimal voice chatter/explosions over it).
2.  Note the timestamp (e.g., 45 seconds in).
3.  Run calibration:
    ```bash
    python apex_highlights.py --calibrate --file "C:\Path\To\Your\Clip.mp4" --knock-at 45
    ```
    This will generate `knock_template.npy` in the `cut/` subfolder.
4.  Now run the full scan:
    ```bash
    python apex_highlights.py --input-dir "C:\Path\To\Your\Apex_Clips"
    ```

### 3. Re-Merging (Changing BPM)
Once a scan is complete, detections are saved to `cut/detections.json`. You can re-create the montage with different settings instantly:

```bash
# Change to 140 BPM, with 8 beats before and 8 beats after the knock
python apex_highlights.py --remerge --bpm 140 --before-beats 8 --after-beats 8
```

## Advanced Options

-   `--threshold`: Confidence threshold for audio matches (0.0 - 1.0). Default `0.45`. Lower this if it misses knocks, raise it if you get false positives.
-   `--workers`: Number of CPU threads to use. Default `6`.
-   `--no-visual`: Disable visual scanning (faster, but relies solely on audio).
-   --no-audio`: Disable audio scanning.

## Troubleshooting

-   **"FFmpeg not found"**: Make sure FFmpeg is installed and accessible from your command line.
-   **No highlights found**:
    -   Try running with `--no-audio` to see if visual detection works alone.
    -   If audio calibration was done, try lowering `--threshold 0.35`.
-   **False positives**: Increase `--threshold` to `0.55` or higher.

## License

MIT License. Feel free to modify and use for your own montages!
