#!/usr/bin/env python3
"""Capture a Limelight HTTP stream using FFmpeg (no Python packages needed)."""
import argparse
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("photo", "images", "video", "preview"))
    parser.add_argument("--url", default="http://192.168.43.1:5800/")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "captures")
    parser.add_argument("--seconds", type=float, help="Stop after this many seconds; otherwise Ctrl+C")
    parser.add_argument("--interval", type=float, default=1, help="Seconds between training images (default: 1)")
    args = parser.parse_args()
    if args.interval <= 0 or (args.seconds is not None and args.seconds <= 0):
        parser.error("--interval and --seconds must be positive")
    executable = "ffplay" if args.mode == "preview" else "ffmpeg"
    if not shutil.which(executable):
        parser.error(f"{executable} is required; install FFmpeg first")
    network = ["-rw_timeout", "10000000"]
    if args.mode == "preview":
        command = [executable, "-hide_banner", *network, "-i", args.url]
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        destination = args.output.expanduser().resolve() / stamp
        destination.mkdir(parents=True, exist_ok=False)
        command = [executable, "-hide_banner", "-n", *network, "-i", args.url,
                   "-map", "0:v:0", "-an"]
        if args.seconds is not None:
            command += ["-t", str(args.seconds)]
        if args.mode == "photo":
            command += ["-frames:v", "1", "-update", "1", "-q:v", "2", str(destination / "photo.jpg")]
        elif args.mode == "images":
            command += ["-vf", f"fps=1/{args.interval}", "-q:v", "2", str(destination / "image_%06d.jpg")]
        else:
            # Matroska preserves the camera's encoded frames and tolerates interruptions.
            command += ["-c:v", "copy", str(destination / "video.mkv")]
        print(f"Saving to {destination}", flush=True)
    print("Press Ctrl+C to stop (Q closes preview).", flush=True)
    process = subprocess.Popen(command)
    try:
        result = process.wait()
    except KeyboardInterrupt:
        # The terminal also delivers SIGINT to FFmpeg, which finalizes the video.
        try:
            result = process.wait(timeout=15)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            process.terminate()
            process.wait()
            return 1
    if result not in (0, 255):
        print("Capture failed. Check robot Wi-Fi, port forwarding, and the stream URL above.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
