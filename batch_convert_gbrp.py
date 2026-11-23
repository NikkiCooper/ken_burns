#!/usr/bin/env python
#  batch_convert_gbrp.py Copyright (c) 2026 Nikki Cooper
#
#  This program and the accompanying materials are made available under the
#  terms of the GNU Lesser General Public License, version 3.0 which is available at
#  https://www.gnu.org/licenses/gpl-3.0.html#license-text
#

"""
Batch transcode GBR planar video files with NVENC.

This module provides command-line batch processing for video files stored on a
source volume and written to a target directory. It combines glob-based file
discovery, optional numeric brace-range expansion, FFprobe stream inspection,
parallel FFmpeg execution, output verification, and aggregate status reporting.

The module is intended for workflows that need to identify source videos using the ``gbrp`` pixel format and convert
them to HEVC using NVIDIA NVENC settings suitable for high-quality yuv444p16le (10-bit 4:4:4, HEVC RExt -> yuv444p10le)
stream output.  This requires NVIDIA's NVENC hardware acceleration on enterprise workstation level hardware
(RTX A4500 or higher). The ``bgrp`` pixel format is incompatible with vaapi codecs on Linux. ken_burns prior to v0.5.0
defaulted to this mode and the result was files that were not only huge in size, but ones whose colors were washed out
when using vaapi. This script converts a video originally transcoded this way to HEVC (Rext), yuv444p10le (also known as
 10-bit 4:4:4, HEVC RExt). If you have never used ken_burns before, then you do not need this utility.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
from pathlib import Path
import re
import subprocess
import sys


def parse_numeric_ranges(pattern):
  """Expands custom brace ranges like {001-020} alongside standard globs."""
  match = re.search(r"\{(\d+)-(\d+)\}", pattern)
  if not match:
    return sorted(glob.glob(pattern, recursive=True))

  start_str, end_str = match.groups()
  width = len(start_str)
  start, end = int(start_str), int(end_str)

  expanded_files = []
  for i in range(start, end + 1):
    num_formatted = str(i).zfill(width)
    sub_pattern = pattern.replace(f"{{{start_str}-{end_str}}}", num_formatted)
    expanded_files.extend(glob.glob(sub_pattern, recursive=True))

  return sorted(list(set(expanded_files)))


def get_stream_info(filepath):
  cmd = [
      "ffprobe",
      "-v",
      "error",
      "-select_streams",
      "v:0",
      "-show_entries",
      "stream=pix_fmt,nb_frames",
      "-of",
      "csv=p=0",
      str(filepath),
  ]
  try:
    res = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
    )
    parts = res.stdout.strip().split(",")
    pix_fmt = parts[0] if len(parts) > 0 else None
    nb_frames = parts[1] if len(parts) > 1 else None
    return pix_fmt, nb_frames
  except subprocess.CalledProcessError:
    return None, None


def process_single_file(video_file, out_path, cq):
  if not video_file.exists():
    return ("missing", video_file.name, "File not found on source RAID")

  pix_fmt, src_frames = get_stream_info(video_file)
  if pix_fmt != "gbrp":
    return ("skipped", video_file.name, f"pix_fmt={pix_fmt}")

  dest_file = out_path / video_file.name
  if dest_file.exists():
    return ("skipped", video_file.name, "Already exists on target RAID")

  cmd = [
      "ffmpeg",
      "-hide_banner",
      "-loglevel",
      "error",
      "-y",
      "-i",
      str(video_file),
      "-c:v",
      "hevc_nvenc",
      "-preset",
      "p7",
      "-tune",
      "hq",
      "-profile:v",
      "rext",
      "-pix_fmt",
      "yuv444p16le",
      "-colorspace",
      "bt709",
      "-color_primaries",
      "bt709",
      "-color_trc",
      "bt709",
      "-color_range",
      "tv",
      "-rc",
      "vbr",
      "-cq",
      str(cq),
      "-b:v",
      "0",
      "-g",
      "250",
      "-bf",
      "4",
      "-b_ref_mode",
      "middle",
      "-rc-lookahead",
      "32",
      "-spatial-aq",
      "1",
      "-aq-strength",
      "4",
      "-an",
      "-movflags",
      "+faststart",
      str(dest_file),
  ]

  result = subprocess.run(cmd)
  if result.returncode != 0:
    if dest_file.exists():
      dest_file.unlink()
    return ("error", video_file.name, "FFmpeg exited non-zero")

  new_pix, dst_frames = get_stream_info(dest_file)
  if new_pix != "yuv444p10le" or (
      src_frames and dst_frames and src_frames != dst_frames
  ):
    dest_file.unlink()
    return (
        "error",
        video_file.name,
        (
            f"Verification mismatch (pix={new_pix},"
            f" frames={dst_frames}/{src_frames})"
        ),
    )

  src_mb = video_file.stat().st_size / (1024 * 1024)
  dst_mb = dest_file.stat().st_size / (1024 * 1024)
  saved_pct = (1 - (dst_mb / src_mb)) * 100
  return (
      "success",
      video_file.name,
      f"{src_mb:.1f} MiB -> {dst_mb:.1f} MiB (-{saved_pct:.1f}%)",
  )


def main():
  parser = argparse.ArgumentParser(
      description="Dual-RAID NVENC Batch Transcoder for Legacy gbrp Slideshows"
  )
  parser.add_argument(
      "--inputGlob",
      required=True,
      help=(
          "Input glob or range (e.g."
          " '/mnt/raid_src/VIDEOS/Kacy-Slideshow-{001-020}_Kenburns.mp4')"
      ),
  )
  parser.add_argument(
      "--outputDir",
      required=True,
      help="Destination directory on separate NVMe RAID",
  )
  parser.add_argument(
      "--workers",
      type=int,
      default=2,
      help="Concurrent NVENC encodes (default: 2 for A4500 dual NVENC)",
  )
  parser.add_argument(
      "--cq", type=int, default=26, help="NVENC constant quality (default: 26)"
  )
  parser.add_argument(
      "--dryRun",
      action="store_true",
      help="Audit matched files without encoding",
  )
  args = parser.parse_args()

  matched_files = [Path(p) for p in parse_numeric_ranges(args.inputGlob)]
  if not matched_files:
    print(f"[Warning] No files matched: {args.inputGlob}")
    sys.exit(0)

  out_path = Path(args.outputDir)
  out_path.mkdir(parents=True, exist_ok=True)

  print(
      f"[Target Audit] Matched {len(matched_files)} file(s) | Workers:"
      f" {args.workers} | CQ: {args.cq}"
  )
  if args.dryRun:
    for f in matched_files:
      pix, _ = get_stream_info(f)
      print(f"  [{pix}] {f}")
    sys.exit(0)

  stats = {"success": 0, "skipped": 0, "error": 0, "missing": 0}
  with ThreadPoolExecutor(max_workers=args.workers) as pool:
    futures = {
        pool.submit(process_single_file, f, out_path, args.cq): f
        for f in matched_files
    }
    for future in as_completed(futures):
      status, name, detail = future.result()
      stats[status] += 1
      print(f"[{status.upper()}] {name} — {detail}")

  print(
      f"\n[Batch Complete] Converted: {stats['success']} | Skipped:"
      f" {stats['skipped']} | Errors: {stats['error']}"
  )


if __name__ == "__main__":
  main()
