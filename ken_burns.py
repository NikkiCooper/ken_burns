#!/usr/bin/env python3
import cv2
import numpy as np
import os
import sys
import glob
import random
import subprocess
from cliOpts import cliOpts
from Bcolors import Bcolors

TARGET_W = 1920
TARGET_H = 1080


def collect_images(img_path):
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG')
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(img_path, ext)))
    files.sort()
    return files


def calculate_report(opts, img_path):
    bc = Bcolors()
    image_files = collect_images(img_path)
    N = len(image_files)
    duration = opts.duration
    ftb = opts.ftb
    fps = opts.fps
    total_seconds = N * duration
    visible_per_slide = duration - 2 * ftb
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)

    sep = f"{bc.BOLD+bc.Dark_Gray_f}{'─' * 58}{bc.ENDC}"
    print()
    print(f"{bc.BOLD+bc.White_f}  Ken Burns  {bc.Magenta_f}─{bc.OKBLUE}  Duration Calculator{bc.ENDC}")
    print(sep)
    print(f"  {bc.BOLD+bc.Magenta_f}Source Path      {bc.White_f}: {bc.OKBLUE}{img_path}{bc.ENDC}")
    print(f"  {bc.BOLD+bc.Magenta_f}Slides Found     {bc.White_f}: {bc.OKGREEN}{N}{bc.ENDC}")
    print(sep)
    print(f"  {bc.BOLD+bc.Magenta_f}Duration/Slide   {bc.White_f}: {bc.OKBLUE}{duration}s{bc.ENDC}")
    print(f"  {bc.BOLD+bc.Magenta_f}FTB              {bc.White_f}: {bc.OKBLUE}{ftb}s{bc.ENDC}")
    print(f"  {bc.BOLD+bc.Magenta_f}FPS              {bc.White_f}: {bc.OKBLUE}{fps}{bc.ENDC}")
    print(sep)
    if visible_per_slide <= 0:
        print(f"  {bc.BOLD+bc.Red_f}WARNING: --ftb ({ftb}s × 2 = {2*ftb}s) meets or exceeds --duration ({duration}s){bc.ENDC}")
    else:
        print(f"  {bc.BOLD+bc.Magenta_f}Visible Motion   {bc.White_f}: {bc.OKBLUE}{visible_per_slide:.1f}s{bc.White_f}"
              f" per slide  {bc.Dark_Gray_f}(duration − 2 × ftb){bc.ENDC}")
    print(f"  {bc.BOLD+bc.White_f}Total Duration   {bc.White_f}: {bc.BOLD+bc.OKGREEN}{minutes:02d}:{seconds:02d}{bc.White_f}"
          f"  {bc.Dark_Gray_f}({total_seconds:.0f}s  |  {N} slides){bc.ENDC}")
    print()


def render_ken_burns(image_files, output_path, duration_sec, fps, ftb_sec,
                     base_zoom_in, use_hevc, metadata_comment=None,
                     target_w=TARGET_W, target_h=TARGET_H):
    bc = Bcolors()
    N = len(image_files)
    total_frames = int(duration_sec * fps)
    fade_frames  = int(ftb_sec * fps)

    codec = "hevc_nvenc" if use_hevc else "h264_nvenc"
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{target_w}x{target_h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "pipe:0",
        "-an",
        "-c:v", codec, "-preset", "p4", "-cq", "22",
        *((["-metadata", f"comment={metadata_comment}"]) if metadata_comment else []),
        output_path
    ]

    proc = subprocess.Popen(
        ffmpeg_cmd, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    # Pre-allocate black frame on GPU for FTB blending
    gpu_black = cv2.cuda_GpuMat()
    gpu_black.upload(np.zeros((target_h, target_w, 3), dtype=np.uint8))

    current_direction = base_zoom_in
    streak_count = 1

    try:
        for idx, img_path in enumerate(image_files):
            # Zoom direction: alternate with randomised streaks (max 3 same direction)
            if idx > 0:
                if streak_count >= 3:
                    current_direction = not current_direction
                    streak_count = 1
                else:
                    if random.random() < 0.70:
                        current_direction = not current_direction
                        streak_count = 1
                    else:
                        streak_count += 1

            dir_col = bc.Cyan_f if current_direction else bc.Magenta_f
            dir_str = "zoom-in" if current_direction else "zoom-out"
            print(f"  {bc.Yellow_f}[{idx+1:3d}/{N}]{bc.Magenta_f}  "
                  f"{bc.OKBLUE}{os.path.basename(img_path)}"
                  f"  {bc.Dark_Gray_f}→ {dir_col}{dir_str}{bc.ENDC}")

            img = cv2.imread(img_path)
            if img is None:
                print(f"         {bc.Red_f}unreadable — skipped{bc.ENDC}")
                continue

            h, w = img.shape[:2]
            target_aspect = target_w / target_h

            # Wide crop: largest rectangle at target aspect ratio centred in image
            wide_w = w
            wide_h = int(w / target_aspect)
            if wide_h > h:
                wide_h = h
                wide_w = int(h * target_aspect)
            wide_x = (w - wide_w) // 2
            wide_y = (h - wide_h) // 2

            # Tight crop: 85% of wide, shifted slightly toward top (0.3 bias)
            tight_w = int(wide_w * 0.85)
            tight_h = int(wide_h * 0.85)
            tight_x = (w - tight_w) // 2
            tight_y = int((h - tight_h) * 0.3)

            if current_direction:  # zoom in: wide → tight
                start_w, start_h, start_x, start_y = wide_w,  wide_h,  wide_x,  wide_y
                end_w,   end_h,   end_x,   end_y   = tight_w, tight_h, tight_x, tight_y
            else:                  # zoom out: tight → wide
                start_w, start_h, start_x, start_y = tight_w, tight_h, tight_x, tight_y
                end_w,   end_h,   end_x,   end_y   = wide_w,  wide_h,  wide_x,  wide_y

            # Upload full image to GPU once per slide — all crops stay on device
            gpu_full = cv2.cuda_GpuMat()
            gpu_full.upload(img)

            for i in range(total_frames):
                t   = i / max(total_frames - 1, 1)
                t_s = (1.0 - np.cos(t * np.pi)) / 2.0  # cosine ease

                curr_x = int(start_x + (end_x - start_x) * t_s)
                curr_y = int(start_y + (end_y - start_y) * t_s)
                curr_w = max(1, int(start_w + (end_w - start_w) * t_s))
                curr_h = max(1, int(start_h + (end_h - start_h) * t_s))

                # GPU ROI (zero-copy) + resize on device — no PCIe transfer per frame
                gpu_roi   = gpu_full.rowRange(curr_y, curr_y + curr_h) \
                                    .colRange(curr_x, curr_x + curr_w)
                gpu_frame = cv2.cuda.resize(gpu_roi, (target_w, target_h),
                                            interpolation=cv2.INTER_CUBIC)

                # FTB fade — blend on GPU
                alpha = 1.0
                if fade_frames > 0:
                    if i < fade_frames:
                        alpha = i / fade_frames
                    elif i >= total_frames - fade_frames:
                        alpha = (total_frames - 1 - i) / fade_frames

                if alpha < 1.0:
                    gpu_frame = cv2.cuda.addWeighted(gpu_frame, alpha,
                                                     gpu_black, 1.0 - alpha, 0.0)

                proc.stdin.write(gpu_frame.download().tobytes())

    except BrokenPipeError:
        print(f"\n{bc.BOLD+bc.Red_f}Pipeline error: ffmpeg closed unexpectedly.{bc.ENDC}")
        return False
    finally:
        proc.stdin.close()
        proc.wait()

    return proc.returncode == 0


if __name__ == "__main__":
    bc = Bcolors()
    opts, src_path, dest_vid = cliOpts()

    if opts.calculate:
        calculate_report(opts, src_path)
        sys.exit(0)

    image_files = collect_images(src_path)
    N = len(image_files)
    if N == 0:
        print(f"{bc.BOLD+bc.Red_f}No images found in: {bc.OKBLUE}{src_path}{bc.ENDC}")
        sys.exit(1)

    duration_sec = opts.duration
    fps          = opts.fps
    ftb_sec      = opts.ftb
    use_hevc     = not opts.h264
    base_zoom_in = opts.zoom_in

    sep = f"{bc.BOLD+bc.Dark_Gray_f}{'─' * 58}{bc.ENDC}"
    print()
    print(f"{bc.BOLD+bc.White_f}  Ken Burns  {bc.Magenta_f}─{bc.OKBLUE}  Direct GPU Pipeline{bc.ENDC}")
    print(sep)
    print(f"  {bc.BOLD+bc.Magenta_f}Source       {bc.White_f}: {bc.OKBLUE}{src_path}{bc.ENDC}")
    print(f"  {bc.BOLD+bc.Magenta_f}Output       {bc.White_f}: {bc.OKBLUE}{dest_vid}{bc.ENDC}")
    print(f"  {bc.BOLD+bc.Magenta_f}Slides       {bc.White_f}: {bc.OKGREEN}{N}{bc.ENDC}")
    print(f"  {bc.BOLD+bc.Magenta_f}Duration     {bc.White_f}: {bc.OKBLUE}{duration_sec}s{bc.White_f} per slide{bc.ENDC}")
    print(f"  {bc.BOLD+bc.Magenta_f}FTB          {bc.White_f}: {bc.OKBLUE}{ftb_sec}s{bc.White_f} per side{bc.ENDC}")
    print(f"  {bc.BOLD+bc.Magenta_f}FPS          {bc.White_f}: {bc.OKBLUE}{fps}{bc.ENDC}")
    print(f"  {bc.BOLD+bc.Magenta_f}Zoom start   {bc.White_f}: {bc.OKBLUE}{'in' if base_zoom_in else 'out'}"
          f"{bc.Dark_Gray_f}  (alternates per slide){bc.ENDC}")
    print(f"  {bc.BOLD+bc.Magenta_f}Codec        {bc.White_f}: {bc.OKBLUE}"
          f"{'HEVC (H.265 via NVENC)' if use_hevc else 'H.264 (via NVENC)'}{bc.ENDC}")
    print(sep)
    print()

    metadata_comment = (
        f"ken_burns.py"
        f" | Slides: {N}"
        f" | Duration: {duration_sec}s"
        f" | FTB: {ftb_sec}s"
        f" | FPS: {fps}"
        f" | ZoomStart: {'in' if base_zoom_in else 'out'}"
        f" | Codec: {'HEVC' if use_hevc else 'H.264'}"
        f" | Source: {src_path}"
    )

    ok = render_ken_burns(image_files, dest_vid, duration_sec, fps, ftb_sec,
                          base_zoom_in, use_hevc, metadata_comment)

    if ok:
        print(f"\n{bc.BOLD+bc.Green_f}✨ Done.{bc.White_f}  Video saved to: {bc.OKBLUE}{dest_vid}{bc.ENDC}\n")
    else:
        print(f"\n{bc.BOLD+bc.Red_f}Render failed.{bc.ENDC}\n")
        sys.exit(1)
