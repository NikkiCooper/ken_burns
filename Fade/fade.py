#!/usr/bin/env python3
import cv2
import numpy as np
import os
import sys
import glob
import json
import random
import subprocess
from cliOpts import cliOpts
from Bcolors import Bcolors

# Source images are downscaled to 1080p for processing
TARGET_W = 1920
TARGET_H = 1080

# Bilateral filter presets for ESRGAN artefact smoothing
# intensity: blend factor 0.0–1.0 (0=no effect, 1=full filter, matches PyVid2 Intensity slider)
SMOOTH_PRESETS = {
    'subtle': dict(d=3,  sigmaColor=15,  sigmaSpace=15,  intensity=0.40),
    'light':  dict(d=5,  sigmaColor=35,  sigmaSpace=35,  intensity=0.50),
    'medium': dict(d=9,  sigmaColor=75,  sigmaSpace=75,  intensity=0.70),
    'strong': dict(d=13, sigmaColor=120, sigmaSpace=120, intensity=0.85),
}

# Laplacian boost presets for edge sharpness recovery
SHARPEN_PRESETS = {
    'subtle': 0.10,
    'light':  0.25,
    'medium': 0.45,
    'strong': 0.65,
}


def print_filter_help():
    bc = Bcolors()
    sep  = f"{bc.BOLD + bc.Dark_Gray_f}{'─' * 68}{bc.ENDC}"
    sep2 = f"{bc.Dark_Gray_f}{'╌' * 68}{bc.ENDC}"

    def hdr(title):
        print(f"\n{bc.BOLD + bc.White_f}  {title}{bc.ENDC}")
        print(sep)

    def sub(title):
        print(f"\n{bc.BOLD + bc.Cyan_f}  {title}{bc.ENDC}")
        print(sep2)

    def row(label, value):
        print(f"  {bc.BOLD + bc.Magenta_f}{label:<22}{bc.White_f}: {value}{bc.ENDC}")

    def para(text):
        import textwrap
        for line in textwrap.wrap(text, width=64):
            print(f"  {bc.Dark_Gray_f}{line}{bc.ENDC}")

    print()
    print(f"{bc.BOLD + bc.White_f}  Fade  {bc.Magenta_f}─{bc.OKBLUE}  Image Filter Reference{bc.ENDC}")
    print(sep)

    # ── OVERVIEW ──────────────────────────────────────────────────────────────
    hdr("Overview")
    para("Two optional CUDA filters operate on each image after it is resized "
         "to 1920×1080 on the GPU. Both are purely on-device — no CPU round-trips "
         "at any point in the pipeline.")
    print()
    para("--smooth  runs first  (bilateral filter  — suppresses artefacts)")
    para("--sharpen runs second (Laplacian boost   — recovers edge crispness)")
    print()
    para("Either filter can be used alone. When used together, always apply "
         "--smooth before --sharpen; reversing the order sharpens artefacts "
         "instead of suppressing them.")

    # ── --smooth ──────────────────────────────────────────────────────────────
    hdr("--smooth  {light | medium | strong}")
    para("A CUDA bilateral filter. Smooths flat regions and softens jagged "
         "staircase artefacts introduced by upscalers such as Real-ESRGAN, "
         "while preserving real edges (subject outlines, fine detail) by "
         "weighting neighbours by both spatial distance and colour distance.")
    print()

    sub("Presets")
    row("light ", f"{bc.OKBLUE}d=5,  sigmaColor=35,  sigmaSpace=35{bc.ENDC}")
    para("  Barely perceptible. Good for mild aliasing or as a safety net "
         "when you are not sure whether smoothing is needed.")
    print()
    row("medium", f"{bc.OKBLUE}d=9,  sigmaColor=75,  sigmaSpace=75{bc.ENDC}")
    para("  Recommended starting point for ESRGAN output. Removes most "
         "staircase artefacts on clothing and hair without visibly softening "
         "the subject.")
    print()
    row("strong", f"{bc.OKBLUE}d=13, sigmaColor=120, sigmaSpace=120{bc.ENDC}")
    para("  Aggressive. Use when medium leaves visible jaggies. Produces "
         "a painterly look on fine texture (hair, fabric weave). Almost "
         "always pair with --sharpen to recover definition.")

    sub("When to use --smooth")
    para("• Source images were processed by Real-ESRGAN or a similar upscaler "
         "and show staircase/aliasing artefacts on curved edges.")
    para("• Skin tones or clothing show high-frequency noise from the upscaler.")
    para("• Images were compressed (JPEG) at low quality and show blocking "
         "artefacts.")

    sub("When NOT to use --smooth")
    para("• Source images are clean (raw camera output, lossless PNG) — "
         "bilateral filtering adds no value and costs GPU time.")
    para("• Images contain fine repeating texture (brickwork, fabric close-ups) "
         "that you want to preserve — bilateral can wash out periodically "
         "structured detail.")
    para("• You are rendering a test/proof run and want to see native "
         "upscaler output unmodified.")

    sub("Performance note")
    para("Bilateral cost scales with d². light (~25 ops/pixel) is fast. "
         "strong (~169 ops/pixel) is ~6.8× heavier. On drift slides it runs "
         "every frame; on static slides it runs once at load time.")

    # ── --sharpen ─────────────────────────────────────────────────────────────
    hdr("--sharpen  {light | medium | strong}")
    para("A CUDA Laplacian edge boost. Converts the image to greyscale, "
         "applies a 3×3 Laplacian kernel to extract edge detail, then blends "
         "the absolute edge map back onto the original BGR image. The result "
         "is increased perceived sharpness without altering colour.")
    print()

    sub("Presets")
    row("light ", f"{bc.OKBLUE}boost_strength=0.25{bc.ENDC}")
    para("  Subtle crispening. Suitable after --smooth medium or as a "
         "standalone on slightly soft source images.")
    print()
    row("medium", f"{bc.OKBLUE}boost_strength=0.45{bc.ENDC}")
    para("  Clearly visible sharpening. Good recovery after --smooth strong. "
         "May over-sharpen noise on poor source images.")
    print()
    row("strong", f"{bc.OKBLUE}boost_strength=0.65{bc.ENDC}")
    para("  Heavy sharpening. Use only on clean, high-quality source images "
         "or when a stylised high-contrast look is intentional. Amplifies "
         "any residual noise or artefacts aggressively.")

    sub("When to use --sharpen")
    para("• After --smooth, to recover subject definition softened by "
         "the bilateral pass.")
    para("• Source images are slightly soft (minor camera shake, "
         "mild defocus) and a crispening pass is desirable.")
    para("• A high-contrast editorial look is intentional.")

    sub("When NOT to use --sharpen")
    para("• Without --smooth on ESRGAN output — Laplacian boost will "
         "amplify the very artefacts you are trying to suppress.")
    para("• Source images already have strong in-camera sharpening "
         "or were processed with a dedicated sharpening tool — "
         "double-sharpening produces haloing and ringing.")
    para("• Images contain significant digital noise — boost_strength "
         "above 0.25 will make noise visually dominant.")

    # ── COMBINATIONS ──────────────────────────────────────────────────────────
    hdr("Recommended Combinations")

    sub("ESRGAN portraits — typical starting point")
    print(f"  {bc.OKBLUE}--smooth medium --sharpen light{bc.ENDC}")
    para("  Suppresses staircase artefacts on skin and clothing, then "
         "recovers subject-edge definition. The most generally useful "
         "combination for human-subject slideshows.")
    print()

    sub("Heavy ESRGAN artefacts")
    print(f"  {bc.OKBLUE}--smooth strong --sharpen medium{bc.ENDC}")
    para("  Stronger suppression pass followed by stronger recovery. "
         "Expect a slightly painterly look on fine texture.")
    print()

    sub("Clean source, mild crispening only")
    print(f"  {bc.OKBLUE}--sharpen light{bc.ENDC}")
    para("  No smoothing needed. Light Laplacian adds a subtle lift "
         "to perceived sharpness without visible artefacts.")
    print()

    sub("Compressed/noisy source, smoothing only")
    print(f"  {bc.OKBLUE}--smooth light{bc.ENDC}")
    para("  Suppresses JPEG blocking and mild noise. Do not add --sharpen "
         "unless source quality is good enough to survive it.")
    print()

    sub("Combinations to avoid")
    print(f"  {bc.BOLD + bc.Yellow_f}--sharpen strong  (alone on ESRGAN output){bc.ENDC}")
    para("  Amplifies upscaler artefacts. Always smooth first.")
    print()
    print(f"  {bc.BOLD + bc.Yellow_f}--smooth strong   (alone on fine-texture images){bc.ENDC}")
    para("  Destroys fine periodic detail. Use medium or pair with sharpen.")
    print()

    # ── PIPELINE POSITION ─────────────────────────────────────────────────────
    hdr("Pipeline Position")
    para("Both filters are applied inside compute_slide_params() immediately "
         "after the GPU resize to 1920×1080. For non-drift slides this is a "
         "one-time cost per slide. For drift slides (--drift > 0) it is "
         "per-frame, since each frame has a different crop/resize. The "
         "Laplacian filter object is recreated each call; bilateral is a "
         "direct CUDA call — both are GPU-native with no host memory access.")
    print()


def print_fmap_help():
    bc = Bcolors()
    sep  = f"{bc.BOLD + bc.Dark_Gray_f}{'─' * 68}{bc.ENDC}"
    sep2 = f"{bc.Dark_Gray_f}{'╌' * 68}{bc.ENDC}"

    def hdr(title):
        print(f"\n{bc.BOLD + bc.White_f}  {title}{bc.ENDC}")
        print(sep)

    def sub(title):
        print(f"\n{bc.BOLD + bc.Cyan_f}  {title}{bc.ENDC}")
        print(sep2)

    def para(text):
        import textwrap
        for line in textwrap.wrap(text, width=64):
            print(f"  {bc.Dark_Gray_f}{line}{bc.ENDC}")

    def cmd(text):
        print(f"  {bc.OKBLUE}{text}{bc.ENDC}")

    def note(text):
        print(f"  {bc.Yellow_f}{text}{bc.ENDC}")

    print()
    print(f"{bc.BOLD + bc.White_f}  Fade  {bc.Magenta_f}─{bc.OKBLUE}  Filter Map Reference{bc.ENDC}")
    print(sep)

    # ── OVERVIEW ──────────────────────────────────────────────────────────────
    hdr("Overview")
    para("The filter map system lets you apply --smooth and --sharpen "
         "selectively on a per-image basis, instead of globally to every "
         "slide. This is useful when only some images in a set exhibit "
         "ESRGAN artefacts and you don't want to degrade the rest.")
    print()
    para("A filter map is a JSON file (filter_map.json) stored in the image "
         "directory. Each entry maps an image filename to either a filter "
         "spec or null (no filtering).")
    print()
    para("Use --use_filter_map at render time to load it. The map is always "
         "read from --img_path/filter_map.json — it is inseparable from the "
         "image set it was generated for.")

    # ── WORKFLOW ──────────────────────────────────────────────────────────────
    hdr("Workflow")

    sub("Step 1 — Generate the map")
    cmd("python fade.py --img_path /path/to/imgs --gen_filter_map")
    print()
    para("Analyses every image using an edge-direction discontinuity detector "
         "and writes filter_map.json to --img_path. Prints a score table "
         "sorted lowest→highest. Images above --fmap_threshold are "
         "pre-populated with a filter suggestion; the rest are set to null.")
    print()
    para("The detector scores edge zigzag: 0 = smooth consistent directions, "
         "100 = pure H/V staircase. Default threshold is 30.")

    sub("Step 2 — Review the score table")
    para("The printed table shows every image, its score, and the suggested "
         "action. Use this to calibrate the threshold for your dataset:")
    print()
    cmd("python fade.py --img_path /path/to/imgs --gen_filter_map --fmap_threshold 25")
    print()
    para("Lower threshold = more images flagged. Higher = fewer. Re-run as "
         "many times as needed — each run overwrites filter_map.json.")

    sub("Step 3 — Edit the JSON (optional)")
    para("Open filter_map.json in any text editor. Each entry is one of:")
    print()
    print(f"  {bc.Dark_Gray_f}Auto-generated (smooth only — sharpen not auto-suggested):{bc.ENDC}")
    print(f'  {bc.OKBLUE}"img_042.jpg": {{"smooth": "light"}}{bc.ENDC}')
    print()
    print(f"  {bc.Dark_Gray_f}Manually add sharpen to a specific entry if desired:{bc.ENDC}")
    print(f'  {bc.OKBLUE}"img_042.jpg": {{"smooth": "light", "sharpen": "light"}}{bc.ENDC}')
    print()
    print(f"  {bc.Dark_Gray_f}Suppressed (no filtering for this image):{bc.ENDC}")
    print(f'  {bc.OKBLUE}"img_017.jpg": null{bc.ENDC}')
    print()
    para("Sharpen is deliberately not auto-suggested — whether to sharpen after "
         "smoothing depends on how the result looks. Add it manually to specific "
         "entries, or use global --sharpen at render time as a fallback for all "
         "images not explicitly specifying it in the map.")
    para("To override a smooth suggestion: change the preset string "
         "(subtle/light/medium/strong) or set the entry to null.")
    para("Images not present in the map fall back to the global --smooth/"
         "--sharpen flags set on the command line.")

    sub("Step 4 — Render")
    para("Add --use_filter_map to load filter_map.json from --img_path:")
    print()
    cmd("python fade.py --img_path /path/to/imgs -o ~/out.mp4 --use_filter_map")
    print()
    para("If filter_map.json is not found in --img_path, an error is printed "
         "explaining that it needs to be generated first with --gen_filter_map.")
    print()
    para("The header shows the map filename and how many images are flagged.")

    # ── JSON FORMAT ────────────────────────────────────────────────────────────
    hdr("filter_map.json Format")
    para("Keys are bare filenames (no directory path). Values are either "
         "null or a dict with 'smooth' and/or 'sharpen' keys.")
    print()
    print(f"  {bc.Dark_Gray_f}Example:{bc.ENDC}")
    lines = [
        '{',
        '  "img_001.jpg": {"smooth": "light", "sharpen": "light"},',
        '  "img_002.jpg": null,',
        '  "img_003.jpg": {"smooth": "medium"},',
        '  "img_004.jpg": {"sharpen": "subtle"},',
        '  "img_005.jpg": null',
        '}',
    ]
    for line in lines:
        print(f"  {bc.OKBLUE}{line}{bc.ENDC}")
    print()
    para("Either 'smooth' or 'sharpen' may be omitted from a spec — the "
         "missing key means no filter of that type for that image, "
         "regardless of global flags.")

    # ── PRECEDENCE ────────────────────────────────────────────────────────────
    hdr("Precedence Rules")
    rules = [
        ("Image in map, spec entry",    "uses map smooth + sharpen  (global flags ignored)"),
        ("Image in map, null entry",     "no filtering  (global flags ignored)"),
        ("Image not in map",             "falls back to global --smooth / --sharpen"),
        ("--use_filter_map not given",   "global --smooth / --sharpen apply to all images"),
        ("--use_filter_map, no JSON",    "error: map not found — run --gen_filter_map first"),
    ]
    for label, value in rules:
        print(f"  {bc.BOLD + bc.Magenta_f}{label:<36}{bc.White_f}: "
              f"{bc.Dark_Gray_f}{value}{bc.ENDC}")

    # ── GENERATION OPTIONS ────────────────────────────────────────────────────
    hdr("Generation Options")

    sub("Control suggested filter preset")
    para("Pass --smooth and/or --sharpen with --gen_filter_map to control "
         "what gets written into flagged entries:")
    print()
    cmd("python fade.py --img_path /path --gen_filter_map --smooth subtle")
    para("  → flagged images get: {\"smooth\": \"subtle\", \"sharpen\": null}")
    print()
    cmd("python fade.py --img_path /path --gen_filter_map --smooth light --sharpen light")
    para("  → flagged images get: {\"smooth\": \"light\", \"sharpen\": \"light\"}")
    print()
    note("Tip: omit --sharpen from gen command if you want smooth-only "
         "entries. You can always add sharpen manually to specific entries "
         "in the JSON.")

    sub("Adjust detection threshold")
    cmd("python fade.py --img_path /path --gen_filter_map --fmap_threshold 20")
    para("  → more images flagged (lower bar)")
    print()
    cmd("python fade.py --img_path /path --gen_filter_map --fmap_threshold 45")
    para("  → fewer images flagged (higher bar)")
    print()
    para("Threshold is on a 0–100 scale. The score table printed during "
         "generation shows the full distribution — use it to find a natural "
         "gap in your dataset's scores. If no gap exists (scores form a "
         "continuous bell curve), automated detection cannot separate "
         "artefacted from clean images and manual editing is more reliable.")

    # ── EXAMPLES END-TO-END ───────────────────────────────────────────────────
    hdr("Complete Examples")

    sub("Typical ESRGAN portrait set — generate then render")
    cmd("# Generate and review score table")
    cmd("python fade.py --img_path ~/imgs/set_042 --gen_filter_map --smooth light")
    print()
    cmd("# Edit ~/imgs/set_042/filter_map.json if needed, then render")
    cmd("python fade.py --img_path ~/imgs/set_042 -o ~/out/set_042.mp4 \\")
    cmd("    --duration 9 --drift 0.33 --rand_drift --use_filter_map")
    print()

    sub("With global fallback for images not in the map")
    cmd("python fade.py --img_path ~/imgs/set_042 -o ~/out/set_042.mp4 \\")
    cmd("    --duration 9 --drift 0.33 --use_filter_map --smooth subtle")
    para("  Images in the map use their per-image settings. Images not in "
         "the map get --smooth subtle as fallback.")
    print()

    sub("Re-generate with tighter threshold after reviewing scores")
    cmd("python fade.py --img_path ~/imgs/set_042 --gen_filter_map \\")
    cmd("    --fmap_threshold 22 --smooth light")
    para("  Overwrites the previous filter_map.json with the new threshold.")
    print()

    # ── LIMITATIONS ───────────────────────────────────────────────────────────
    hdr("Detector Limitations")
    para("The edge-direction discontinuity detector works best on images "
         "where staircasing appears on smooth areas (sky, plain backgrounds, "
         "clean body contours). It is less effective on:")
    print()
    para("  • Fashion/portrait images with complex clothing — textile texture "
         "has naturally inconsistent edge directions that score similarly to "
         "ESRGAN staircasing")
    para("  • Images with fine repeating patterns (grids, fabric weave)")
    para("  • Sets where all images score in a narrow continuous band — "
         "no threshold will cleanly separate artefacted from clean")
    print()
    para("In these cases, run --gen_filter_map to get the template, then "
         "populate it manually based on visual inspection. The JSON structure "
         "and render integration work the same regardless of how the map "
         "was created.")
    print()


def apply_bilateral_smooth(gpu_img, smooth, intensity_override=None):
    """
    CUDA bilateral filter with intensity blend (mirrors PyVid2 Intensity slider).
    intensity_override: 0.0–1.0, overrides preset default when provided.
    """
    p = SMOOTH_PRESETS[smooth]
    intensity = intensity_override if intensity_override is not None else p['intensity']
    gpu_filtered = cv2.cuda.bilateralFilter(gpu_img, p['d'], p['sigmaColor'], p['sigmaSpace'])
    if intensity >= 1.0:
        return gpu_filtered
    return cv2.cuda.addWeighted(gpu_img, 1.0 - intensity, gpu_filtered, intensity, 0.0)


def apply_laplacian_boost(gpu_img, boost_strength):
    """CUDA Laplacian edge boost. gpu_img must be a BGR GpuMat. Returns a GpuMat."""
    gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
    lap_filter = cv2.cuda.createLaplacianFilter(cv2.CV_8UC1, cv2.CV_8UC1, 3)
    gpu_lap = lap_filter.apply(gpu_gray)
    gpu_lap_abs = cv2.cuda.abs(gpu_lap)
    gpu_lap_bgr = cv2.cuda.cvtColor(gpu_lap_abs, cv2.COLOR_GRAY2BGR)
    return cv2.cuda.addWeighted(gpu_img, 1.0, gpu_lap_bgr, boost_strength, 0.0)


# ---------------------------------------------------------------------------
# Easing helpers
# ---------------------------------------------------------------------------

def ease_in(t):
    """Cosine ease-in: 0.0 at t=0, 1.0 at t=1."""
    return (1.0 - np.cos(t * np.pi)) / 2.0


def ease_out(t):
    """Cosine ease-out: 1.0 at t=0, 0.0 at t=1."""
    return (1.0 + np.cos(t * np.pi)) / 2.0


# ---------------------------------------------------------------------------
# Per-slide geometry + GPU pre-processing
# ---------------------------------------------------------------------------

def compute_slide_params(img, drift_value, direction, smooth=None, sharpen=None,
                         smooth_intensity=None, target_w=TARGET_W, target_h=TARGET_H):
    """
    Compute crop geometry for one slide and, when drift is disabled, pre-resize
    to the target resolution on the GPU (done once per slide).
    """
    h, w = img.shape[:2]
    target_aspect = target_w / target_h

    wide_w = w
    wide_h = int(w / target_aspect)
    if wide_h > h:
        wide_h = h
        wide_w = int(h * target_aspect)
    wide_x = (w - wide_w) // 2
    wide_y = (h - wide_h) // 2

    if drift_value == 0.0:
        gpu = cv2.cuda_GpuMat()
        crop = img[wide_y:wide_y + wide_h, wide_x:wide_x + wide_w]
        gpu.upload(crop)
        gpu_resized = cv2.cuda.resize(gpu, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        if smooth:
            gpu_resized = apply_bilateral_smooth(gpu_resized, smooth, smooth_intensity)
        if sharpen:
            gpu_resized = apply_laplacian_boost(gpu_resized, SHARPEN_PRESETS[sharpen])
        return {
            'drift': False,
            'smooth': smooth,
            'sharpen': sharpen,
            'smooth_intensity': smooth_intensity,
            'gpu_frame': gpu_resized,
            'cpu_frame': gpu_resized.download(),
        }

    scale = 1.0 - drift_value * 0.15
    tight_w = int(wide_w * scale)
    tight_h = int(wide_h * scale)
    tight_x = (w - tight_w) // 2
    tight_y = int((h - tight_h) * 0.3)

    if direction:  # zoom in: wide → tight
        start = (wide_w, wide_h, wide_x, wide_y)
        end = (tight_w, tight_h, tight_x, tight_y)
    else:  # zoom out: tight → wide
        start = (tight_w, tight_h, tight_x, tight_y)
        end = (wide_w, wide_h, wide_x, wide_y)

    return {
        'drift': True,
        'smooth': smooth,
        'sharpen': sharpen,
        'smooth_intensity': smooth_intensity,
        'img': img,
        'start': start,
        'end': end,
    }


def get_gpu_frame(params, frame_idx, hold_frames, gpu_scratch,
                  target_w=TARGET_W, target_h=TARGET_H):
    """
    Return a GPU mat for this slide at position frame_idx within [0, hold_frames).
    smooth/sharpen settings are read from params so per-image values are respected.
    """
    if not params['drift']:
        return params['gpu_frame']

    t = frame_idx / max(hold_frames - 1, 1)
    t_s = (1.0 - np.cos(t * np.pi)) / 2.0

    w0, h0, x0, y0 = params['start']
    w1, h1, x1, y1 = params['end']

    ar = target_w / target_h
    cw_f = w0 + (w1 - w0) * t_s
    ch_f = cw_f / ar

    cx_f = (x0 + w0 / 2.0) + ((x1 + w1 / 2.0) - (x0 + w0 / 2.0)) * t_s
    cy_f = (y0 + h0 / 2.0) + ((y1 + h1 / 2.0) - (y0 + h0 / 2.0)) * t_s

    icw = round(cw_f)
    ich = round(ch_f)
    icx = round(cx_f - icw / 2.0)
    icy = round(cy_f - ich / 2.0)

    crop = params['img'][icy:icy + ich, icx:icx + icw]
    gpu_scratch.upload(crop)
    gpu_frame = cv2.cuda.resize(gpu_scratch, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    smooth = params.get('smooth')
    sharpen = params.get('sharpen')
    if smooth:
        gpu_frame = apply_bilateral_smooth(gpu_frame, smooth, params.get('smooth_intensity'))
    if sharpen:
        gpu_frame = apply_laplacian_boost(gpu_frame, SHARPEN_PRESETS[sharpen])
    return gpu_frame


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

def collect_images(img_path):
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(img_path, ext)))
    files.sort()
    return files


# ---------------------------------------------------------------------------
# Filter map generator
# ---------------------------------------------------------------------------

def _staircase_score(gray):
    """
    Edge-direction discontinuity score for a greyscale thumbnail (numpy uint8).

    Returns a float ≥ 0.  Higher = more staircase-like zigzag on edges.
    Smooth diagonals → near 0.  ESRGAN H/V staircasing → elevated score.

    Algorithm:
      - Canny edges → binary edge mask
      - Sobel X+Y → gradient angle at every edge pixel
      - 8-connected flood-fill to group edge pixels into chains
      - Per-chain: circular variance of angles (handles 0°/180° wrap correctly)
      - Final score: chain-length-weighted mean circular variance × 100
    """
    edges = cv2.Canny(gray, 50, 150)
    if not edges.any():
        return 0.0

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    angle_map = np.arctan2(sobely, sobelx)   # radians, -π to π

    # Label connected edge components (8-connectivity)
    edge_u8 = (edges > 0).astype(np.uint8)
    n_labels, label_map = cv2.connectedComponents(edge_u8, connectivity=8)

    total_weight = 0.0
    weighted_var = 0.0

    for label in range(1, n_labels):
        ys, xs = np.where(label_map == label)
        chain_len = len(ys)
        if chain_len < 4:          # ignore tiny fragments
            continue
        angles = angle_map[ys, xs]  # in radians
        # Circular variance for angles: use doubled angle to handle ±π wrap
        # (edge direction is axial: 0° == 180°, so double to map onto full circle)
        doubled = 2.0 * angles
        mean_sin = np.sin(doubled).mean()
        mean_cos = np.cos(doubled).mean()
        circ_var = 1.0 - np.sqrt(mean_sin**2 + mean_cos**2)  # 0=uniform dir, 1=all same
        # Invert: high circ_var means consistent direction (not staircasing)
        # We want HIGH score = high directional INCONSISTENCY
        weighted_var += circ_var * chain_len
        total_weight  += chain_len

    if total_weight == 0:
        return 0.0
    return (weighted_var / total_weight) * 100.0


def _score_to_preset(score, threshold):
    """Map a staircase score to a smooth preset based on distance above threshold."""
    delta = score - threshold
    if delta <= 15:
        return 'subtle'
    elif delta <= 30:
        return 'light'
    elif delta <= 45:
        return 'medium'
    else:
        return 'strong'


def generate_filter_map(img_path, threshold, smooth=None, sharpen=None):
    """
    Analyse each image for ESRGAN staircase artefacts and write filter_map.json.

    Detection method: edge-direction discontinuity.
    ESRGAN staircasing on smooth diagonal contours (body outlines, etc.) creates
    edges that zigzag between 0° and 90° instead of holding a consistent angle.
    The detector:
      1. Canny edge detection on a thumbnail
      2. Sobel X+Y at each edge pixel → local gradient angle
      3. Divide edges into short chains; measure angle variance per chain
      4. Score = mean angle variance across all chains, weighted by chain length
    Clean diagonal edges score near 0; staircase edges score high.
    Smooth regions, horizontal lines, and vertical lines score near 0 (they are
    internally consistent — only H/V zigzag transitions score high).

    If smooth/sharpen are given, all flagged images get those exact presets.
    Otherwise preset is chosen by score band:
      threshold+0  to threshold+15 → subtle
      threshold+15 to threshold+30 → light
      threshold+30 to threshold+45 → medium
      threshold+45 and above       → strong
    Images scoring AT OR BELOW threshold → null (no filtering).
    """
    bc = Bcolors()
    image_files = collect_images(img_path)
    N = len(image_files)
    if N == 0:
        print(f"{bc.BOLD + bc.Red_f}No images found in {img_path}{bc.ENDC}")
        return

    sep = f"{bc.BOLD + bc.Dark_Gray_f}{'─' * 70}{bc.ENDC}"
    print()
    print(f"{bc.BOLD + bc.White_f}  Fade  {bc.Magenta_f}─{bc.OKBLUE}  Filter Map Generator{bc.ENDC}")
    print(sep)
    print(f"  {bc.BOLD + bc.Magenta_f}Source      {bc.White_f}: {bc.OKBLUE}{img_path}{bc.ENDC}")
    print(f"  {bc.BOLD + bc.Magenta_f}Images      {bc.White_f}: {bc.OKGREEN}{N}{bc.ENDC}")
    print(f"  {bc.BOLD + bc.Magenta_f}Threshold   {bc.White_f}: {bc.OKBLUE}{threshold:.0f}{bc.ENDC}"
          f"  {bc.Dark_Gray_f}(images above → flagged){bc.ENDC}")
    if smooth or sharpen:
        override_parts = []
        if smooth:   override_parts.append(f"smooth={smooth}")
        if sharpen:  override_parts.append(f"sharpen={sharpen}")
        print(f"  {bc.BOLD + bc.Magenta_f}Suggestion  {bc.White_f}: {bc.Yellow_f}override  "
              f"{bc.OKBLUE}{',  '.join(override_parts)}{bc.Dark_Gray_f}"
              f"  (all flagged images get these presets){bc.ENDC}")
    else:
        print(f"  {bc.BOLD + bc.Magenta_f}Suggestion  {bc.White_f}: {bc.Dark_Gray_f}score-based  "
              f"(+0→subtle  +15→light  +30→medium  +45→strong){bc.ENDC}")
    print(sep)
    print(f"  {bc.Dark_Gray_f}Analysing …{bc.ENDC}")
    print()

    scores = []
    for path in image_files:
        img = cv2.imread(path)
        if img is None:
            scores.append((os.path.basename(path), None))
            continue
        h, w = img.shape[:2]
        scale = 512.0 / max(w, h)
        thumb = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
        gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
        score = _staircase_score(gray)
        scores.append((os.path.basename(path), score))

    scores_sorted = sorted(scores, key=lambda x: (x[1] is None, x[1] or 0))

    flagged = sum(1 for _, v in scores if v is not None and v > threshold)
    print(f"  {bc.BOLD + bc.White_f}{'Filename':<45} {'Score':>8}  {'Action'}{bc.ENDC}")
    print(sep)
    for fname, var in scores_sorted:
        if var is None:
            score_str = f"{bc.GD_RED}{'unreadable':>8}{bc.ENDC}"
            action_str = f"{bc.Dark_Gray_f}skip{bc.ENDC}"
        elif var > threshold:
            eff_smooth  = smooth  if smooth  else _score_to_preset(var, threshold)
            eff_sharpen = sharpen if sharpen else None
            score_str = f"{bc.BOLD + bc.Yellow_f}{var:>8.0f}{bc.ENDC}"
            action_parts = [f"smooth={eff_smooth}"]
            if eff_sharpen:
                action_parts.append(f"sharpen={eff_sharpen}")
            action_str = f"{bc.OKBLUE}{',  '.join(action_parts)}{bc.ENDC}"
        else:
            score_str = f"{bc.Dark_Gray_f}{var:>8.0f}{bc.ENDC}"
            action_str = f"{bc.Dark_Gray_f}none{bc.ENDC}"
        print(f"  {bc.OKBLUE}{fname:<45}{bc.ENDC} {score_str}  {action_str}")

    print(sep)
    print(f"  {bc.BOLD + bc.Magenta_f}Flagged     {bc.White_f}: {bc.Yellow_f}{flagged}{bc.ENDC}"
          f"{bc.Dark_Gray_f} of {N} images above threshold{bc.ENDC}")

    # Build and write filter_map.json (keyed by basename, sorted by filename)
    # JSON doesn't support comments natively; use a reserved key as a header
    if smooth or sharpen:
        suggestion_note = f"override: smooth={smooth or 'none'},  sharpen={sharpen or 'none'}"
    else:
        suggestion_note = "score-based (+0=subtle +15=light +30=medium +45=strong)"
    filter_map = {"_info": f"filter_map for: {img_path}  |  threshold: {threshold}  |  "
                           f"suggestion: {suggestion_note}"}
    for fname, var in sorted(scores, key=lambda x: x[0]):
        if var is not None and var > threshold:
            eff_smooth  = smooth  if smooth  else _score_to_preset(var, threshold)
            eff_sharpen = sharpen if sharpen else None
            entry = {'smooth': eff_smooth}
            if eff_sharpen:
                entry['sharpen'] = eff_sharpen
            filter_map[fname] = entry
        else:
            filter_map[fname] = None

    out_path = os.path.join(img_path, 'filter_map.json')
    with open(out_path, 'w') as fh:
        json.dump(filter_map, fh, indent=2)

    print(f"\n  {bc.BOLD + bc.White_f}Written     {bc.White_f}: {bc.OKBLUE}{out_path}{bc.ENDC}")
    print(f"  {bc.Dark_Gray_f}Edit null entries or flagged entries as needed, then render with:{bc.ENDC}")
    print(f"  {bc.OKBLUE}  --img_path {img_path} --use_filter_map{bc.ENDC}")
    print()


# ---------------------------------------------------------------------------
# Main render pipeline
# ---------------------------------------------------------------------------

def render_video(image_files, output_path, duration_sec, fps, trans_time_sec,
                 transition_types, drift_value, rand_drift, use_hevc,
                 smooth=None, sharpen=None, smooth_intensity=None, filter_map_data=None,
                 metadata_comment=None, target_w=TARGET_W, target_h=TARGET_H):
    """
    Direct GPU-accelerated pipeline with optimized on-device blending.
    """
    bc = Bcolors()
    N = len(image_files)
    hold_frames = int(duration_sec * fps)
    trans_frames = int(trans_time_sec * fps)

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

    # Reusable GPU structures
    gpu_scratch_a = cv2.cuda_GpuMat()
    gpu_scratch_b = cv2.cuda_GpuMat()

    # Pre-initialize a static black frame on the GPU to handle hardware-accelerated fades
    gpu_black = cv2.cuda_GpuMat(target_h, target_w, cv2.CV_8UC3, (0, 0, 0))

    if rand_drift:
        half = N // 2
        bag = [True] * (N - half) + [False] * half
        random.shuffle(bag)
        drift_directions = bag
    else:
        drift_directions = [(i % 2 == 0) for i in range(N)]

    slide_cache = {}

    def load_slide(idx):
        if idx in slide_cache:
            return slide_cache[idx]
        img = cv2.imread(image_files[idx])
        if img is None:
            return None
        direction = drift_directions[idx]
        # Per-image filter settings: map entry overrides global; null in map = no filter
        fname = os.path.basename(image_files[idx])
        if filter_map_data is not None and fname in filter_map_data:
            entry = filter_map_data[fname]
            img_smooth  = entry.get('smooth')  if entry else None
            img_sharpen = entry.get('sharpen') if entry else None
        else:
            img_smooth, img_sharpen = smooth, sharpen
        params = compute_slide_params(img, drift_value, direction, img_smooth, img_sharpen,
                                      smooth_intensity, target_w, target_h)
        slide_cache[idx] = params
        stale = idx - 2
        if stale in slide_cache:
            del slide_cache[stale]
        return params

    try:
        for i in range(N):
            if i < N - 1:
                t_type = transition_types[i]
                arrow = (f"  {bc.Dark_Gray_f}→ {bc.Cyan_f}dissolve{bc.ENDC}"
                         if t_type == 'crossfade' else
                         f"  {bc.Dark_Gray_f}→ {bc.Yellow_f}fade-to-black{bc.ENDC}")
            else:
                arrow = ""
            print(f"  {bc.Yellow_f}[{i + 1:3d}/{N}]{bc.Magenta_f}  "
                  f"{bc.OKBLUE}{os.path.basename(image_files[i])}{arrow}{bc.ENDC}")

            params_i = load_slide(i)
            if params_i is None:
                print(f"  {bc.GD_RED}Warning: could not load image {i + 1}, skipping.{bc.ENDC}")
                continue

            incoming = 'opening' if i == 0 else transition_types[i - 1]
            outgoing = 'closing' if i == N - 1 else transition_types[i]

            params_next = None
            if outgoing == 'crossfade':
                params_next = load_slide(i + 1)

            # Frame boundary management
            frame_start = trans_frames if incoming == 'crossfade' else 0

            for f in range(frame_start, hold_frames):

                # ---- Opening / FTB incoming: Hardware-accelerated fade-in ----
                if incoming in ('opening', 'ftb') and f < trans_frames:
                    gpu_f = get_gpu_frame(params_i, f, hold_frames, gpu_scratch_a, target_w, target_h)
                    alpha = ease_in(f / trans_frames)
                    blended = cv2.cuda.addWeighted(gpu_f, alpha, gpu_black, 1.0 - alpha, 0.0)
                    output = blended.download()

                # ---- FTB / closing outgoing: Hardware-accelerated fade-out ----
                elif outgoing in ('ftb', 'closing') and f >= hold_frames - trans_frames:
                    gpu_f = get_gpu_frame(params_i, f, hold_frames, gpu_scratch_a, target_w, target_h)
                    t_out = (f - (hold_frames - trans_frames)) / trans_frames
                    alpha = ease_out(t_out)
                    blended = cv2.cuda.addWeighted(gpu_f, alpha, gpu_black, 1.0 - alpha, 0.0)
                    output = blended.download()

                # ---- Crossfade outgoing: Pure on-device GPU matrix blend ----
                elif outgoing == 'crossfade' and f >= hold_frames - trans_frames:
                    t_idx = f - (hold_frames - trans_frames)
                    alpha_b = ease_in(t_idx / trans_frames)
                    alpha_a = 1.0 - alpha_b

                    gpu_f = get_gpu_frame(params_i, f, hold_frames, gpu_scratch_a, target_w, target_h)
                    gpu_next = get_gpu_frame(params_next, t_idx, hold_frames, gpu_scratch_b, target_w, target_h)
                    blended = cv2.cuda.addWeighted(gpu_f, alpha_a, gpu_next, alpha_b, 0.0)
                    output = blended.download()

                # ---- Pure hold (Zero-copy optimization for static layers) ----
                else:
                    if params_i['drift']:
                        gpu_f = get_gpu_frame(params_i, f, hold_frames, gpu_scratch_a, target_w, target_h)
                        output = gpu_f.download()
                    else:
                        output = params_i['cpu_frame']

                proc.stdin.write(output)

    except BrokenPipeError:
        print(f"\n{bc.GD_RED}Pipeline error: ffmpeg exited unexpectedly.{bc.ENDC}")
        proc.wait()
        return False

    proc.stdin.close()
    return proc.wait() == 0


# ---------------------------------------------------------------------------
# Duration calculator
# ---------------------------------------------------------------------------

def calculate_report(opts, img_path):
    bc = Bcolors()
    image_files = collect_images(img_path)
    N = len(image_files)

    duration = opts.duration
    trans_time = opts.trans_time
    fps = opts.fps
    drift = opts.drift

    crossfade_total = N * duration - (N - 1) * trans_time
    ftb_total = N * duration
    visible_hold = duration - 2.0 * trans_time

    if opts.random:
        mode_str = "Random (crossfade / fade-to-black per boundary)"
        mins_cf, secs_cf = divmod(int(crossfade_total), 60)
        mins_ftb, secs_ftb = divmod(int(ftb_total), 60)
        dur_str = (f"{bc.BOLD + bc.OKGREEN}~{(crossfade_total + ftb_total) / 2:.0f}s{bc.ENDC}  "
                   f"{bc.Dark_Gray_f}(range {mins_cf:02d}:{secs_cf:02d} – "
                   f"{mins_ftb:02d}:{secs_ftb:02d}  |  {N} slides){bc.ENDC}")
    elif opts.ftb:
        mode_str = "Fade-to-black"
        mins, secs = divmod(int(ftb_total), 60)
        dur_str = (f"{bc.BOLD + bc.OKGREEN}{mins:02d}:{secs:02d}{bc.White_f}  "
                   f"{bc.Dark_Gray_f}({ftb_total:.0f}s  |  {N} slides){bc.ENDC}")
    else:
        mode_str = "Crossfade dissolve"
        mins, secs = divmod(int(crossfade_total), 60)
        dur_str = (f"{bc.BOLD + bc.OKGREEN}{mins:02d}:{secs:02d}{bc.White_f}  "
                   f"{bc.Dark_Gray_f}({crossfade_total:.0f}s  |  {N} slides){bc.ENDC}")

    if drift > 0:
        dir_note = f"  {bc.Dark_Gray_f}random direction{bc.ENDC}" if opts.rand_drift else f"  {bc.Dark_Gray_f}alternating direction{bc.ENDC}"
        drift_str = (f"{bc.OKBLUE}{drift:.2f}{bc.White_f}  "
                     f"{bc.Dark_Gray_f}(~{drift * 15:.1f}% crop change){bc.ENDC}{dir_note}")
    else:
        drift_str = f"{bc.Dark_Gray_f}disabled{bc.ENDC}"

    sep = f"{bc.BOLD + bc.Dark_Gray_f}{'─' * 58}{bc.ENDC}"
    print()
    print(f"{bc.BOLD + bc.White_f}  Fade  {bc.Magenta_f}─{bc.OKBLUE}  Duration Calculator{bc.ENDC}")
    print(sep)
    print(f"  {bc.BOLD + bc.Magenta_f}Source Path      {bc.White_f}: {bc.OKBLUE}{img_path}{bc.ENDC}")
    print(f"  {bc.BOLD + bc.Magenta_f}Images Found     {bc.White_f}: {bc.OKGREEN}{N}{bc.ENDC}")
    print(sep)
    print(f"  {bc.BOLD + bc.Magenta_f}Duration/Slide   {bc.White_f}: {bc.OKBLUE}{duration}s{bc.ENDC}")
    print(f"  {bc.BOLD + bc.Magenta_f}Trans Time       {bc.White_f}: {bc.OKBLUE}{trans_time}s{bc.ENDC}")
    print(f"  {bc.BOLD + bc.Magenta_f}FPS              {bc.White_f}: {bc.OKBLUE}{fps}{bc.ENDC}")
    print(f"  {bc.BOLD + bc.Magenta_f}Transition Mode  {bc.White_f}: {bc.OKBLUE}{mode_str}{bc.ENDC}")
    print(f"  {bc.BOLD + bc.Magenta_f}Drift            {bc.White_f}: {drift_str}")
    print(sep)

    if visible_hold <= 0:
        print(f"  {bc.BOLD + bc.Red_f}WARNING: trans_time ({trans_time}s × 2 = {2 * trans_time}s) "
              f"meets or exceeds duration ({duration}s){bc.ENDC}")
    else:
        print(f"  {bc.BOLD + bc.Magenta_f}Visible Hold     {bc.White_f}: {bc.OKBLUE}{visible_hold:.1f}s{bc.White_f}"
              f" per slide  {bc.Dark_Gray_f}(duration − 2 × trans_time){bc.ENDC}")

    print(f"  {bc.BOLD + bc.White_f}Total Duration   {bc.White_f}: {dur_str}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bc = Bcolors()
    opts, src_path, dest_vid = cliOpts()

    if opts.calculate:
        calculate_report(opts, src_path)
        sys.exit(0)

    if opts.gen_filter_map:
        generate_filter_map(src_path, opts.fmap_threshold, opts.smooth, opts.sharpen)
        sys.exit(0)

    image_files = collect_images(src_path)
    N = len(image_files)
    if N == 0:
        print(f"{bc.BOLD + bc.Red_f}No valid images found in: {bc.OKBLUE}{src_path}{bc.ENDC}")
        sys.exit(1)

    use_hevc = not opts.h264
    duration_sec = opts.duration
    fps = opts.fps
    trans_time = opts.trans_time
    drift_value = opts.drift

    if opts.random:
        transition_types = [random.choice(['crossfade', 'ftb']) for _ in range(N - 1)]
        mode_label = f"{bc.OKBLUE}Random{bc.ENDC}"
    elif opts.ftb:
        transition_types = ['ftb'] * (N - 1)
        mode_label = f"{bc.Yellow_f}Fade-to-black{bc.ENDC}"
    else:
        transition_types = ['crossfade'] * (N - 1)
        mode_label = f"{bc.Cyan_f}Crossfade dissolve{bc.ENDC}"

    if drift_value > 0:
        dir_label = (f"  {bc.Dark_Gray_f}(random direction){bc.ENDC}" if opts.rand_drift
                     else f"  {bc.Dark_Gray_f}(alternating direction){bc.ENDC}")
        drift_label = f"{bc.OKBLUE}{drift_value:.2f}{dir_label}"
    else:
        drift_label = f"{bc.Dark_Gray_f}disabled{bc.ENDC}"

    sep = f"{bc.BOLD + bc.Dark_Gray_f}{'─' * 58}{bc.ENDC}"
    print()
    print(f"{bc.BOLD + bc.White_f}  Fade  {bc.Magenta_f}─{bc.OKBLUE}  Direct GPU Pipeline{bc.ENDC}")
    print(sep)
    print(f"  {bc.BOLD + bc.Magenta_f}Source       {bc.White_f}: {bc.OKBLUE}{src_path}{bc.ENDC}")
    print(f"  {bc.BOLD + bc.Magenta_f}Output       {bc.White_f}: {bc.OKBLUE}{dest_vid}{bc.ENDC}")
    print(f"  {bc.BOLD + bc.Magenta_f}Images       {bc.White_f}: {bc.OKGREEN}{N}{bc.ENDC}")
    print(f"  {bc.BOLD + bc.Magenta_f}Transition   {bc.White_f}: {mode_label}")
    print(f"  {bc.BOLD + bc.Magenta_f}Trans Time   {bc.White_f}: {bc.OKBLUE}{trans_time}s{bc.ENDC}")
    print(
        f"  {bc.BOLD + bc.Magenta_f}Duration     {bc.White_f}: {bc.OKBLUE}{duration_sec}s{bc.White_f} per slide{bc.ENDC}")
    print(f"  {bc.BOLD + bc.Magenta_f}FPS          {bc.White_f}: {bc.OKBLUE}{fps}{bc.ENDC}")
    print(f"  {bc.BOLD + bc.Magenta_f}Drift        {bc.White_f}: {drift_label}")
    smooth = opts.smooth
    smooth_intensity = opts.smooth_intensity  # None = use preset default

    # Load per-image filter map if --use_filter_map requested
    filter_map_data = None
    filter_map_label = f"{bc.Dark_Gray_f}off{bc.ENDC}"
    if opts.use_filter_map:
        fmap_path = os.path.join(src_path, 'filter_map.json')
        if not os.path.isfile(fmap_path):
            print(f"\n{bc.BOLD + bc.Red_f}  filter_map.json not found in --img_path.{bc.ENDC}")
            print(f"  {bc.Dark_Gray_f}Generate one first:{bc.ENDC}")
            print(f"  {bc.OKBLUE}  python fade.py --img_path {src_path} --gen_filter_map{bc.ENDC}\n")
            sys.exit(1)
        with open(fmap_path) as fh:
            raw = json.load(fh)
        # Strip the _info header key before use
        filter_map_data = {k: v for k, v in raw.items() if not k.startswith('_')}
        flagged = sum(1 for v in filter_map_data.values() if v is not None)
        filter_map_label = (f"{bc.OKBLUE}filter_map.json{bc.ENDC}"
                            f"  {bc.Dark_Gray_f}({flagged} of {len(filter_map_data)} images flagged){bc.ENDC}")

    # Build smooth label — note if per-image map overrides are active
    if smooth:
        p = SMOOTH_PRESETS[smooth]
        eff_intensity = smooth_intensity if smooth_intensity is not None else p['intensity']
        smooth_label = (f"{bc.OKBLUE}{smooth}{bc.Dark_Gray_f}  "
                        f"(d={p['d']}, sigmaColor={p['sigmaColor']}, sigmaSpace={p['sigmaSpace']}"
                        f", intensity={eff_intensity*100:.0f}%){bc.ENDC}")
    elif filter_map_data is not None and flagged > 0:
        smooth_label = (f"{bc.Dark_Gray_f}off (global){bc.ENDC}"
                        f"  {bc.Yellow_f}per-image settings active via filter map{bc.ENDC}")
    else:
        smooth_label = f"{bc.Dark_Gray_f}off{bc.ENDC}"
    print(f"  {bc.BOLD + bc.Magenta_f}Smooth       {bc.White_f}: {smooth_label}")

    sharpen = opts.sharpen
    if sharpen:
        sharpen_label = (f"{bc.OKBLUE}{sharpen}{bc.Dark_Gray_f}  "
                         f"(boost_strength={SHARPEN_PRESETS[sharpen]:.2f}){bc.ENDC}")
    else:
        sharpen_label = f"{bc.Dark_Gray_f}off{bc.ENDC}"
    print(f"  {bc.BOLD + bc.Magenta_f}Sharpen      {bc.White_f}: {sharpen_label}")
    print(f"  {bc.BOLD + bc.Magenta_f}Filter Map   {bc.White_f}: {filter_map_label}")

    print(f"  {bc.BOLD + bc.Magenta_f}Codec        {bc.White_f}: {bc.OKBLUE}"
          f"{'HEVC (H.265 via NVENC)' if use_hevc else 'H.264 (via NVENC)'}{bc.ENDC}")
    print(sep)
    print()

    # Build metadata comment for MP4 container
    trans_mode = 'random' if opts.random else ('ftb' if opts.ftb else 'crossfade')
    drift_str = f"{drift_value:.2f}({'random' if opts.rand_drift else 'alt'})" if drift_value > 0 else 'off'
    fmap_meta = 'on' if opts.use_filter_map else 'off'
    metadata_comment = (
        f"fade.py"
        f" | Slides: {N}"
        f" | Duration: {duration_sec}s"
        f" | Trans: {trans_mode}"
        f" | Trans_time: {trans_time}s"
        f" | FPS: {fps}"
        f" | Drift: {drift_str}"
        f" | Smooth: {smooth or 'off'}"
          + (f"@{smooth_intensity*100:.0f}%" if smooth and smooth_intensity is not None else "")
        + f" | Sharpen: {sharpen or 'off'}"
        f" | FilterMap: {fmap_meta}"
        f" | Codec: {'HEVC' if use_hevc else 'H.264'}"
    )

    ok = render_video(image_files, dest_vid, duration_sec, fps, trans_time,
                      transition_types, drift_value, opts.rand_drift, use_hevc,
                      smooth, sharpen, smooth_intensity, filter_map_data, metadata_comment)

    if ok:
        print(f"\n{bc.BOLD + bc.Green_f}✨ Done.{bc.White_f}  Video saved to: {bc.OKBLUE}{dest_vid}{bc.ENDC}\n")
    else:
        print(f"\n{bc.BOLD + bc.Red_f}Render failed.{bc.ENDC}\n")
        sys.exit(1)