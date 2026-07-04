# Ken Burns — GPU-Accelerated Slideshow Renderer

A high-performance slideshow renderer that runs almost entirely on the GPU. Three motion modes, three transition styles, and an optional CUDA image-filtering pipeline — all piped directly into FFmpeg NVENC without touching system RAM per frame.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Motion Modes](#motion-modes)
- [Transition Styles](#transition-styles)
- [Image Filters](#image-filters)
- [Filter Map System](#filter-map-system)
- [CLI Reference](#cli-reference)
- [Examples](#examples)
- [Under the Hood — CUDA Pipeline](#under-the-hood--cuda-pipeline)
- [Hardware Recommendations](#hardware-recommendations)

---

## Features

- **Ken Burns zoom** — full zoom-in/out with streak-based alternating direction
- **Crossfade dissolve** — smooth GPU-blended overlap between consecutive slides
- **Fade to black** — each slide fades to black; the next fades in from black
- **CUDA bilateral filter** — suppresses ESRGAN/upscaler staircase artefacts on-device
- **CUDA Laplacian sharpening** — recovers edge definition after smoothing, on-device
- **Per-image filter map** — apply different filter presets to individual images
- **HEVC or H.264** via NVENC — hardware encoding, no software codec overhead
- **Zero PCIe transfers per frame** — the entire pipeline from crop to encode stays on the GPU

---

## Requirements

| Component | Minimum |
|-----------|---------|
| Python | 3.8+ |
| OpenCV | Built with CUDA support (`cv2.cuda` module present) |
| NVIDIA GPU | Maxwell or newer (GTX 9xx+), NVENC capable |
| CUDA toolkit | 11.x+ |
| FFmpeg | Built with `hevc_nvenc` / `h264_nvenc` |
| numpy | Any recent version |

```bash
# Verify your OpenCV has CUDA support
python -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
# Should print 1 or more

# Verify FFmpeg has NVENC
ffmpeg -hide_banner -encoders | grep nvenc
```

---

## Quick Start

```bash
# Crossfade dissolve (default) — 8s per slide, 1s crossfade, HEVC
python ken_burns.py --img_path ~/photos/holiday -o ~/Videos/holiday.mp4

# Classic Ken Burns zoom + fade to black
python ken_burns.py --img_path ~/photos/holiday -o ~/Videos/holiday.mp4 \
    --ken_burns --ftb

# Ken Burns starting zoom-in, crossfade transitions
python ken_burns.py --img_path ~/photos/holiday -o ~/Videos/holiday.mp4 \
    --ken_burns --zoom_in

# Check estimated duration before rendering
python ken_burns.py --img_path ~/photos/holiday --calculate --ken_burns --ftb
```

---

## Motion Modes

### Crossfade / Static Hold (default)

Without `--ken_burns` or `--drift`, each slide is a static hold — the image fills the frame with no movement. Transitions are crossfade dissolves by default.

```bash
python ken_burns.py --img_path ~/imgs -o out.mp4
```

### Drift (`--drift`)

A subtle, configurable per-slide zoom. The image slowly drifts in or out by a small crop amount, giving a gentle sense of motion without the full Ken Burns effect.

```
0.0      = disabled (pure static hold)
0.33     = ~5% crop change — visible but subtle (recommended starting point)
1.0      = ~15% crop change (equivalent to full Ken Burns geometry)
```

Direction alternates slide-by-slide by default. Add `--rand_drift` to randomise per slide independently.

```bash
python ken_burns.py --img_path ~/imgs -o out.mp4 --drift 0.33 --rand_drift
```

### Ken Burns (`--ken_burns`)

Full zoom-in/zoom-out per slide (~15% crop change). Direction is governed by a streak-based algorithm:

- 70% chance to flip direction on each new slide
- Never more than 3 consecutive slides in the same direction
- Starting direction controlled by `--zoom_in` (default: zoom-out first)

This reproduces the classic documentary zoom effect while avoiding monotonous repetition.

```bash
python ken_burns.py --img_path ~/imgs -o out.mp4 --ken_burns --zoom_in
```

> `--ken_burns` and `--drift` are mutually exclusive.

---

## Transition Styles

Transition style is orthogonal to motion mode — any combination works.

| Flag | Behaviour |
|------|-----------|
| *(none)* | **Crossfade dissolve** — slides overlap during the transition window |
| `--ftb` | **Fade to black** — current slide fades out, next fades in |
| `--random` | **Random** — each boundary independently picks crossfade or FTB |

Duration of the transition (overlap or fade window) is set with `--trans_time` (default 1.0s).

### Examples

```bash
# Crossfade (default), 1.5s overlap
python ken_burns.py --img_path ~/imgs -o out.mp4 --trans_time 1.5

# Fade to black, Ken Burns zoom
python ken_burns.py --img_path ~/imgs -o out.mp4 --ken_burns --ftb --trans_time 1.0

# Random transitions, subtle drift
python ken_burns.py --img_path ~/imgs -o out.mp4 --drift 0.25 --random
```

---

## Image Filters

Two optional CUDA filters can be applied per-image. They are applied immediately after the GPU resize to 1920×1080 — no CPU round-trip at any point.

### `--smooth` — CUDA Bilateral Filter

Suppresses staircase artefacts introduced by upscalers (Real-ESRGAN, etc.) while preserving real edges. The filter weights neighbouring pixels by both spatial distance and colour distance, so it smooths flat regions without blurring subject outlines.

| Preset | Diameter | Sigma Color | Sigma Space | Intensity |
|--------|----------|-------------|-------------|-----------|
| `subtle` | 3 | 15 | 15 | 40% |
| `light` | 5 | 35 | 35 | 50% |
| `medium` | 9 | 75 | 75 | 70% |
| `strong` | 13 | 120 | 120 | 85% |

**Recommended starting point for ESRGAN portraits:** `--smooth medium`

The `intensity` is a blend factor: `output = (1 - intensity) × original + intensity × filtered`. This lets you dial back the effect without changing the filter kernel. Override with `--smooth_intensity` (0–100).

### `--sharpen` — CUDA Laplacian Edge Boost

Recovers edge sharpness softened by the bilateral pass. Converts to greyscale, applies a 3×3 Laplacian kernel, then blends the absolute edge map back onto the BGR frame.

| Preset | boost_strength |
|--------|---------------|
| `subtle` | 0.10 |
| `light` | 0.25 |
| `medium` | 0.45 |
| `strong` | 0.65 |

### Order matters

Always `--smooth` before `--sharpen`. Reversing the order sharpens artefacts instead of suppressing them.

```bash
# Typical ESRGAN portrait set
python ken_burns.py --img_path ~/imgs -o out.mp4 \
    --smooth medium --sharpen light

# Heavy artefacts
python ken_burns.py --img_path ~/imgs -o out.mp4 \
    --smooth strong --sharpen medium

# Clean source — mild crispening only
python ken_burns.py --img_path ~/imgs -o out.mp4 --sharpen light
```

> **Performance note:** Bilateral filter cost scales with d². `light` (~25 ops/pixel) is fast; `strong` (~169 ops/pixel) is ~6.8× heavier. On Ken Burns and drift slides the filter runs every frame; on static slides it runs once at load time.

### When NOT to smooth

- Clean source (raw camera, lossless PNG) — no benefit, costs GPU time
- Fine repeating texture (brickwork, fabric weave) — bilateral can wash out periodic detail
- Test/proof renders where you want to see native upscaler output

### When NOT to sharpen

- Without `--smooth` on ESRGAN output — Laplacian amplifies the artefacts you are trying to suppress
- Already-sharpened source — double-sharpening produces haloing and ringing
- Noisy images — `boost_strength` above 0.25 makes noise dominant

---

## Filter Map System

Apply different filter settings to individual images in a set, rather than globally to every slide.

### Step 1 — Generate the map

```bash
python ken_burns.py --img_path ~/imgs/set_042 --gen_filter_map
```

Analyses every image with an edge-direction discontinuity detector (scores 0–100; higher = more staircase-like artefacts). Writes `filter_map.json` to the image directory and prints a score table sorted lowest to highest.

```bash
# Adjust sensitivity
python ken_burns.py --img_path ~/imgs/set_042 --gen_filter_map --fmap_threshold 25

# Pre-populate flagged entries with a specific preset
python ken_burns.py --img_path ~/imgs/set_042 --gen_filter_map --smooth light
```

Default threshold is 30. Images above the threshold are flagged with an auto-chosen smooth preset (subtle/light/medium/strong based on score distance from threshold). Images at or below threshold are set to `null`.

### Step 2 — Review and edit

```json
{
  "img_001.jpg": {"smooth": "light", "sharpen": "light"},
  "img_002.jpg": null,
  "img_003.jpg": {"smooth": "medium"},
  "img_004.jpg": null
}
```

- **Spec entry** — uses map `smooth` and/or `sharpen` (global flags ignored for this image)
- **`null` entry** — no filtering (global flags ignored for this image)
- **Not in map** — falls back to global `--smooth` / `--sharpen`

Sharpen is deliberately not auto-suggested — add it manually to specific entries, or use `--sharpen` at render time as a global fallback.

### Step 3 — Render with the map

```bash
python ken_burns.py --img_path ~/imgs/set_042 -o ~/out/set_042.mp4 \
    --ken_burns --ftb --use_filter_map

# With global fallback for images not in the map
python ken_burns.py --img_path ~/imgs/set_042 -o ~/out/set_042.mp4 \
    --use_filter_map --smooth subtle
```

Full filter map guidance: `python ken_burns.py --fmap_help`  
Full filter guidance: `python ken_burns.py --filter_help`

---

## CLI Reference

### File I/O

| Argument | Default | Description |
|----------|---------|-------------|
| `--img_path PATH` | *(required)* | Source directory of images (JPG, JPEG, PNG) |
| `-o / --output PATH` | *(required for render)* | Full output path including filename, e.g. `~/Videos/out.mp4` |

### Timing

| Argument | Default | Description |
|----------|---------|-------------|
| `-d / --duration SECS` | `8.0` | Per-slide display duration in seconds |
| `-f / --fps N` | `30` | Output frame rate |
| `--trans_time SECS` | `1.0` | Transition duration — crossfade overlap or FTB fade window per side |

### Ken Burns

| Argument | Default | Description |
|----------|---------|-------------|
| `--ken_burns` | off | Enable full Ken Burns zoom (~15% crop change, streak-based direction alternation). Mutually exclusive with `--drift`. |
| `-z / --zoom_in` | off | Start first slide zooming in (Ken Burns mode only; default is zoom-out first) |

### Transitions

| Argument | Default | Description |
|----------|---------|-------------|
| `--ftb` | off | Fade to black at each slide boundary. Mutually exclusive with `--random`. |
| `--random` | off | Randomly assign crossfade or FTB per boundary. Mutually exclusive with `--ftb`. |
| `--drift AMOUNT` | `0.0` | Subtle zoom drift per slide (0.0 = disabled, range 0.01–1.0). Mutually exclusive with `--ken_burns`. |
| `--rand_drift` | off | Randomise per-slide zoom direction when using `--drift`. Default is alternating. |

### Image Processing

| Argument | Default | Description |
|----------|---------|-------------|
| `--smooth PRESET` | off | CUDA bilateral filter. Choices: `subtle` `light` `medium` `strong` |
| `--sharpen PRESET` | off | CUDA Laplacian edge boost. Choices: `subtle` `light` `medium` `strong` |
| `--smooth_intensity 0-100` | preset value | Override bilateral blend intensity without changing the filter kernel |
| `--use_filter_map` | off | Load `filter_map.json` from `--img_path` for per-image filter settings |

### Encoding

| Argument | Default | Description |
|----------|---------|-------------|
| `--h264` | off | Use H.264 via NVENC instead of default HEVC (H.265) |

### Utility

| Argument | Description |
|----------|-------------|
| `--calculate` | Print estimated video duration and settings without rendering |
| `--filter_help` | Comprehensive guidance on `--smooth`, `--sharpen`, and `--smooth_intensity` |
| `--fmap_help` | Comprehensive guidance on the filter map workflow |
| `--gen_filter_map` | Analyse images and write `filter_map.json` to `--img_path` |
| `--fmap_threshold 0-100` | Score threshold for `--gen_filter_map` (default: 30.0) |

---

## Examples

### Classic documentary slideshow

```bash
python ken_burns.py \
    --img_path ~/photos/project \
    -o ~/Videos/documentary.mp4 \
    --ken_burns \
    --ftb \
    --duration 10 \
    --trans_time 1.5 \
    --zoom_in
```

### Crossfade with subtle drift and ESRGAN smoothing

```bash
python ken_burns.py \
    --img_path ~/photos/esrgan_output \
    -o ~/Videos/smooth.mp4 \
    --drift 0.33 \
    --rand_drift \
    --smooth medium \
    --sharpen light \
    --duration 9
```

### Ken Burns + per-image filters

```bash
# 1. Generate filter map
python ken_burns.py --img_path ~/photos/set -gen_filter_map --smooth light

# 2. Edit filter_map.json as needed, then render
python ken_burns.py \
    --img_path ~/photos/set \
    -o ~/Videos/set.mp4 \
    --ken_burns \
    --ftb \
    --use_filter_map \
    --smooth subtle        # fallback for images not in the map
```

### Check duration before rendering

```bash
python ken_burns.py \
    --img_path ~/photos/holiday \
    --calculate \
    --ken_burns \
    --ftb \
    --duration 8 \
    --trans_time 1
```

Output:
```
  Ken Burns  ─  Duration Calculator
  ──────────────────────────────────────────────────────────
  Source Path      : /home/user/photos/holiday
  Images Found     : 47
  ──────────────────────────────────────────────────────────
  Duration/Slide   : 8s
  Trans Time       : 1s
  FPS              : 30
  Transition Mode  : Fade-to-black
  Zoom             : Ken Burns  (zoom-out first, streak-based alternation)
  ──────────────────────────────────────────────────────────
  Visible Hold     : 6.0s per slide  (duration − 2 × trans_time)
  Total Duration   : 06:16  (376s  |  47 slides)
```

### H.264 output (for wider compatibility)

```bash
python ken_burns.py \
    --img_path ~/photos/set \
    -o ~/Videos/out_h264.mp4 \
    --ken_burns --ftb --h264
```

---

## Under the Hood — CUDA Pipeline

This is where the project differs fundamentally from most Python slideshow tools. The render pipeline is designed to keep data on the GPU from the moment an image is decoded until the compressed frame exits the NVENC encoder. The CPU orchestrates but does not process pixels.

### 1. Image load (CPU → GPU, once per slide)

```
cv2.imread()  →  numpy array (CPU RAM)
    ↓
cv2.cuda_GpuMat.upload()  →  VRAM
```

This is the only PCIe transfer in the entire pipeline per slide. Every subsequent operation on that image happens in VRAM.

### 2. GPU crop and resize (zero-copy ROI)

For Ken Burns and drift slides, each frame requires a different crop window. OpenCV CUDA implements this as a **zero-copy ROI** — a view into the already-uploaded GpuMat rather than a new allocation:

```python
gpu_roi   = gpu_full.rowRange(y, y + h).colRange(x, x + w)   # zero-copy view
gpu_frame = cv2.cuda.resize(gpu_roi, (1920, 1080), interpolation=cv2.INTER_CUBIC)
```

The `INTER_CUBIC` resize is a GPU kernel. No data leaves the device.

For static slides (no drift, no Ken Burns), the crop and resize happen **once at load time** and the result is cached — subsequent frames reuse the cached GpuMat, so there is zero per-frame GPU work during holds.

### 3. Bilateral filter (optional, GPU-native)

```python
gpu_filtered = cv2.cuda.bilateralFilter(gpu_img, d, sigmaColor, sigmaSpace)
gpu_blended  = cv2.cuda.addWeighted(gpu_img, 1 - intensity, gpu_filtered, intensity, 0)
```

`cv2.cuda.bilateralFilter` is a direct CUDA kernel call. The intensity blend (`addWeighted`) is a second CUDA kernel. Both operate in VRAM — no CPU involvement.

**Performance:** CUDA bilateral filtering runs at 30–120+ FPS on modern cards vs. 5–15 FPS on CPU — a 6–25× advantage. On Ken Burns and drift slides the filter runs every frame. On static slides it runs once at load time.

Bilateral filter cost scales with `d²` (neighbourhood area). `strong` (d=13) is ~6.8× heavier than `light` (d=5).

### 4. Laplacian sharpening (optional, GPU-native)

```python
gpu_gray    = cv2.cuda.cvtColor(gpu_img, COLOR_BGR2GRAY)     # GPU colour space
lap_filter  = cv2.cuda.createLaplacianFilter(...)             # GPU filter object
gpu_lap     = lap_filter.apply(gpu_gray)                      # GPU kernel
gpu_lap_abs = cv2.cuda.abs(gpu_lap)                          # GPU op
gpu_lap_bgr = cv2.cuda.cvtColor(gpu_lap_abs, COLOR_GRAY2BGR) # GPU op
result      = cv2.cuda.addWeighted(gpu_img, 1.0, gpu_lap_bgr, strength, 0.0)
```

Six GPU operations, zero CPU operations, zero host memory access.

### 5. Transition blending (GPU-native)

All three transition types are implemented as GPU `addWeighted` calls:

```python
# Crossfade: blend current slide with next slide
blended = cv2.cuda.addWeighted(gpu_current, alpha_a, gpu_next, alpha_b, 0.0)

# Fade to black / opening: blend with pre-allocated black GpuMat
gpu_black = cv2.cuda_GpuMat(1080, 1920, cv2.CV_8UC3, (0, 0, 0))  # allocated once
blended   = cv2.cuda.addWeighted(gpu_frame, alpha, gpu_black, 1 - alpha, 0.0)
```

The black frame is allocated once and reused for the entire render. Easing curves (cosine ease-in / ease-out) are applied to the alpha value before the blend, computed in Python as a single float — the GPU only sees the resulting blend weights.

### 6. Frame output — GPU → FFmpeg pipe

```python
proc.stdin.write(gpu_frame.download())
```

`.download()` is the second and final PCIe transfer: the finished 1920×1080 BGR frame moves from VRAM to CPU RAM, then immediately into the FFmpeg stdin pipe. FFmpeg feeds it directly into the NVENC hardware encoder.

**The frame never enters Python NumPy processing.** The only CPU work is computing crop coordinates and blend alphas (scalar arithmetic).

### 7. NVENC hardware encoding

FFmpeg receives raw BGR24 frames and passes them to the NVENC fixed-function encoder on the GPU. The codec (HEVC or H.264), preset (`p4`), and quality (`cq 22`) are set at pipeline start:

```
-c:v hevc_nvenc -preset p4 -cq 22
```

`p4` is the balanced quality/speed preset. `cq 22` is constant-quality mode — visually lossless for slideshow content.

### Pipeline summary

```
Disk  →  CPU RAM  →  VRAM  →─────────────────────────────────────────┐
           imread    upload   crop → resize → filter → sharpen → blend │
                                                                        ↓
                                              CPU RAM  ←  download  ←──┘
                                                 ↓
                                            FFmpeg pipe  →  NVENC  →  MP4
```

The only data moving over PCIe is:
- **Upload:** one full image per slide (at slide load time)
- **Download:** one finished 1920×1080 frame per frame output

Everything in between — crop, resize, bilateral filter, Laplacian boost, transition blend — is VRAM-resident CUDA kernels.

---

## Hardware Recommendations

### GPU

Any CUDA-capable NVIDIA GPU with NVENC support works. Performance scales with CUDA core count and memory bandwidth.

| Tier | Cards | Notes |
|------|-------|-------|
| Excellent | RTX 40 series, RTX 30 series | Any preset, any filter, 4K capable |
| Good | RTX 20 series, GTX 16 series | `medium` filter at full 30fps |
| Adequate | GTX 10 series (1060 6GB+) | `light` filter; `strong` may bottleneck on drift/KB |
| Marginal | GTX 1050, older | `subtle` or no filter; static slides only for smooth encode |

### VRAM

- **4 GB:** Comfortable for 1080p output with all features
- **8 GB+:** Recommended if source images are large (24MP+) since the full image is uploaded per slide

### Checking your GPU

```bash
nvidia-smi
# Look for CUDA version and VRAM available

python -c "import cv2; d = cv2.cuda.getDevice(); \
    cv2.cuda.printCudaDeviceInfo(d)"
# Full CUDA device info
```

---

## File Structure

```
ken_burns/
├── ken_burns.py   # Main script — all render logic
├── cliOpts.py     # CLI argument parser
├── Bcolors.py     # ANSI terminal colour utility
└── README.md
```

Images can be JPG, JPEG, or PNG. Files are sorted alphabetically within `--img_path` to determine slide order. Name them numerically (`001.jpg`, `002.jpg`, …) to control sequence.

---

## License

GNU Lesser General Public License v3.0 — see [LICENSE](LICENSE).