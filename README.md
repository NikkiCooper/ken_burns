# Ken Burns — GPU-Accelerated Slideshow Renderer

A slideshow renderer that does its pixel work on the GPU with OpenCV CUDA and encodes with FFmpeg NVENC. It has three motion modes, three transition styles and an optional CUDA image-filtering pipeline. The default encode is HEVC 10-bit 4:4:4 (RExt), tuned for slow, sub-pixel zoom and drift motion.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Batch Processing](#batch-processing)
- [Motion Modes](#motion-modes)
- [Transition Styles](#transition-styles)
- [Image Filters](#image-filters)
- [Filter Map System](#filter-map-system)
- [Encoding](#encoding)
- [CLI Reference](#cli-reference)
- [Examples](#examples)
- [Under the Hood — Pipeline](#under-the-hood--pipeline)
- [Hardware Notes](#hardware-notes)

---

## Features

- **Ken Burns zoom**: full zoom-in/out with streak-based alternating direction
- **Sub-pixel motion**: the zoom/drift crop is never rounded to whole pixels; each frame is a single CUDA affine warp
- **Crossfade dissolve**: GPU-blended overlap between consecutive slides
- **Fade to black**: each slide fades to black and the next fades in from black
- **CUDA bilateral filter**: suppresses ESRGAN/upscaler staircase artefacts on the GPU
- **CUDA Laplacian sharpening**: recovers edge definition after smoothing, on the GPU
- **Per-image filter map**: apply different filter presets to individual images
- **HEVC 10-bit 4:4:4 (RExt) via NVENC by default**, with 10-bit 4:2:0, 8-bit 4:2:0 and H.264 for compatibility
- **`--dryRun` capability probe**: confirms your GPU, driver and FFmpeg can encode the chosen format before a long render

---

## Requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.8+ |
| OpenCV | Built with CUDA (`cv2.cuda` module present). Tested with OpenCV 5.x |
| NVIDIA GPU | CUDA-capable, with NVENC. For the default encode: 7th-gen NVENC or newer (GTX 1660, RTX 20 / 30 / 40 / 50, RTX A-series / Ada / Blackwell pro). See [Hardware Notes](#hardware-notes) |
| CUDA toolkit | Matching your OpenCV CUDA build |
| FFmpeg | Built with `hevc_nvenc` / `h264_nvenc` (tested with FFmpeg 9.0) |
| numpy | Any recent version |

```bash
# Verify your OpenCV has CUDA support
python -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
# Should print 1 or more

# Verify FFmpeg has NVENC
ffmpeg -hide_banner -encoders | grep nvenc

# Verify the default encode actually works on this GPU / driver / FFmpeg
python ken_burns.py --dryRun
```

---

## Quick Start

```bash
# Crossfade dissolve (default): 8s per slide, 1s crossfade, HEVC RExt 4:4:4 10-bit
python ken_burns.py --img_path ~/photos/holiday -o ~/Videos/holiday.mp4 --suffix AI_Kenburns

# Classic Ken Burns zoom + fade to black
python ken_burns.py --img_path ~/photos/holiday -o ~/Videos/holiday.mp4 --suffix AI_Kenburns \
    --ken_burns --ftb

# Ken Burns starting zoom-in, crossfade transitions
python ken_burns.py --img_path ~/photos/holiday -o ~/Videos/holiday.mp4 --suffix AI_Kenburns \
    --ken_burns --zoom_in

# Check estimated duration before rendering
python ken_burns.py --img_path ~/photos/holiday --calculate --ken_burns --ftb

# Check the encoder works before a long render
python ken_burns.py --dryRun
```

Output resolution is fixed at 1920×1080 (`TARGET_W` / `TARGET_H` in `ken_burns.py`). Source images of any size are centre-cropped to 16:9 and scaled to fit.

---

## Batch Processing

`--img_path` accepts several folders and patterns, and `--auto_filename` derives
each output name from its folder, so one command can render a whole model's sets:

```bash
python ken_burns.py --img_path '~/stills/Bobbie-Slideshow-{077-082}*' \
    --auto_filename -o ~/Videos --suffix AI_Kenburns --ken_burns --ftb
```

### Selecting folders

| Pattern | Matches |
|---|---|
| `*` / `?` | any run of characters / exactly one character |
| `[abc]`, `[!abc]` | one character from / not from the set |
| `{N-M}` | a whole number from N to M (by value: `{77-82}` matches `077`) |
| `{N-}` | a whole number from N to 999 |

Wildcards apply to the folder name (the last path component); without a
directory the current directory is searched. Quote patterns — single quotes are
safest (see the shell quoting primer in [`work_scripts/README.md`](work_scripts/README.md#shell-quoting-primer)).
The same rules apply to `extractStills.py --video`; both use `globrange.py`.

More than one folder is a **batch**, and a batch requires `--auto_filename`.
Image file names inside a folder do not matter — slides are taken in sorted
order, as before.

### Output names

With `--auto_filename`, `--output` must be an existing directory. Folders are named
`<model>-Slideshow-<set>[_SUFFIX]` followed by `_Extracted` (`extractStills.py`)
or `_extracted_frames` (`getFrames.py`). The output is
`<model>-Slideshow-<set>_<--suffix>.mp4`: `--suffix` replaces the folder's SUFFIX,
or is appended when there is none. A leading `_` on `--suffix` is optional.

| Folder | `--suffix` | Output |
|---|---|---|
| `Bobbie-Slideshow-077_AI_Extracted` | `AI_Kenburns` | `Bobbie-Slideshow-077_AI_Kenburns.mp4` |
| `Cristina-Slideshow-001_AI_extracted_frames` | `Kenburns` | `Cristina-Slideshow-001_Kenburns.mp4` |
| `Cristina-Slideshow-002_extracted_frames` | `AI_Kenburns` | `Cristina-Slideshow-002_AI_Kenburns.mp4` |
| `Bobbie-Slideshow-077_AI_Extracted` | `_AI` | error — see below |

`--suffix` is required for every render and is rejected when it:

- contains spaces (a folder whose SUFFIX contains spaces is rejected too);
- is empty or `AI` — by convention a video with no suffix, or with `_AI`, was
  **not** made by ken_burns, so ken_burns never produces one;
- equals the folder's own SUFFIX (case-insensitive) — the result would carry the
  name of the very video the images were extracted from.

With an explicit `--output path/name.mp4` (single folder, no `--auto_filename`),
the name is used exactly as given and `--suffix` is only validated. This is the
hook for wrapper scripts that choose their own file names.

### Never overwrite

An output video that already exists is always an error — single render or batch.
Each video is rendered to a hidden `.<name>.partial-<pid>.mp4` beside the target
and moved into place only when complete, so a failed or interrupted render never
leaves a partial file under the final name.

In a batch, a folder that cannot be rendered (output exists, name cannot be
derived, no images) is reported and skipped; the rest still render. The batch
ends with a summary of rendered, failed and skipped folders with their reasons.

| Exit code | Meaning |
|---|---|
| 0 | every folder rendered |
| 1 | at least one folder skipped or failed, or a command-line error |
| 130 | interrupted (Ctrl-C); the partial file is removed |

`--calculate` and `--gen_filter_map` also accept several folders and run once per folder.

---

## Motion Modes

### Crossfade / Static Hold (default)

Without `--ken_burns` or `--drift`, each slide is a static hold: the image fills the frame with no movement. Transitions are crossfade dissolves by default.

```bash
python ken_burns.py --img_path ~/imgs -o out.mp4 --suffix AI_Kenburns
```

### Drift (`--drift`)

A subtle, configurable zoom on each slide. The image slowly drifts in or out by a small crop amount. You get a gentle sense of motion without the full Ken Burns effect.

```
0.0      = disabled (pure static hold)
0.33     = ~5% crop change — visible but subtle (recommended starting point)
1.0      = ~15% crop change (equivalent to full Ken Burns geometry)
```

By default the direction alternates from slide to slide. Add `--rand_drift` to pick each slide's direction at random.

```bash
python ken_burns.py --img_path ~/imgs -o out.mp4 --suffix AI_Kenburns --drift 0.33 --rand_drift
```

### Ken Burns (`--ken_burns`)

Full zoom-in or zoom-out on each slide (~15% crop change). A streak-based algorithm sets the direction:

- 70% chance to flip direction on each new slide
- Never more than 3 consecutive slides in the same direction
- Starting direction controlled by `--zoom_in` (default: zoom-out first)

```bash
python ken_burns.py --img_path ~/imgs -o out.mp4 --suffix AI_Kenburns --ken_burns --zoom_in
```

> `--ken_burns` and `--drift` are mutually exclusive.

Motion uses cosine easing, and the crop rectangle is fractional: it is never rounded to whole source pixels. Slow zooms therefore move smoothly rather than stepping one pixel at a time.

---

## Transition Styles

Transition style is independent of motion mode, so any combination works.

| Flag | Behaviour |
|------|-----------|
| *(none)* | **Crossfade dissolve**: slides overlap during the transition window |
| `--ftb` | **Fade to black**: the current slide fades out, then the next fades in |
| `--random` | **Random**: each boundary independently picks crossfade or fade to black |

`--trans_time` sets the length of the transition (the overlap or fade window). The default is 1.0s.

```bash
# Crossfade (default), 1.5s overlap
python ken_burns.py --img_path ~/imgs -o out.mp4 --suffix AI_Kenburns --trans_time 1.5

# Fade to black, Ken Burns zoom
python ken_burns.py --img_path ~/imgs -o out.mp4 --suffix AI_Kenburns --ken_burns --ftb --trans_time 1.0

# Random transitions, subtle drift
python ken_burns.py --img_path ~/imgs -o out.mp4 --suffix AI_Kenburns --drift 0.25 --random
```

---

## Image Filters

Two optional CUDA filters can be applied to each image. They run on the GPU right after the image is scaled to 1920×1080.

### `--smooth`: CUDA Bilateral Filter

Suppresses staircase artefacts from upscalers (Real-ESRGAN, etc.) while keeping real edges. Each neighbouring pixel is weighted by both its distance and its colour difference. Flat regions are smoothed and subject outlines stay sharp.

| Preset | Diameter | Sigma Color | Sigma Space | Intensity |
|--------|----------|-------------|-------------|-----------|
| `subtle` | 3 | 15 | 15 | 40% |
| `light` | 5 | 35 | 35 | 50% |
| `medium` | 9 | 75 | 75 | 70% |
| `strong` | 13 | 120 | 120 | 85% |

**Recommended starting point for ESRGAN portraits:** `--smooth medium`

`intensity` is a blend factor: `output = (1 - intensity) × original + intensity × filtered`. It dials the effect back without changing the filter kernel. Override it with `--smooth_intensity` (0–100).

### `--sharpen`: CUDA Laplacian Edge Boost

Recovers edge sharpness softened by the bilateral pass. The frame is converted to greyscale and run through a 3×3 Laplacian. The positive part of the edge response (8-bit output clamps negatives to 0) is added back onto the BGR frame.

| Preset | boost_strength |
|--------|---------------|
| `subtle` | 0.10 |
| `light` | 0.25 |
| `medium` | 0.45 |
| `strong` | 0.65 |

### Order matters

The filters always run smooth first, then sharpen. Sharpening first would sharpen the artefacts instead of suppressing them.

```bash
# Typical ESRGAN portrait set
python ken_burns.py --img_path ~/imgs -o out.mp4 --suffix AI_Kenburns --smooth medium --sharpen light

# Heavy artefacts
python ken_burns.py --img_path ~/imgs -o out.mp4 --suffix AI_Kenburns --smooth strong --sharpen medium

# Clean source: mild crispening only
python ken_burns.py --img_path ~/imgs -o out.mp4 --suffix AI_Kenburns --sharpen light
```

> **Performance note:** bilateral filter cost scales with d². `light` (~25 ops/pixel) is fast; `strong` (~169 ops/pixel) is ~6.8× heavier. On Ken Burns and drift slides the filters run every frame. On static slides they run once, when the slide is loaded.

### When NOT to smooth

- Clean source (raw camera, lossless PNG): no benefit, and it costs GPU time
- Fine repeating texture (brickwork, fabric weave): the bilateral filter can wash out periodic detail
- Test/proof renders where you want to see the upscaler's native output

### When NOT to sharpen

- Without `--smooth` on ESRGAN output: the Laplacian amplifies the very artefacts you are trying to suppress
- Already-sharpened source: double-sharpening produces haloing and ringing
- Noisy images: above `boost_strength` 0.25 the noise dominates

Full filter guidance: `python ken_burns.py --filter_help`

---

## Filter Map System

Apply different filter settings to individual images in a set, instead of the same settings to every slide.

### Step 1: Generate the map

```bash
python ken_burns.py --img_path ~/imgs/set_042 --gen_filter_map
```

This scores every image with an edge-direction discontinuity detector: 0–100, where higher means more staircase-like artefacts. It writes `filter_map.json` to the image directory and prints the scores from lowest to highest.

```bash
# Adjust sensitivity
python ken_burns.py --img_path ~/imgs/set_042 --gen_filter_map --fmap_threshold 25

# Pre-populate flagged entries with a specific preset
python ken_burns.py --img_path ~/imgs/set_042 --gen_filter_map --smooth light
```

The default threshold is 30. Images above it get a smooth preset picked by how far they score above the threshold (subtle, light, medium or strong). Images at or below it are set to `null`.

### Step 2: Review and edit

```json
{
  "img_001.jpg": {"smooth": "light", "sharpen": "light"},
  "img_002.jpg": null,
  "img_003.jpg": {"smooth": "medium"},
  "img_004.jpg": null
}
```

- **Spec entry**: uses the map's `smooth` and/or `sharpen`; global flags are ignored for this image
- **`null` entry**: no filtering; global flags are ignored for this image
- **Not in map**: falls back to the global `--smooth` / `--sharpen`

Sharpen is never suggested automatically. Add it by hand to specific entries, or pass `--sharpen` at render time as a global fallback.

### Step 3: Render with the map

```bash
python ken_burns.py --img_path ~/imgs/set_042 -o ~/out/set_042.mp4 --suffix AI_Kenburns \
    --ken_burns --ftb --use_filter_map

# With global fallback for images not in the map
python ken_burns.py --img_path ~/imgs/set_042 -o ~/out/set_042.mp4 --suffix AI_Kenburns \
    --use_filter_map --smooth subtle
```

Full filter map guidance: `python ken_burns.py --fmap_help`

---

## Encoding

Slow, continuous sub-pixel zoom and drift make quantization stepping, banding and chroma edge shimmer much easier to see than in static slideshows. The default encode uses full chroma (4:4:4) and 10-bit precision to keep these under control.

### Default (no flags)

```
-c:v hevc_nvenc -preset p7 -tune hq -profile:v rext -pix_fmt yuv444p16le
-colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv
-rc vbr -cq 26 -b:v 0
-g 250 -bf 4 -b_ref_mode middle -rc-lookahead 32 -spatial-aq 1 -aq-strength 4
-an -movflags +faststart
```

The result is an HEVC **RExt** stream in **`yuv444p10le`**, tagged BT.709 limited range. `yuv444p16le` is the format ffmpeg hands to NVENC (a 16-bit container); NVENC encodes its top 10 bits. 4:4:4 at 10-bit requires the HEVC Range Extensions (RExt) profile, so `-profile:v rext` is set automatically.

### `--pix_fmt`

| Value | Stream | Profile | Use when |
|-------|--------|---------|----------|
| `yuv444p16le` *(default)* | 10-bit 4:4:4 | HEVC RExt | Masters, archival, local playback on NVDEC / capable decoders |
| `p010le` | 10-bit 4:2:0 | HEVC Main10 | Wide playback compatibility while keeping 10-bit gradients |
| `yuv420p` | 8-bit 4:2:0 | HEVC Main / H.264 High | Maximum compatibility (browsers, web upload, older devices). Consider a lower `--cq` |
| `auto` | | | Same as `yuv444p16le`; only falls back to `yuv420p` with `--h264` |

**Playback compatibility is usually the harder limit, not encoding.** Many browsers, TVs, phones and older hardware decoders cannot play HEVC 4:4:4. NVIDIA NVDEC on Turing and newer decodes it, and FFmpeg software decoding always works. Use `p010le` or `yuv420p` for files you distribute.

### `--h264`

Switches to `h264_nvenc`, with the pixel format falling back to `yuv420p`. Use it only when the target cannot decode HEVC. H.264 NVENC cannot encode 10-bit before Blackwell, so `--h264 --pix_fmt p010le` (or `yuv444p16le`) fails on Ampere / Ada and older. `--dryRun` reports this immediately.

### `--cq`

NVENC constant-quality target (1–51, default `26`). Rate control is VBR with no bitrate cap (`-b:v 0`). Lower values mean higher quality and larger files. Try 20–24 for `yuv420p` or `--h264` if slow motion shows stepping.

### `--dryRun`

```bash
python ken_burns.py --dryRun
python ken_burns.py --dryRun --pix_fmt p010le --cq 24
python ken_burns.py --dryRun --h264
```

Encodes a single 256×256 frame to `-f null -` using **exactly** the flags the render would use, prints PASS/FAIL with FFmpeg's error output, and exits 0 or 1. It takes about 0.3 s, mostly NVENC startup, and needs no `--img_path` or `--output`.

Because it runs the real encoder instead of matching GPU model names, it catches:

- An unsupported profile, pixel format or B-frame mode on your NVENC generation
- No free NVENC sessions (GeForce cards allow 12 at once)
- An outdated driver, or an FFmpeg build without NVENC

Run it on a new machine, after a driver or FFmpeg upgrade, or before a long batch.

The probe is 256×256 rather than 1×1 or 64×64 because NVENC rejects frames that small.

### Errors

ffmpeg runs with `-loglevel error`, so normal renders are quiet. If ffmpeg exits with an error or the pipe breaks, the last lines of its stderr are printed with the exit code and the number of frames written, and the script exits 1. A partially written output file is left in place.

Full encoding guidance: `python ken_burns.py --encoding_help`

---

## CLI Reference

`python ken_burns.py -h` shows the version and all options.

### File I/O

| Argument | Default | Description |
|----------|---------|-------------|
| `--img_path PATH [PATH ...]` | *(required)* | Source folder(s) of images (JPG, JPEG, PNG, either case); paths or patterns — see [Batch Processing](#batch-processing) |
| `-o / --output PATH` | *(required for render)* | Output path including filename, e.g. `~/Videos/out.mp4`; an existing directory with `--auto_filename`. Never overwritten |
| `--auto_filename` | off | Derive each output name from its folder name; required for batch (more than one folder) |
| `--suffix SUFFIX` | *(required for render)* | Output name suffix, e.g. `AI_Kenburns`; replaces the folder's SUFFIX. Not empty, `AI`, the folder's own SUFFIX, or containing spaces |

### Timing

| Argument | Default | Description |
|----------|---------|-------------|
| `-d / --duration SECS` | `8.0` | Per-slide display duration in seconds |
| `-f / --fps N` | `30` | Output frame rate |
| `--trans_time SECS` | `1.0` | Transition duration: crossfade overlap, or the fade-to-black window on each side |

### Ken Burns

| Argument | Default | Description |
|----------|---------|-------------|
| `--ken_burns` | off | Full Ken Burns zoom (~15% crop change, streak-based direction). Mutually exclusive with `--drift`. |
| `-z / --zoom_in` | off | Start first slide zooming in (Ken Burns mode only; default is zoom-out first) |

### Transitions

| Argument | Default | Description |
|----------|---------|-------------|
| `--ftb` | off | Fade to black at each slide boundary. Mutually exclusive with `--random`. |
| `--random` | off | Randomly assign crossfade or fade to black per boundary. Mutually exclusive with `--ftb`. |
| `--drift AMOUNT` | `0.0` | Subtle zoom drift per slide (0.0 = disabled, range 0.01–1.0). Mutually exclusive with `--ken_burns`. |
| `--rand_drift` | off | Random zoom direction per slide when using `--drift` (default: alternating) |

### Image Processing

| Argument | Default | Description |
|----------|---------|-------------|
| `--smooth PRESET` | off | CUDA bilateral filter: `subtle` `light` `medium` `strong` |
| `--sharpen PRESET` | off | CUDA Laplacian edge boost: `subtle` `light` `medium` `strong` |
| `--smooth_intensity 0-100` | preset value | Override bilateral blend intensity without changing the filter kernel |
| `--use_filter_map` | off | Load `filter_map.json` from `--img_path` for per-image filter settings |

### Encoding

| Argument | Default | Description |
|----------|---------|-------------|
| `--h264` | off (HEVC) | Use `h264_nvenc` instead of `hevc_nvenc`; `--pix_fmt auto` becomes `yuv420p` |
| `--pix_fmt FMT` | `yuv444p16le` (via `auto`) | `yuv444p16le` (HEVC RExt 10-bit 4:4:4), `p010le` (Main10), `yuv420p`, `auto` |
| `--cq 1-51` | `26` | NVENC constant quality (VBR, `-b:v 0`) |
| `--dryRun` | | Probe the resolved encoder settings with one frame, print PASS/FAIL, exit |
| `--encoding_help` | | Detailed guidance on `--h264`, `--pix_fmt`, `--cq`, `--dryRun` |

### Utility

| Argument | Description |
|----------|-------------|
| `--calculate` | Print estimated video duration and settings without rendering |
| `--filter_help` | Detailed guidance on `--smooth`, `--sharpen` and `--smooth_intensity` |
| `--fmap_help` | Detailed guidance on the filter map workflow |
| `--gen_filter_map` | Analyse images and write `filter_map.json` to `--img_path` |
| `--fmap_threshold 0-100` | Score threshold for `--gen_filter_map` (default: 30.0) |

Bash completion for all options: `source ken_burns-bash-completion.sh`

---

## Examples

### Classic documentary slideshow

```bash
python ken_burns.py \
    --img_path ~/photos/project \
    -o ~/Videos/documentary.mp4 --suffix AI_Kenburns \
    --ken_burns --ftb --zoom_in \
    --duration 10 --trans_time 1.5
```

### Crossfade with subtle drift and ESRGAN smoothing

```bash
python ken_burns.py \
    --img_path ~/photos/esrgan_output \
    -o ~/Videos/smooth.mp4 --suffix AI_Kenburns \
    --drift 0.33 --rand_drift \
    --smooth medium --sharpen light \
    --duration 9
```

### Ken Burns + per-image filters

```bash
# 1. Generate filter map
python ken_burns.py --img_path ~/photos/set --gen_filter_map --smooth light

# 2. Edit filter_map.json as needed, then render
python ken_burns.py \
    --img_path ~/photos/set \
    -o ~/Videos/set.mp4 --suffix AI_Kenburns \
    --ken_burns --ftb \
    --use_filter_map \
    --smooth subtle        # fallback for images not in the map
```

### Check duration before rendering

```bash
python ken_burns.py --img_path ~/photos/holiday --calculate \
    --ken_burns --ftb --duration 8 --trans_time 1
```

Output:
```
  Ken Burns  ─  Duration Calculator
  ──────────────────────────────────────────────────────────
  Source Path      : /home/user/photos/holiday
  Images Found     : 47
  ──────────────────────────────────────────────────────────
  Duration/Slide   : 8.0s
  Trans Time       : 1.0s
  FPS              : 30
  Transition Mode  : Fade-to-black
  Zoom             : Ken Burns  (zoom-out first, streak-based alternation)
  ──────────────────────────────────────────────────────────
  Visible Hold     : 6.0s per slide  (duration − 2 × trans_time)
  Total Duration   : 06:16  (376s  |  47 slides)
```

### Distribution-friendly output

```bash
# 10-bit 4:2:0 HEVC: plays almost everywhere, keeps smooth gradients
python ken_burns.py --img_path ~/photos/set -o ~/Videos/out_main10.mp4 --suffix AI_Kenburns \
    --ken_burns --pix_fmt p010le

# H.264 8-bit 4:2:0: maximum compatibility
python ken_burns.py --img_path ~/photos/set -o ~/Videos/out_h264.mp4 --suffix AI_Kenburns \
    --ken_burns --ftb --h264 --cq 22
```

---

## Under the Hood — Pipeline

The CPU loads images and computes scalar parameters (crop rectangles, blend weights). Resizing, warping, filtering and blending run as CUDA kernels on the GPU. Finished frames are copied back to host memory and piped to FFmpeg, which converts the colour format and feeds NVENC.

### 1. Image load (CPU → GPU, once per slide)

```python
img = cv2.imread(path)            # decode on CPU
gpu.upload(...)                   # one host → device copy per slide
```

- **Static slides:** the 16:9 centre crop is cut on the CPU and uploaded, then scaled to 1920×1080 with `cv2.cuda.resize` (`INTER_CUBIC`).
- **Ken Burns and drift slides:** the full image is uploaded once.

At most two slides are cached at a time, so a crossfade can access both the outgoing and the incoming slide.

### 2. Per-frame motion (CUDA affine warp, sub-pixel)

For Ken Burns and drift slides, every frame is a single `cv2.cuda.warpAffine` from the full uploaded image:

```python
s = target_w / cw_f                              # fractional crop width
M = [[s, 0, target_w/2 - 0.5 + s*(0.5 - cx_f)],
     [0, s, target_h/2 - 0.5 + s*(0.5 - cy_f)]]
gpu_out = cv2.cuda.warpAffine(gpu_img, M, (1920, 1080),
                              flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
```

The crop rectangle (`cw_f`, `cx_f`, `cy_f`) is interpolated with cosine easing and **never rounded**. Rounding would snap the motion to the source pixel grid, which shows up as visible stutter when there is little downscale headroom. The ±0.5 terms match `cv2.resize`'s pixel-centre convention, so framing lines up exactly with the static-slide path.

There is no per-frame upload: every frame is derived on the GPU from the image uploaded once in step 1.

### 3. Bilateral filter (optional, GPU)

```python
gpu_filtered = cv2.cuda.bilateralFilter(gpu_img, d, sigmaColor, sigmaSpace)
gpu_blended  = cv2.cuda.addWeighted(gpu_img, 1 - intensity, gpu_filtered, intensity, 0)
```

### 4. Laplacian sharpening (optional, GPU)

```python
gpu_gray    = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
lap_filter  = cv2.cuda.createLaplacianFilter(cv2.CV_8UC1, cv2.CV_8UC1, 3)
gpu_lap     = lap_filter.apply(gpu_gray)                     # CV_8U clamps negatives to 0
gpu_lap_bgr = cv2.cuda.cvtColor(gpu_lap, cv2.COLOR_GRAY2BGR)
result      = cv2.cuda.addWeighted(gpu_img, 1.0, gpu_lap_bgr, strength, 0.0)
```

On static slides, steps 3–4 run once at load time. On Ken Burns and drift slides they run on every frame, after the warp.

### 5. Transition blending (GPU)

```python
# Crossfade: blend current slide with next slide
blended = cv2.cuda.addWeighted(gpu_current, alpha_a, gpu_next, alpha_b, 0.0)

# Fade to black / opening / closing: blend with a black GpuMat
gpu_black = cv2.cuda.GpuMat(); gpu_black.upload(np.zeros((1080, 1920, 3), np.uint8))  # once
blended   = cv2.cuda.addWeighted(gpu_frame, alpha, gpu_black, 1 - alpha, 0.0)
```

The black frame is uploaded once and reused for the entire render. Ease-in / ease-out curves are evaluated in Python as a single float per frame.

### 6. Frame output (GPU → CPU → FFmpeg pipe)

```python
proc.stdin.write(gpu_frame.download())   # 1920×1080 bgr24, ~6.2 MB
```

- **Moving or transitioning frames:** every frame is downloaded to host memory, then written to FFmpeg's stdin.
- **Static holds:** a host-side copy made at load time is reused, so these frames skip the download.

### 7. FFmpeg conversion + NVENC encode

FFmpeg reads raw `bgr24` from the pipe. Its software scaler (`swscale`, on the CPU) converts each frame to the encoder input format, `yuv444p16le` by default, using the BT.709 matrix. The NVENC fixed-function encoder then compresses it (see [Encoding](#encoding)). NVENC runs on its own engine, separate from the CUDA cores, so encoding doesn't take compute away from the warp and filter kernels.

The input pipe is also tagged with BT.709 primaries and transfer. With FFmpeg 9, output-side `-color_primaries` / `-color_trc` alone are dropped, and the stream would otherwise be tagged `unknown`.

### Pipeline summary

```
Disk ─ imread (CPU) ─ upload ─► VRAM: warp / resize → smooth → sharpen → blend
                                                                        │
              FFmpeg stdin ◄─ host RAM ◄─ download (per frame) ◄────────┘
                    │
   swscale bgr24 → yuv444p16le (CPU) ─► NVENC (GPU) ─► HEVC RExt yuv444p10le MP4
```

What crosses PCIe:

- **Upload:** one image per slide
- **Download:** one 1920×1080 frame per output frame (except static holds)
- **FFmpeg → NVENC:** the converted frame, uploaded by the encoder

All pixel processing in between runs as CUDA kernels in VRAM.

---

## Hardware Notes

### NVENC generations and the default encode

NVENC encode features depend on the **NVENC generation**, not on GeForce vs. professional branding. For example, the GeForce RTX 3060 and the RTX A4500 (both Ampere, 7th-gen NVENC) have identical codec features. The main branding difference is concurrent sessions: 12 on GeForce, unrestricted on most pro cards.

| NVENC generation | Example GPUs | Default (HEVC RExt 4:4:4 10-bit, B-frames) | `p010le` / `yuv420p` HEVC | H.264 10-bit |
|------------------|--------------|-------------------------------------------|---------------------------|--------------|
| 9th (Blackwell) | RTX 50, RTX PRO Blackwell | ✔ | ✔ | ✔ (Blackwell adds 4:2:2 / high bit depth H.264) |
| 8th (Ada) | RTX 40, RTX Ada | ✔ | ✔ | ✘ |
| 7th (Ampere / Turing) | RTX 30, RTX A-series, RTX 20, GTX 1660 | ✔ | ✔ | ✘ |
| 6th (Pascal, GTX 1650/1630) | GTX 10-series, GTX 1650 | ✘: no HEVC B-frames (`-bf 4` / `-b_ref_mode` fail) | ✘ with default B-frame flags | ✘ |
| 5th (Maxwell 2nd gen) | GTX 950/960/970/980 | ✘: no HEVC 4:4:4 or 10-bit | 8-bit 4:2:0 only, no B-frames | ✘ |

Sources: NVIDIA Video Encode and Decode GPU Support Matrix, and testing on an RTX A4500. Not every row has been tested; driver and FFmpeg versions matter too. Treat `--dryRun` as the authority for your machine.

The encoder flags (B-frames, `-b_ref_mode middle`) are the same for every `--pix_fmt`. So on 6th-gen and older NVENC, the 4:2:0 options fail as well. The script has no switch to turn B-frames off.

### Throughput

Rendering is usually limited by CUDA warp/filter work and the per-frame download + CPU colour conversion, not by NVENC. Measured reference: RTX A4500 with 4K PNG sources and Ken Burns zoom, no filters, renders ~117 fps end to end. Heavy `--smooth` presets on Ken Burns or drift slides reduce this, because the bilateral filter runs on every frame.

### VRAM

- **4 GB:** enough for 1080p output with all features and typical photo sizes
- **8 GB+:** recommended for very large sources (24 MP+), since Ken Burns and drift slides keep the full-resolution image, and up to two slides, in VRAM

### Checking your GPU

```bash
nvidia-smi                       # driver, CUDA version, VRAM
python ken_burns.py --dryRun     # can this GPU/driver/FFmpeg encode the default?

python -c "import cv2; cv2.cuda.printCudaDeviceInfo(cv2.cuda.getDevice())"
```

---

## File Structure

```
ken_burns/
├── ken_burns.py                  # Main script: render pipeline, encoding, help pages
├── cliOpts.py                    # CLI argument parser
├── imgbatch.py                   # --img_path folders, --auto_filename / --suffix naming rules
├── globrange.py                  # Path patterns (* ? [...] {N-M} {N-}), shared with extractStills.py
├── Bcolors.py                    # ANSI terminal colour utility
├── version.py                    # __version__
├── batch_convert_gbrp.py         # Batch re-encoder for legacy gbrp slideshows (reference NVENC settings)
├── ken_burns-bash-completion.sh  # Bash completion
└── README.md
```

Images can be JPG, JPEG or PNG (either case). Slides are ordered by full file path, sorted alphabetically. Name files numerically (`001.jpg`, `002.jpg`, …) to control the sequence.

---

## License

GNU Lesser General Public License v3.0 — see [LICENSE](LICENSE).
