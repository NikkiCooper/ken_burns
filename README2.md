# Ken Burns — GPU‑Accelerated Slideshow Renderer  
*A CUDA‑powered motion and transition engine with optional per‑image filtering.*

---

## Overview

This project implements a GPU‑accelerated slideshow renderer using OpenCV’s CUDA modules and FFmpeg’s NVENC encoder. All image processing — crop, resize, drift/Ken Burns motion, bilateral smoothing, Laplacian sharpening, and transition blending — is performed on the GPU. Frames are streamed directly to FFmpeg for hardware encoding.

The design goal is simple:

**Keep pixel operations on the GPU, minimize PCIe transfers, and produce clean, cinematic motion.**

This README is written for future reference. It explains the pipeline, the motion modes, the filter system, and the CUDA internals in detail.

---

## Features

- Ken Burns zoom with streak‑based direction alternation  
- Subtle drift motion with alternating or random direction  
- Crossfade dissolve transitions  
- Fade‑to‑black transitions  
- Random transition mode  
- CUDA bilateral smoothing  
- CUDA Laplacian sharpening  
- Per‑image filter map system  
- HEVC or H.264 NVENC encoding  
- GPU‑native blending and filtering  
- Minimal PCIe transfers  
- Deterministic motion and timing  

---

## Requirements

- Python 3.8+  
- OpenCV built with CUDA (`cv2.cuda` must exist)  
- NVIDIA GPU with NVENC  
- CUDA Toolkit 11+  
- FFmpeg with NVENC support  
- numpy  

---

## Quick Start

```bash
python ken_burns.py --img_path ~/photos --output out.mp4
```

Ken Burns mode:

```bash
python ken_burns.py --img_path ~/photos --output out.mp4 --ken_burns
```

Fade‑to‑black:

```bash
python ken_burns.py --img_path ~/photos --output out.mp4 --ftb
```

Duration calculator:

```bash
python ken_burns.py --img_path ~/photos --calculate
```

---

## Motion Modes

### Static Hold (default)

No zoom. No drift. Crossfade transitions unless overridden.

### Drift (`--drift`)

A subtle zoom applied per slide.  
Crop change amount:

- `0.0` → disabled  
- `0.33` → ~5% crop  
- `1.0` → ~15% crop  

Direction alternates unless `--rand_drift` is used.

### Ken Burns (`--ken_burns`)

Full zoom‑in/out motion (~15% crop).  
Direction alternates using a streak‑based algorithm:

- 70% chance to flip  
- Max streak of 3  
- Starting direction controlled by `--zoom_in`

Drift and Ken Burns are mutually exclusive.

---

## Transition Styles

- **Crossfade** (default)  
- **Fade‑to‑black** (`--ftb`)  
- **Random** (`--random`)  

Transition duration is controlled by `--trans_time`.

---

## Image Filters

Two optional CUDA filters can be applied:

### Bilateral Smoothing (`--smooth`)

Reduces staircase artifacts from upscalers.  
Preset controls kernel size and sigma values.  
Blend intensity can be overridden with `--smooth_intensity`.

### Laplacian Sharpening (`--sharpen`)

Restores edge definition after smoothing.  
Preset controls boost strength.

### Filter Order

Always smooth → then sharpen.

---

## Filter Map System

A JSON file (`filter_map.json`) can specify per‑image filter settings.

Workflow:

1. Generate map  
2. Edit entries  
3. Render with `--use_filter_map`

Images above the threshold get auto‑assigned smoothing presets.  
Sharpening is manual.

---

## CLI Reference

### Required

- `--img_path PATH`  
- `--output PATH` (unless using `--calculate` or `--gen_filter_map`)  

### Timing

- `--duration SECS`  
- `--fps N`  
- `--trans_time SECS`  

### Motion

- `--ken_burns`  
- `--zoom_in`  
- `--drift AMOUNT`  
- `--rand_drift`  

### Transitions

- `--ftb`  
- `--random`  

### Filters

- `--smooth PRESET`  
- `--sharpen PRESET`  
- `--smooth_intensity PERCENT`  
- `--use_filter_map`  

### Encoding

- `--h264` (otherwise HEVC)  

### Utilities

- `--calculate`  
- `--gen_filter_map`  
- `--fmap_threshold`  
- `--filter_help`  
- `--fmap_help`  

---

## Examples

Ken Burns + fade‑to‑black:

```bash
python ken_burns.py --img_path ~/photos --output out.mp4 --ken_burns --ftb
```

Drift + smoothing:

```bash
python ken_burns.py --img_path ~/photos --output out.mp4 --drift 0.33 --smooth medium
```

Filter map workflow:

```bash
python ken_burns.py --img_path ~/set --gen_filter_map
python ken_burns.py --img_path ~/set --output out.mp4 --use_filter_map
```

---

# Under the Hood — CUDA Pipeline (Deep Technical Notes)

This section explains the internal pipeline in detail.  
It is intentionally long and explicit — useful when revisiting the project later.

---

## 1. Image Loading (CPU → GPU)

Images are decoded with `cv2.imread()` into CPU RAM.  
Each slide is uploaded once:

```
GpuMat.upload()
```

This is the **first PCIe transfer**.

Static slides upload once.  
Ken Burns/drift slides upload once per slide, but require per‑frame cropping.

---

## 2. Crop and Resize (GPU)

### Static Slides

Crop + resize happens **once** at load time.  
The result is cached in a GpuMat.

### Ken Burns / Drift Slides

Each frame has a different crop window.  
OpenCV CUDA uses **zero‑copy ROI**:

```
roi = gpu_full.rowRange(...).colRange(...)
```

This is a view, not a copy.

Resize is a CUDA kernel:

```
cv2.cuda.resize(roi, ...)
```

No CPU involvement.

---

## 3. Bilateral Filter (GPU)

Runs either:

- once per slide (static), or  
- once per frame (Ken Burns/drift)

Blend is also GPU‑native.

Cost scales with `d²`.

---

## 4. Laplacian Sharpening (GPU)

Sequence:

- Convert to grayscale  
- Laplacian filter  
- abs()  
- convert back to BGR  
- blend with original  

All operations are CUDA kernels.

---

## 5. Transition Blending (GPU)

Crossfade:

```
addWeighted(current, next)
```

Fade‑to‑black:

```
addWeighted(frame, black)
```

Black GpuMat is allocated once.

---

## 6. Frame Download (GPU → CPU)

Final frame is downloaded:

```
GpuMat.download()
```

This is the **second PCIe transfer**.

Frames are streamed directly into FFmpeg’s stdin pipe.

No NumPy processing.

---

## 7. NVENC Encoding

FFmpeg receives raw BGR24 frames and passes them to NVENC.

Preset: `p4`  
Quality: `cq 22`

Encoding is fully hardware‑accelerated.

---

## Pipeline Summary

```
Disk → CPU → GPU → [crop → resize → filter → sharpen → blend] → CPU → FFmpeg → NVENC → MP4
```

Only two PCIe transfers:

- Upload at slide load  
- Download per frame  

Everything else is GPU‑resident.

---

## Hardware Notes

- 4GB VRAM is enough for 1080p  
- 8GB+ recommended for large source images  
- RTX cards handle strong bilateral filters easily  
- GTX 10‑series may struggle with strong smoothing in Ken Burns mode  

---

## File Structure

```
ken_burns.py
cliOpts.py
Bcolors.py
README.md
```

---

## License

LGPL‑3.0
