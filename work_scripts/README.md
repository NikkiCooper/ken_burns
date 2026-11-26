# extractStills

Extracts one clean, transition-free still per slide from Imagination-rendered
HEVC slideshow videos (`.mp4`), using each video's `.img` project file to work
out exactly where every slide's still begins and ends.

Output stills (3840×2160 by default) feed `ken_burns`. Timing is derived from
the project rather than guessed, every plan is checked for consistency before
anything is written, and every saved frame is verified to be a true still.

---

## Contents

- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Selecting videos](#selecting-videos)
  - [Excluding sets with `[!abc]`](#excluding-sets-with-abc)
  - [Shell quoting primer](#shell-quoting-primer)
- [Output layout and naming](#output-layout-and-naming)
- [Verbosity](#verbosity)
- [Logging](#logging)
- [Exit codes](#exit-codes)
- [Batch processing](#batch-processing)
- [Options](#options)
- [How it works](#how-it-works)
- [Confidence checks](#confidence-checks)
- [Bash completion](#bash-completion)
- [Troubleshooting](#troubleshooting)

---

## Requirements

| Component | Notes |
|---|---|
| Python ≥ 3.10 | system interpreter (`/usr/bin/python3`) |
| `ffmpeg` / `ffprobe` | with CUDA hwaccel (`ffmpeg -hwaccels` lists `cuda`) |
| OpenCV with CUDA | `cv2.cuda` is used for the 2160p upscale |
| NumPy, Pillow | Pillow reads source-image sizes (header only) |
| `../globrange.py` | pattern matching shared with `ken_burns.py --img_path`; must stay one directory above this script (symlinking the script is fine) |
| NVIDIA GPU | NVDEC decode; `--cpu` runs without one |

---

## Quick start

```bash
# 1. Check a set: plan, timing math and confidence - nothing is written
extractStills.py --dryRun --project-dir projects --video Cristina-Slideshow-080_AI.mp4

# 2. Extract it
extractStills.py --project-dir projects --output_base stills --video Cristina-Slideshow-080_AI.mp4

# 3. Sweep every set of a model quietly, keeping full detail in a log
extractStills.py --dryRun -q --logfile --project-dir projects \
                 --video 'Nikki Cooper-Slideshow-{001-025}*'
```

---

## Selecting videos

`--video` takes one or more files or patterns. A name without a directory is
looked up in the current directory. Patterns match against the file name only;
only `.mp4` files are matched.

| Pattern | Matches |
|---|---|
| `*` | any run of characters |
| `?` | exactly one character |
| `[abc]`, `[!abc]` | one character from / not from the set |
| `{N-M}` | a whole number from N to M — `{001-025}`, `{7-15}` |
| `{N-}` | a whole number from N to 999 |

Numeric ranges compare **values**, so `{7-15}` matches `007`, `012` and `015`.
A range always covers a complete digit run: `0{07-15}` does not match `012`
(write `{7-15}` or `{007-015}`).

```bash
extractStills.py ... --video 'Nikki Cooper-Slideshow-{001-025}*'  ~/Videos/Hanna\ F-Slideshow-???_AI.mp4
```

**Quote patterns.** Bash expands `*`, `?` and `[...]` itself when they match,
which is harmless, but with `failglob` or `nullglob` set a non-matching pattern
would be dropped or rejected before the tool sees it. Bash has no `{N-M}` syntax
(its own is `{N..M}`), so ranges always reach the tool — quoting just keeps the
behaviour predictable. Duplicate matches are processed once.

### Excluding sets with `[!abc]`

`[!abc]` matches exactly one character that is **not** in the brackets
(`[^abc]` is equivalent). A dash inside the brackets denotes a range.

```bash
# Skip the 010–019 sets: the middle digit must not be 1
extractStills.py -q --dryRun --project-dir proj --video 'Nikki Cooper-Slideshow-0[!1]?_AI.mp4'

# Middle digit not 0 or 1: only sets 020–099
extractStills.py -q --dryRun --project-dir proj --video 'Nikki Cooper-Slideshow-0[!0-1]?_AI.mp4'

# Skip every set ending in 5
extractStills.py -q --dryRun --project-dir proj --video 'Nikki Cooper-Slideshow-0?[!5]_AI.mp4'

# Every set whose suffix does not start with A (excludes the _AI videos)
extractStills.py -q --dryRun --project-dir proj --video 'Cristina-Slideshow-???_[!A]*.mp4'
```

### Shell quoting primer

Double quotes **do** stop globbing — inside `"…"` the shell never expands `*`,
`?` or `[...]`. What double quotes do *not* stop:

| Inside `"…"` | Example |
|---|---|
| `!` history expansion (interactive shells) | `"0[!1]?"` → `bash: !1: event not found` |
| `$` variable expansion | `"$HOME/x"` → `/home/nikki/x` |
| `` `…` `` / `$(…)` command substitution | `"$(date)"` |
| `\` before `$` `` ` `` `"` `\` | `"\$"` → `$` |

Single quotes stop **all** of it: everything between `'…'` is passed through
literally. That makes them the safe default for patterns — and `!`, the one
character in `[!abc]` that double quotes leave exposed, is exactly why.

Both of these work, because neither exposes a `!` to the shell:

```bash
--video 'Nikki Cooper-Slideshow-0[!1]?_AI.mp4'      # single quotes
--video "Nikki Cooper-Slideshow-0[^1]?_AI.mp4"      # double quotes, [^…] has no !
```

When a variable is part of the path, combine the two — adjacent quoted pieces
form one argument, so the variable expands and the pattern stays literal:

```bash
--video "$VIDEOS"/'Nikki Cooper-Slideshow-0[!1]?_AI.mp4'
```

---

## Output layout and naming

For `Cristina-Slideshow-080_AI.mp4`:

| Part | Value |
|---|---|
| model name | `Cristina` (may contain spaces, e.g. `Hanna F`) |
| set number | `080` |
| suffix | `_AI` |
| `<video_basename>` | `Cristina-Slideshow-080_AI` — also the project name: `Cristina-Slideshow-080_AI.img` |

```
<output_base>/Cristina-Slideshow-080_AI_Extracted/Cristina-080-001.jpg
                                                  Cristina-080-002.jpg
                                                  ...
```

- Images are named `<model>-<set>-<NNN>.<ext>` — no `-Slideshow-`, no suffix.
- `--suffix NAME` replaces the suffix in the folder name only
  (`--suffix ''` → `Cristina-Slideshow-080_Extracted/`).
- `.jpg` (quality 95) by default, `.png` with `--png`.
- 3840×2160 by default, 1920×1080 with `--resolution 1080p`.
- Frames are written to a hidden staging folder and moved into place only when
  the whole set has succeeded; a failed or interrupted run leaves nothing behind.
- Existing images are never replaced without `--overwrite` (exit code 5).

---

## Verbosity

| Level | Flag | Console output |
|---|---|---|
| 0 | `-qq` | nothing — the exit code is the result (only command-line and log-file errors are printed) |
| 1 | `-q` | one summary line per video, plus the batch summary |
| 2 | *(default)* | per video: header, consistency checks, confidence, verdict; a progress counter while extracting |
| 3 | `-v` | everything: per-slide plan table, per-file progress, tracebacks |

`-q` and `-v` cannot be combined.

Summary line (`-q`) — fixed columns, status first, so it sorts and greps cleanly:

```
PASS   Nikki Cooper-Slideshow-001_AI  conf=100  slides=55/55    delta=+0      would save 55
WARN   Nikki Cooper-Slideshow-004_AI  conf=90   slides=40/40    delta=+0      would save 40  warn: resolution (project vs video)
FAIL   Nikki Cooper-Slideshow-012_AI  conf=0    slides=80/80    delta=-9740   blocked: total frame count
ERROR  Nikki Cooper-Slideshow-015_AI                                          project not found: Nikki Cooper-Slideshow-015_AI.img
```

| Status | Meaning |
|---|---|
| `PASS` | saved / would save, no warnings |
| `WARN` | saved / would save, but at least one check warned (still exit 0) |
| `FAIL` | the set itself is the problem (exit 3, 4 or 5) |
| `ERROR` | the run could not complete (exit 1 or 2) |

Colour is used on terminals only (`--color auto`); `NO_COLOR` / `FORCE_COLOR`
are honoured; `--color always|never` overrides both.

`--help` and `--help-all` follow the same colour setting. In help text, option
names are bold cyan, placeholders bold yellow, values (choices, defaults,
patterns, exit codes) green, and labels such as `Choices:` / `Default:` magenta.
Every entry ends with its `Choices:` and `Default:` lines where they apply.

---

## Logging

`--logfile` appends to `~/extractStills.log`; `--logfile PATH` appends to `PATH`.

**The log always receives full `-v` detail, whatever the console verbosity.**
`-q`, `-qq` and `-v` only control the console. This gives a quiet console for
batch runs with nothing lost:

| Use | Command |
|---|---|
| unattended / cron | `-qq --logfile` — no output, exit code only, everything in the log |
| batch sweep | `-q --logfile` — one line per set, investigate failures in the log |
| interactive | default or `-v`, log optional |

Each run in the log starts with a separator, timestamp, working directory and
the exact command line, and ends with its exit code and elapsed time. Each
video is timestamped. The log is plain text (no colour codes).

---

## Exit codes

| Code | Name | Meaning |
|---|---|---|
| 0 | success | saved, or (`--dryRun`) would save — warnings do not change this |
| 1 | error | runtime/input error: ffmpeg/NVDEC failure, video or project not found, unreadable or malformed project, write failure, unexpected exception |
| 2 | usage | bad command line, including malformed patterns (e.g. `{9-3}`) |
| 3 | blocked | confidence gate: the timing math disagrees with the video, or the score is below `--min-confidence`; nothing decoded or written |
| 4 | not still | a target frame failed the stillness check; nothing written |
| 5 | exists | the output folder already holds target files (use `--overwrite`) |
| 130 | interrupted | Ctrl-C; any partial set is cleaned up |

**Batches** return the most serious code among their videos:
`1 > 2 > 4 > 3 > 5 > 0`. A batch exits 0 only when every video succeeded.

`--dryRun` predicts the live outcome: it returns 3 for sets that would be
blocked and, when `--output_base` is given, 5 for sets whose output already
exists. Add `--verify` to also run the stillness check (exit 4) — this decodes
the frames, ~15 s per set instead of ~0.5 s.

---

## Batch processing

Multi-video runs end with a summary:

```
=== Batch summary: 25 of 25 videos in 0m14s ===
  PASS    22   (1 with warnings)
  FAIL     2   blocked by confidence gate (exit 3)
  ERROR    1   input / runtime errors (exit 1 or 2)

  Needs attention:
    FAIL   Nikki Cooper-Slideshow-012_AI  exit 3  blocked: total frame count
    FAIL   Nikki Cooper-Slideshow-019_AI  exit 3  blocked: project slide sections
    ERROR  Nikki Cooper-Slideshow-015_AI  exit 1  project not found: Nikki Cooper-Slideshow-015_AI.img

  Hint: re-run the affected sets with -v for full diagnostics; add --logfile to keep full diagnostics while the console stays quiet.

  Exit code: 1
```

A bash wrapper that sorts sets by outcome:

```bash
#!/usr/bin/env bash
proj=~/slideshows/projects
for v in ~/Videos/"Nikki Cooper-Slideshow-"*.mp4; do
    extractStills.py -qq --dryRun --logfile --project-dir "$proj" --video "$v"
    case $? in
        0)       ready+=("$v") ;;
        3|4|5)   failing+=("$v") ;;          # the set needs attention
        130)     exit 130 ;;
        *)       broken+=("$v") ;;           # environment / input problem
    esac
done
printf 'ready:   %d\nfailing: %d\nbroken:  %d\n' ${#ready[@]} ${#failing[@]} ${#broken[@]}
printf '  %s\n' "${failing[@]}"
```

Or let the tool do the batching: `extractStills.py -q --dryRun ... --video ~/Videos/*.mp4`.

---

## Options

| Option | Default | Description |
|---|---|---|
| `--video VIDEO...` | *required* | files or patterns, see [Selecting videos](#selecting-videos) |
| `--project-dir DIR` | | searched recursively for `<video_basename>.img`; required in auto mode |
| `--output_base DIR` | | output root; required unless `--dryRun` |
| `--resolution` | `2160p` | `2160p` (3840×2160) or `1080p` (1920×1080) |
| `--png` | | lossless PNG instead of JPEG q95 |
| `--suffix SUFFIX` | from name | replaces the suffix in the output folder name; may be `''` |
| `--overwrite` | | allow replacing existing images |
| `--mode` | `auto` | `auto`: timing from the project; `manual`: fixed timing from `-t`/`-s`/`--drift` |
| `-t, --transition SEC` | | manual: time from slide start to the start of its still (the transition length) |
| `-s, --still SEC` | | manual: still duration |
| `--drift SEC` | `0` | manual: per-slide drift correction |
| `--frame-pick` | `keyframe` | `keyframe`: in-window I-frame, else centre; `center`: always centre |
| `--margin SEC` | `0.5` | minimum distance between a target and any transition |
| `--cpu` | | CPU decode/resize instead of NVDEC + `cv2.cuda` |
| `--dryRun` | | plan and check only, write nothing |
| `--verify` | | with `--dryRun`: also decode and run the stillness check |
| `--min-confidence N` | `90` | minimum score required to save |
| `-q`, `-qq`, `-v` | | console verbosity, see [Verbosity](#verbosity) |
| `--logfile [FILE]` | `~/extractStills.log` | append full detail to a log |
| `--color` | `auto` | `auto`, `always`, `never` |
| `-h, --help` | | help |
| `--help-all` | | help plus video patterns, verbosity levels, exit codes and examples |
| `--version` | | version |

---

## How it works

### Timing model

Verified to the frame against the sample sets (`Cristina-Slideshow-034_AI`,
`-080_AI`). Every slide starts with an **incoming** transition of `speed`
seconds (Fast 1, Normal 4, Slow 8 — slide 1 fades in from black), followed by
`duration` seconds of still. There is no outro.

```
still window of slide k = [ sum of earlier blocks + speed_k ,  sum of blocks 1..k )
total frames            = floor( Σ(speed + duration) × fps )      fps "30 (NTSC)" = 30000/1001

034:  55 × (4 + 9) = 715 s   → floor(715  × 29.97003) = 21428 frames   (video: 21428)
080:  80 × (4 + 9) = 1040 s  → floor(1040 × 29.97003) = 31168 frames   (video: 31168)
```

`manual` mode uses fixed timing — first target `transition + still/2`, then
every `transition + still + drift` seconds — and, if the project is found,
cross-checks the slide count and that every target lies in a still.

### Frame choice

The x265 encoder refines a still gradually after each transition and re-codes
it in full at the periodic I-frame (every 250 frames). The default
`--frame-pick keyframe` uses the I-frame inside each still window (at least
`--margin` from either transition, closest to the centre), falling back to the
centre. `--frame-pick center` always takes the window centre.

### Decode and output

One NVDEC pass per video; only the selected frames leave the GPU. YUV → RGB uses
the video's colour matrix (untagged → BT.601, TV range) with accurate chroma
interpolation. Upscaling to 2160p uses `cv2.cuda` bicubic.

### Stillness check

Every saved frame is compared with probe frames about `--margin` away inside the
same GOP and still window. A settled still measures 0.00 mean absolute
difference there; transitions measure 0.8–16 at that distance. Anything above
0.5 stops the set (exit 4).

Pillarboxed (portrait) frames are counted and cross-checked against the source
image orientation when the source file exists; slides with `angle=1` are
rotated 90° in the project and play as landscape.

---

## Confidence checks

The pre-decode assessment scores each set out of 100. Saving requires no `FAIL`
and a score ≥ `--min-confidence`.

| Check | FAIL when | WARN (penalty) when |
|---|---|---|
| project file | missing in auto mode (exit 1) | missing in manual mode (−25) |
| slide sections | `[slide N]` count ≠ `number of slides`, or gaps | |
| fps | project fps ≠ video fps | project fps unparseable (−10) |
| resolution | | project ≠ video (−10) |
| total frame count | mismatch too large to be rounding | mismatch could reach a target (−25) |
| transition speeds | | value other than 1/4/8 (−10) |
| `transition_id=0` | | present — assumed 0 s, unverified (−10) |
| pan/zoom (`no_points>1`) | | present — still window not static (−15) |
| safe window | target beyond the end of the video | still shorter than 2 × margin (−20) |
| constant frame rate | | VFR (−15) |

**Total frame count** is judged by effect, not size: a mismatch of N frames can
move a target by at most N frames, so it passes with no penalty while every
target keeps more than N frames of clearance from its transitions. The delta is
colour-coded: green (0/±1), cyan (non-zero, no effect), yellow (warn), red (fail).

---

## Bash completion

`extractStills-completion.bash` completes options, option values, directories,
and `.mp4` files for `--video` (including names with spaces).

```bash
# current shell
source /path/to/work_scripts/extractStills-completion.bash

# permanently, lazy-loaded by bash-completion
mkdir -p ~/.local/share/bash-completion/completions
cp extractStills-completion.bash ~/.local/share/bash-completion/completions/extractStills.py
```

Completion applies when the script is run directly (`extractStills.py`,
`./extractStills.py`, or any path to it) or through a symlink named
`extractStills` — not via `python3 extractStills.py`. It works with or without
the `bash-completion` package.

---

## Troubleshooting

| Symptom | Action |
|---|---|
| `FAIL … blocked: total frame count` | the project does not describe this video; re-run with `-v` and compare the Timing line |
| `FAIL … blocked: project slide sections` | the `.img` is incomplete or edited; `number of slides` ≠ sections |
| `FAIL … not still: slide NNN` | timing drifted into a transition; check the project, or `--margin` |
| `ERROR … NVDEC/CUDA decode failed` | check the driver and `ffmpeg -hwaccels`, or use `--cpu` |
| `ERROR … OpenCV reports no CUDA device` | check the opencv-cuda build, or use `--cpu` |
| nothing matched a pattern | patterns match file names in one directory; check quoting and `.mp4` |
