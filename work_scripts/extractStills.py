#!/usr/bin/env python3
"""
extractStills.py - pull one clean, transition-free still per slide out of an
Imagination-rendered HEVC slideshow .mp4.

Modes
-----
auto    (default) Timing comes from the matching .img project file.
manual  fixed timing (--transition / --still / --drift); the project file,
        when found, is used only to cross-check the result.

Timing model (verified against Cristina-Slideshow-034_AI / -080_AI)
------------------------------------------------------------------
Every slide k gets an *incoming* transition of `speed` seconds (Fast=1,
Normal=4, Slow=8; slide 1 transitions in from the blank background), followed
by `duration` seconds of still.  There is no outro transition:

    block_k       = speed_k + duration_k
    still window  = [sum(block_1..k-1) + speed_k,  sum(block_1..k)
    total frames  = floor(sum(block) * fps)          fps "30 (NTSC)" = 30000/1001

    034: 55 * (9+4) = 715 s  -> floor(715  * 29.97003) = 21428 frames  (video: 21428)
    080: 80 * (9+4) = 1040 s -> floor(1040 * 29.97003) = 31168 frames  (video: 31168)

Frame choice
------------
x265 renders a still gradually (small P-frames refine it after the transition)
until the periodic I-frame (every 250 frames) re-codes it in full.  By default,
the I-frame inside the safe part of each still window is used (closest to the
window centre); otherwise the window centre.  Use --frame-pick center to
always take the window centre.

Decode: NVDEC (ffmpeg -hwaccel cuda) in one pass; only the selected frames
leave the GPU.  Upscale to 2160p: cv2.cuda INTER_CUBIC.
"""
from __future__ import annotations

import argparse
import bisect
import configparser
import contextlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

# Shared pattern module at the repository root (one level up; symlinks resolved).
sys.path.append(str(Path(__file__).resolve().parent.parent))
import globrange  # noqa: E402

__version__ = "1.1.0"

RESOLUTIONS = {"2160p": (3840, 2160), "1080p": (1920, 1080)}
KNOWN_SPEEDS = {1: "Fast", 4: "Normal", 8: "Slow"}
NAME_RE = re.compile(r"^(?P<model>.+?)-Slideshow-(?P<set>\d+)(?P<suffix>.*)$")

# Empirical motion gate: mean absolute difference (0-255 scale) between the
# chosen frame and probe frames ~--margin away in the same GOP.  Settled stills
# measure 0.00 there; 15 frames into the sample transitions measured 0.84-16.
MOTION_MAD_THRESHOLD = 0.5
NEIGHBOUR_MIN_OFFSET = 2      # frames; actual offset = max(this, --margin)
PILLAR_STRIP_PX = 40          # at 1080p; a portrait 2:3 slide has ~600 px bars
PILLAR_BLACK_MAX = 24

JPEG_QUALITY = 95
PNG_COMPRESSION = 3


# Exit codes (documented in README.md).  A batch exits with the highest-priority
# code among its videos: ERROR > USAGE > NOT_STILL > BLOCKED > EXISTS.
EXIT_OK = 0
EXIT_ERROR = 1          # runtime / input error (ffmpeg, missing files, bad project, ...)
EXIT_USAGE = 2          # bad command line (argparse's own code)
EXIT_BLOCKED = 3        # confidence gate: timing math disagrees / score too low
EXIT_NOT_STILL = 4      # empirical check: a target frame is not a clean still
EXIT_EXISTS = 5         # output files already exist (no --overwrite)
EXIT_INTERRUPTED = 130
EXIT_PRECEDENCE = (EXIT_ERROR, EXIT_USAGE, EXIT_NOT_STILL, EXIT_BLOCKED, EXIT_EXISTS)

DEFAULT_LOGFILE = "~/extractStills.log"


class StillsError(Exception):
    """User-facing error carrying its exit code."""

    def __init__(self, message: str, code: int = EXIT_ERROR, short: str = ""):
        super().__init__(message)
        self.code = code
        self.short = short          # one-line form for summaries


# --------------------------------------------------------------------------- #
# Console colour
# --------------------------------------------------------------------------- #
class Color:
    """ANSI styling, decided per stream (stdout / stderr) by setup()."""
    CODES = {"bold": "1", "dim": "2", "under": "4", "red": "31", "green": "32",
             "yellow": "33", "blue": "34", "magenta": "35", "cyan": "36"}
    out = err = False

    @classmethod
    def setup(cls, mode: str) -> None:
        def wanted(stream) -> bool:
            if mode != "auto":
                return mode == "always"
            if os.environ.get("NO_COLOR"):
                return False
            if os.environ.get("FORCE_COLOR"):
                return True
            return stream.isatty() and os.environ.get("TERM") != "dumb"
        cls.out, cls.err = wanted(sys.stdout), wanted(sys.stderr)


def paint(text, *styles: str, err: bool = False) -> str:
    """Wrap already-padded text in ANSI styles (padding first keeps columns aligned)."""
    if not styles or not (Color.err if err else Color.out):
        return str(text)
    codes = ";".join(Color.CODES[s] for s in styles)
    return f"\033[{codes}m{text}\033[0m"


STATUS_STYLE = {"PASS": ("green", "bold"), "FAIL": ("red", "bold"),
                "WARN": ("yellow", "bold"), "INFO": ("cyan",)}
LEVEL_STYLE = {"HIGH": ("green", "bold"), "MEDIUM": ("yellow", "bold"),
               "LOW": ("red",), "FAIL": ("red", "bold")}


# exact = 0/±1 frame; absorbed = non-zero but cannot move any pick out of its
# still; warn = could reach a tight target; fail = model does not fit the video.
DELTA_STYLE = {"exact": ("green", "bold"), "absorbed": ("cyan", "bold"),
               "warn": ("yellow", "bold"), "fail": ("red", "bold")}


def fmt_delta(delta: int, kind: str) -> str:
    return paint(f"Δ {delta:+d} frames", *DELTA_STYLE[kind])


def tag(status: str, err: bool = False) -> str:
    return paint(f"[{status:4}]", *STATUS_STYLE.get(status, ()), err=err)


def heading(text: str) -> str:
    return paint(f"--- {text} ---", "blue", "bold")


def label(text: str) -> str:
    return paint(f"{text:<8}:", "bold")


# --------------------------------------------------------------------------- #
# Output: console (verbosity-filtered, coloured) + optional log (full, plain)
# --------------------------------------------------------------------------- #
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

QUIETEST, SUMMARY, NORMAL, VERBOSE = 0, 1, 2, 3


class Output:
    """Every message has a level.  The console shows messages up to the chosen
    verbosity; the log file (if any) always receives everything, without colour."""

    def __init__(self, level: int, log=None):
        self.level, self.log = level, log
        self._progress_open = False
        self._tty = sys.stdout.isatty()

    def __call__(self, text: str = "", level: int = NORMAL, *, err: bool = False,
                 exact: bool = False) -> None:
        """exact=True: show on the console only at exactly `level` (summary lines)."""
        shown = self.level == level if exact else self.level >= level
        if shown:
            self._close_progress()
            print(text, file=sys.stderr if err else sys.stdout, flush=True)
        self.to_log(text)

    def to_log(self, text: str) -> None:
        if self.log:
            self.log.write(ANSI_RE.sub("", text) + "\n")
            self.log.flush()

    def progress(self, label_: str, done: int, total: int) -> None:
        """Single self-overwriting counter line; NORMAL level on a terminal only."""
        if self.level == NORMAL and self._tty:
            print(f"\r  {label_} {done}/{total}", end="", flush=True)
            self._progress_open = done < total
            if done >= total:
                print(flush=True)

    def _close_progress(self) -> None:
        if self._progress_open:
            print(flush=True)
            self._progress_open = False


# --------------------------------------------------------------------------- #
# Naming
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NameParts:
    model: str
    set_num: str
    suffix: str

    def video_basename(self, suffix: str | None = None) -> str:
        """'<model>-Slideshow-<set><suffix>' (the .img/.mp4 stem); suffix overridable."""
        return f"{self.model}-Slideshow-{self.set_num}{self.suffix if suffix is None else suffix}"

    @property
    def image_basename(self) -> str:
        """'<model>-<set>': images carry neither '-Slideshow-' nor the suffix."""
        return f"{self.model}-{self.set_num}"


def parse_video_name(video: Path) -> NameParts:
    m = NAME_RE.match(video.stem)
    if not m:
        raise StillsError(
            f"Video name '{video.name}' does not match "
            f"'<model name>-Slideshow-<set number><suffix>.mp4'")
    return NameParts(m["model"], m["set"], m["suffix"])


# --------------------------------------------------------------------------- #
# ffprobe
# --------------------------------------------------------------------------- #
@dataclass
class VideoInfo:
    path: Path
    codec: str
    profile: str
    width: int
    height: int
    pix_fmt: str
    fps: Fraction
    avg_fps: Fraction
    frames: int
    duration: float
    start_time: float
    color_space: str
    color_range: str
    keyframes: list[int] = field(default_factory=list)


def run_json(cmd: list[str]) -> dict:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as e:
        raise StillsError(f"'{cmd[0]}' not found on PATH") from e
    except subprocess.CalledProcessError as e:
        raise StillsError(f"{cmd[0]} failed:\n{e.stderr.strip()}") from e
    return json.loads(res.stdout)


def probe_video(path: Path) -> VideoInfo:
    data = run_json([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
        "-show_entries",
        "stream=codec_name,profile,width,height,pix_fmt,r_frame_rate,avg_frame_rate,"
        "nb_frames,nb_read_packets,duration,start_time,color_space,color_range"
        ":format=duration",
        "-of", "json", str(path)])
    if not data.get("streams"):
        raise StillsError(f"No video stream in '{path}'")
    s = data["streams"][0]
    frames = s.get("nb_frames") or s.get("nb_read_packets")
    if not frames:
        raise StillsError(f"ffprobe could not determine the frame count of '{path}'")
    info = VideoInfo(
        path=path,
        codec=s.get("codec_name", "?"),
        profile=s.get("profile", "?"),
        width=int(s["width"]),
        height=int(s["height"]),
        pix_fmt=s.get("pix_fmt", "?"),
        fps=Fraction(s["r_frame_rate"]),
        avg_fps=Fraction(s.get("avg_frame_rate", s["r_frame_rate"])),
        frames=int(frames),
        duration=float(s.get("duration") or data["format"]["duration"]),
        start_time=float(s.get("start_time") or 0.0),
        color_space=s.get("color_space", "unknown"),
        color_range=s.get("color_range", "unknown"),
    )
    # Keyframe indices from packet flags (no decoding, ~0.1 s per video).
    pk = run_json(["ffprobe", "-v", "error", "-select_streams", "v:0",
                   "-show_entries", "packet=pts_time,flags", "-of", "json", str(path)])
    for p in pk.get("packets", []):
        if "K" in p.get("flags", "") and p.get("pts_time") not in (None, "N/A"):
            info.keyframes.append(round((float(p["pts_time"]) - info.start_time) * info.fps))
    info.keyframes.sort()
    return info


# --------------------------------------------------------------------------- #
# Project (.img)
# --------------------------------------------------------------------------- #
@dataclass
class Slide:
    number: int
    filename: str
    duration: Fraction
    transition_id: int
    speed: int
    no_points: int
    angle: int

    @property
    def transition_s(self) -> Fraction:
        # transition_id 0 = "no transition" -> assumed 0 s (not seen in samples).
        return Fraction(self.speed) if self.transition_id else Fraction(0)


@dataclass
class Project:
    path: Path
    declared_slides: int
    fps_text: str
    fps: Fraction | None
    width: int | None
    height: int | None
    slides: list[Slide]


def parse_project_fps(text: str) -> Fraction | None:
    m = re.match(r"\s*(\d+(?:\.\d+)?)", text)
    if not m:
        return None
    base = Fraction(m.group(1))
    if "NTSC" in text.upper() and base.denominator == 1:
        return base * Fraction(1000, 1001)
    return base


def find_project(project_dir: Path, video_basename: str) -> Path | None:
    target = f"{video_basename}.img"
    direct = project_dir / target
    if direct.is_file():
        return direct
    hits = [Path(root) / target for root, _dirs, files in os.walk(project_dir)
            if target in files]
    if len(hits) > 1:
        raise StillsError(f"Multiple '{target}' found under '{project_dir}':\n  "
                          + "\n  ".join(map(str, hits)))
    return hits[0] if hits else None


def parse_project(path: Path) -> Project:
    cp = configparser.ConfigParser(interpolation=None, strict=False)
    cp.optionxform = str
    try:
        cp.read_string(path.read_text(encoding="utf-8", errors="replace"))
    except configparser.Error as e:
        raise StillsError(f"Cannot parse project '{path}': {e}") from e
    if "slideshow settings" not in cp:
        raise StillsError(f"'{path}' has no [slideshow settings] section")
    st = cp["slideshow settings"]
    try:
        declared = int(st["number of slides"])
    except (KeyError, ValueError) as e:
        raise StillsError(f"'{path}': missing/invalid 'number of slides'") from e

    def opt_int(key: str) -> int | None:
        try:
            return int(st[key])
        except (KeyError, ValueError):
            return None

    slides: list[Slide] = []
    sec_re = re.compile(r"^slide (\d+)$")
    for sec in cp.sections():
        m = sec_re.match(sec)
        if not m:
            continue
        s = cp[sec]
        try:
            slides.append(Slide(
                number=int(m.group(1)),
                filename=s.get("filename", ""),
                duration=Fraction(s["duration"]),
                transition_id=int(s.get("transition_id", "0")),
                speed=int(s.get("speed", "0")),
                no_points=int(s.get("no_points", "0")),
                angle=int(s.get("angle", "0")),
            ))
        except (KeyError, ValueError) as e:
            raise StillsError(f"'{path}' [{sec}]: missing/invalid field ({e})") from e
    slides.sort(key=lambda x: x.number)
    fps_text = st.get("fps", "")
    return Project(path, declared, fps_text, parse_project_fps(fps_text),
                   opt_int("video width"), opt_int("video height"), slides)


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #
@dataclass
class PlannedFrame:
    index: int                     # 1-based output number
    still_start: Fraction          # seconds
    still_end: Fraction
    first_frame: int               # first/last frame of the still window
    last_frame: int
    safe_lo: int                   # window shrunk by --margin
    safe_hi: int
    target: int
    pick: str                      # "keyframe" | "center" | "center!" (no safe window)
    slide: Slide | None = None
    expect_portrait: bool | None = None
    source_size: tuple[int, int] | None = None

    def margin_s(self, fps: Fraction) -> float:
        return float(min(self.target - self.first_frame,
                         self.last_frame - self.target) / fps)


def frame_window(start: Fraction, end: Fraction, fps: Fraction) -> tuple[int, int]:
    """Frames n with start <= n/fps < end."""
    return math.ceil(start * fps), math.ceil(end * fps) - 1


def choose_target(first: int, last: int, margin_f: int, keyframes: list[int],
                  pick_mode: str, total: int) -> tuple[int, int, int, str]:
    last = min(last, total - 1)
    center = (first + last) // 2 if last >= first else first
    lo, hi = first + margin_f, last - margin_f
    if lo > hi:
        return center, center, center, "center!"
    if pick_mode == "keyframe":
        ks = [k for k in keyframes if lo <= k <= hi]
        if ks:
            return min(ks, key=lambda k: abs(k - center)), lo, hi, "keyframe"
    return center, lo, hi, "center"


def plan_auto(project: Project, video: VideoInfo, args) -> list[PlannedFrame]:
    fps = video.fps
    margin_f = round(args.margin * fps)
    plan, cum = [], Fraction(0)
    for i, sl in enumerate(project.slides, 1):
        start = cum + sl.transition_s
        end = start + sl.duration
        cum = end
        first, last = frame_window(start, end, fps)
        tgt, lo, hi, pick = choose_target(first, last, margin_f, video.keyframes,
                                          args.frame_pick, video.frames)
        plan.append(PlannedFrame(i, start, end, first, last, lo, hi, tgt, pick, sl))
    return plan


def plan_manual(video: VideoInfo, args) -> list[PlannedFrame]:
    """Fixed timeline: first target = T + S/2, step = T + S + drift."""
    fps = video.fps
    T, S, D = (Fraction(str(args.transition)), Fraction(str(args.still)),
               Fraction(str(args.drift)))
    margin_f = round(args.margin * fps)
    plan, k = [], 0
    while True:
        base = k * (T + S + D)
        if round((base + T + S / 2) * fps) >= video.frames:
            break
        first, last = frame_window(base + T, base + T + S, fps)
        if args.frame_pick == "center":
            tgt = round((base + T + S / 2) * fps)        # exact centre frame
            lo, hi = first + margin_f, last - margin_f
            pick = "center" if lo <= hi else "center!"
        else:
            tgt, lo, hi, pick = choose_target(first, last, margin_f, video.keyframes,
                                              "keyframe", video.frames)
        plan.append(PlannedFrame(k + 1, base + T, base + T + S, first, last,
                                 lo, hi, tgt, pick))
        k += 1
    return plan


def attach_sources(plan: list[PlannedFrame], project: Project | None) -> None:
    """Expected orientation from source image (if present) + angle; info only."""
    if project is None:
        return
    try:
        from PIL import Image
    except ImportError:
        Image = None
    for pf, sl in zip(plan, project.slides):
        pf.slide = pf.slide or sl
        if Image is None or not sl.filename or not os.path.isfile(sl.filename):
            continue
        try:
            with Image.open(sl.filename) as im:
                w, h = im.size
        except OSError:
            continue
        pf.source_size = (w, h)
        if sl.angle % 2:
            w, h = h, w
        pf.expect_portrait = h > w


# --------------------------------------------------------------------------- #
# Confidence
# --------------------------------------------------------------------------- #
@dataclass
class Check:
    name: str
    status: str          # PASS | WARN | FAIL | INFO
    penalty: int
    detail: str


class Assessment:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(self, name: str, status: str, detail: str, penalty: int = 0) -> None:
        self.checks.append(Check(name, status, penalty if status == "WARN" else 0, detail))

    @property
    def failed(self) -> bool:
        return any(c.status == "FAIL" for c in self.checks)

    @property
    def score(self) -> int:
        return 0 if self.failed else max(0, 100 - sum(c.penalty for c in self.checks))

    @property
    def level(self) -> str:
        if self.failed:
            return "FAIL"
        return "HIGH" if self.score >= 90 else "MEDIUM" if self.score >= 70 else "LOW"


def assess(video: VideoInfo, project: Project | None, plan: list[PlannedFrame],
           args) -> tuple[Assessment, dict]:
    a, stats = Assessment(), {}
    fps = video.fps

    # --- video basics -------------------------------------------------------
    if video.fps != video.avg_fps:
        a.add("constant frame rate", "WARN",
              f"r_frame_rate {video.fps} != avg_frame_rate {video.avg_fps}", 15)
    else:
        a.add("constant frame rate", "PASS", f"{video.fps} ({float(fps):.5f} fps)")
    if video.start_time:
        a.add("stream start time", "WARN", f"start_time={video.start_time} (expected 0)", 5)
    a.add("codec", "PASS" if video.codec == "hevc" else "INFO",
          f"{video.codec} {video.profile} {video.pix_fmt} {video.width}x{video.height}")
    if not video.keyframes:
        a.add("keyframe index", "WARN", "no keyframes found in packet scan", 5)

    # --- project ------------------------------------------------------------
    if project is None:
        if args.mode == "auto":
            a.add("project file", "FAIL", "matching .img not found (required in auto mode)")
        else:
            a.add("project file", "WARN",
                  "not found - manual timing cannot be cross-checked against the project", 25)
    else:
        n_sec = len(project.slides)
        nums = [s.number for s in project.slides]
        a.add("project slide sections", "PASS" if n_sec == project.declared_slides else "FAIL",
              f"[slide N] sections={n_sec}, 'number of slides'={project.declared_slides}")
        if nums != list(range(1, n_sec + 1)):
            a.add("slide numbering", "FAIL", "slide sections are not numbered 1..N contiguously")

        if project.fps is None:
            a.add("fps (project vs video)", "WARN",
                  f"cannot parse project fps '{project.fps_text}'; using video fps", 10)
        elif project.fps != video.fps:
            a.add("fps (project vs video)", "FAIL",
                  f"project '{project.fps_text}' = {project.fps} vs video {video.fps}")
        else:
            a.add("fps (project vs video)", "PASS", f"'{project.fps_text}' = {project.fps}")

        if (project.width, project.height) != (video.width, video.height):
            a.add("resolution (project vs video)", "WARN",
                  f"project {project.width}x{project.height} vs video "
                  f"{video.width}x{video.height}", 10)
        else:
            a.add("resolution (project vs video)", "PASS", f"{video.width}x{video.height}")

        sum_dur = sum((s.duration for s in project.slides), Fraction(0))
        sum_tr = sum((s.transition_s for s in project.slides), Fraction(0))
        total_s = sum_dur + sum_tr
        exp_frames = math.floor(total_s * fps)
        delta = video.frames - exp_frames
        implied_tr = ((Fraction(video.frames) / fps - sum_dur) / n_sec) if n_sec else 0
        # Judge the delta by its effect: a timeline mismatch of |delta| frames can
        # shift any target by at most |delta| frames.  If every target keeps more
        # clearance than that from its transitions, the picks are unaffected.
        shift = abs(delta)
        clearance = min((min(pf.target - pf.first_frame, pf.last_frame - pf.target)
                         for pf in plan), default=0)
        half_window = min(((pf.last_frame - pf.first_frame) // 2 for pf in plan), default=0)
        if shift <= 1:
            status, kind, note = "PASS", "exact", ""
        elif shift < clearance and shift < fps / 2:
            status, kind, note = "PASS", "absorbed", f"< {clearance}f clearance, no effect"
        elif shift < half_window:
            status, kind, note = "WARN", "warn", f"> {clearance}f clearance, verify decides"
        else:
            status, kind, note = "FAIL", "fail", f"implied transition {float(implied_tr):.3f}s"
        stats.update(sum_duration=float(sum_dur), sum_transitions=float(sum_tr),
                     expected_seconds=float(total_s), expected_frames=exp_frames,
                     frame_delta=delta, frame_delta_kind=kind,
                     implied_transition=float(implied_tr))
        a.add("total frame count", status,
              f"{fmt_delta(delta, kind)}  video {video.frames} / expected {exp_frames}"
              + (f"  ({note})" if note else ""), 25)

        bad_speed = sorted({s.speed for s in project.slides
                            if s.transition_id and s.speed not in KNOWN_SPEEDS})
        if bad_speed:
            a.add("transition speed values", "WARN",
                  f"unrecognised speed(s) {bad_speed}; treated as seconds", 10)
        else:
            used = sorted({s.speed for s in project.slides if s.transition_id})
            a.add("transition speed values", "PASS",
                  ", ".join(f"{v}={KNOWN_SPEEDS[v]} ({v}s)" for v in used) or "none")
        no_tr = [s.number for s in project.slides if not s.transition_id]
        if no_tr:
            a.add("slides without transition", "WARN",
                  f"transition_id=0 on slides {no_tr} (assumed 0 s; unverified)", 10)
        moving = [s.number for s in project.slides if s.no_points > 1]
        if moving:
            a.add("pan/zoom motion in still", "WARN",
                  f"no_points>1 on slides {moving}: still window is not static", 15)

        if args.mode == "manual":
            if len(plan) != project.declared_slides:
                a.add("manual count vs project", "FAIL",
                      f"manual timing yields {len(plan)} frames, project has "
                      f"{project.declared_slides} slides")
            else:
                ref = plan_auto(project, video, args)
                outside = [pf.index for pf, r in zip(plan, ref)
                           if not (r.first_frame <= pf.target <= r.last_frame)]
                if outside:
                    a.add("manual targets vs project windows", "FAIL",
                          f"targets outside project still windows: slides {outside}")
                else:
                    worst = min(min(pf.target - r.first_frame, r.last_frame - pf.target)
                                for pf, r in zip(plan, ref)) / fps
                    a.add("manual targets vs project windows", "PASS",
                          f"all inside; tightest margin {float(worst):.2f}s")

    # --- plan ---------------------------------------------------------------
    if not plan:
        a.add("planned frames", "FAIL", "no frames planned")
    else:
        past_end = [pf.index for pf in plan if pf.target >= video.frames]
        if past_end:
            a.add("targets inside video", "FAIL", f"slides {past_end} beyond last frame")
        tight = [pf.index for pf in plan if pf.pick == "center!"]
        if tight:
            a.add("safe window", "WARN",
                  f"still window shorter than 2 x --margin on slides {tight}", 20)
        else:
            worst = min(pf.margin_s(fps) for pf in plan)
            a.add("safe window", "PASS", f"tightest target-to-transition margin {worst:.2f}s")
        nk = sum(pf.pick == "keyframe" for pf in plan)
        a.add("frame pick", "INFO", f"{nk}/{len(plan)} on in-window I-frames, "
                                    f"{len(plan) - nk} at window centre")
        stats["planned"] = len(plan)
    return a, stats


# --------------------------------------------------------------------------- #
# Decode / verify / write
# --------------------------------------------------------------------------- #
def color_args(video: VideoInfo) -> str:
    cs = video.color_space
    matrix = {"bt709": "bt709", "bt2020nc": "bt2020", "bt2020c": "bt2020",
              "fcc": "fcc", "smpte240m": "smpte240m"}.get(cs, "bt601")  # untagged -> bt601
    rng = "pc" if video.color_range == "pc" else "tv"
    return f"in_color_matrix={matrix}:in_range={rng}"


def select_expr(frames: list[int]) -> str:
    """eq(n,a)+eq(n,b)+... as a balanced tree: ffmpeg's expression parser rejects
    flat chains deeper than ~100 terms, a tree stays at log2(n) depth."""
    if len(frames) == 1:
        return f"eq(n\\,{frames[0]})"
    mid = len(frames) // 2
    return f"({select_expr(frames[:mid])}+{select_expr(frames[mid:])})"


def decode_frames(video: VideoInfo, wanted: list[int], use_cpu: bool):
    """Yield (frame_index, bgr24 ndarray) for each wanted index, in order, one pass."""
    wanted = sorted(set(wanted))
    expr = select_expr(wanted)
    convert = (f"scale={color_args(video)}:flags=bicubic+accurate_rnd+full_chroma_int,"
               f"format=bgr24")
    if use_cpu:
        hw, vf = [], f"select='{expr}',{convert}"
    else:
        hw = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        vf = f"select='{expr}',hwdownload,format=nv12,{convert}"
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", *hw,
           "-i", str(video.path), "-map", "0:v:0", "-vf", vf,
           "-fps_mode", "passthrough", "-frames:v", str(len(wanted)),
           "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    fsize = video.width * video.height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    got, complete = 0, False
    try:
        for n in wanted:
            buf = proc.stdout.read(fsize)
            if len(buf) != fsize:
                break
            got += 1
            yield n, np.frombuffer(buf, np.uint8).reshape(video.height, video.width, 3)
        complete = True
    finally:
        if not complete and proc.poll() is None:
            proc.kill()                      # consumer aborted (or short read)
        proc.stdout.close()
        err = proc.stderr.read().decode(errors="replace").strip()
        rc = proc.wait()
    if got != len(wanted) or rc != 0:
        hint = ""
        if not use_cpu and re.search(r"cuda|nvdec|cuvid|hwaccel|hwdownload|device", err, re.I):
            hint = ("\nHint: NVDEC/CUDA decode failed - check the NVIDIA driver and "
                    "ffmpeg CUDA support (`ffmpeg -hwaccels`), or retry with --cpu.")
        err = "\n".join(l if len(l) <= 300 else l[:300] + " ...[truncated]"
                        for l in err.splitlines())
        raise StillsError(f"ffmpeg delivered {got}/{len(wanted)} frames (exit {rc}).\n"
                          f"{err}{hint}")


def mad(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.absdiff(a, b).mean())


def has_pillarbars(img: np.ndarray) -> bool:
    w = max(1, PILLAR_STRIP_PX * img.shape[1] // 1920)
    return bool(img[:, :w].max() <= PILLAR_BLACK_MAX and img[:, -w:].max() <= PILLAR_BLACK_MAX)


class Resizer:
    def __init__(self, size: tuple[int, int], use_cpu: bool):
        self.size, self.use_cpu, self.gpu = size, use_cpu, None
        if not use_cpu:
            try:
                ndev = cv2.cuda.getCudaEnabledDeviceCount()
            except (AttributeError, cv2.error):
                ndev = 0
            if ndev < 1:
                raise StillsError("OpenCV reports no CUDA device (needed for the GPU "
                                  "upscale). Check the opencv-cuda build, or use --cpu.")
            self.gpu = cv2.cuda_GpuMat()

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if (img.shape[1], img.shape[0]) == self.size:
            return img
        if self.use_cpu:
            return cv2.resize(img, self.size, interpolation=cv2.INTER_CUBIC)
        self.gpu.upload(img)
        # OpenCV 5: dst= is not propagated back to Python - use the return value.
        return cv2.cuda.resize(self.gpu, self.size, interpolation=cv2.INTER_CUBIC).download()


@dataclass
class VerifyResult:
    pf: PlannedFrame
    mad: float
    pillarbars: bool

    @property
    def static(self) -> bool:
        return self.mad <= MOTION_MAD_THRESHOLD


def neighbours(pf: PlannedFrame, video: VideoInfo, offset: int) -> list[int]:
    """Target plus probes `offset` frames either side, kept inside the predicted
    still window *and* the target's GOP.  Within one GOP a settled still decodes
    bit-identically (MAD 0.00); across an I-frame the refresh alone is ~1 MAD,
    which would mask the start of a slow fade."""
    k = bisect.bisect_right(video.keyframes, pf.target)
    gop_lo = video.keyframes[k - 1] if k else 0
    gop_hi = video.keyframes[k] - 1 if k < len(video.keyframes) else video.frames - 1
    lo = max(pf.first_frame, gop_lo)
    hi = min(pf.last_frame, gop_hi, video.frames - 1)
    return sorted({max(lo, pf.target - offset), pf.target, min(hi, pf.target + offset)})


def process(video: VideoInfo, plan: list[PlannedFrame], args, writer=None, progress=None):
    """Decode, verify each planned frame; hand verified frames to writer(pf, img).
    progress(done, total) is called after each slide."""
    offset = max(NEIGHBOUR_MIN_OFFSET, round(args.margin * video.fps))
    needs = {pf.index: neighbours(pf, video, offset) for pf in plan}
    owner: dict[int, list[PlannedFrame]] = {}
    for pf in plan:
        for n in needs[pf.index]:
            owner.setdefault(n, []).append(pf)
    cache: dict[int, np.ndarray] = {}
    results: list[VerifyResult] = []
    pending = {pf.index: pf for pf in plan}
    # closing() kills ffmpeg promptly if verification aborts mid-stream.
    with contextlib.closing(decode_frames(video, list(owner), args.cpu)) as frames:
        for n, img in frames:
            cache[n] = img
            for pf in owner[n]:
                if pf.index not in pending or not all(m in cache for m in needs[pf.index]):
                    continue
                del pending[pf.index]
                tgt = cache[pf.target]
                diffs = [mad(tgt, cache[m]) for m in needs[pf.index] if m != pf.target]
                vr = VerifyResult(pf, max(diffs, default=0.0), has_pillarbars(tgt))
                results.append(vr)
                if not vr.static:
                    raise StillsError(
                        f"Slide {pf.index:03d}: motion at frame {pf.target} "
                        f"(MAD {vr.mad:.2f} > {MOTION_MAD_THRESHOLD}) - target is not "
                        f"in a clean still. Nothing was saved.", EXIT_NOT_STILL,
                        short=f"not still: slide {pf.index:03d} (MAD {vr.mad:.2f})")
                if writer:
                    writer(pf, tgt)
                if progress:
                    progress(len(results), len(plan))
            # drop frames whose slides are all done
            for m in [m for m in cache if all(p.index not in pending for p in owner[m])]:
                del cache[m]
    return results


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def fmt_t(sec: float) -> str:
    m, s = divmod(sec, 60)
    return f"{int(m):02d}:{s:06.3f}"


PICK_STYLE = {"keyframe": ("green",), "center": (), "center!": ("yellow", "bold")}


class Cell:
    """Table cell made of (text, styles) segments; width is measured on plain text
    so ANSI codes never disturb alignment."""

    def __init__(self, *segments: tuple[str, tuple[str, ...]]):
        self.segments = list(segments)

    def add(self, text: str, *styles: str) -> "Cell":
        self.segments.append((text, styles))
        return self

    @property
    def plain(self) -> str:
        return "".join(t for t, _ in self.segments)

    def render(self, width: int, block: int, align: str) -> str:
        """Align within the column's data block (`block` = widest value), then
        centre that block in the column so data sits centred under its header."""
        inner = block - len(self.plain)
        left = {"<": 0, ">": inner, "^": inner // 2}[align]
        outer = width - block
        left += outer // 2
        body = "".join(paint(t, *s) for t, s in self.segments)
        return " " * left + body + " " * (width - len(self.plain) - left)


def range_cells(pairs: list[tuple[str, str]]) -> list[Cell]:
    """'a-b' cells with every '-' in the same column."""
    wa = max((len(a) for a, _ in pairs), default=0)
    wb = max((len(b) for _, b in pairs), default=0)
    return [Cell((f"{a:>{wa}}-{b:<{wb}}", ())) for a, b in pairs]


def print_plan(out: Output, plan: list[PlannedFrame], fps: Fraction, names: list[str],
               margin: float) -> None:
    # (header, data alignment); headers are always centred over their column.
    cols = [("#", ">"), ("trans", ">"), ("still window sec", "^"), ("frames", "^"),
            ("target", ">"), ("time", ">"), ("pick", "^"), ("margin", ">"),
            ("output", "<"), ("source", "<"), ("size", ">"), ("notes", "<")]
    windows = range_cells([(f"{float(pf.still_start):.3f}", f"{float(pf.still_end):.3f}")
                           for pf in plan])
    frames = range_cells([(str(pf.first_frame), str(pf.last_frame)) for pf in plan])
    rows: list[list[Cell]] = []
    for pf, name, win, frm in zip(plan, names, windows, frames):
        sl = pf.slide
        src, size, notes = Cell(), Cell(), Cell()
        if sl and sl.filename:
            src.add(os.path.basename(sl.filename), "dim")
            if pf.source_size:
                size.add("{}x{}".format(*pf.source_size), "dim")
                tags = (["rot"] if sl.angle % 2 else []) + \
                       (["portrait"] if pf.expect_portrait else [])
                for i, t in enumerate(tags):
                    notes.add((" " if i else "") + t, "magenta" if t == "rot" else "cyan")
            else:
                size.add("missing", "yellow")
        m = pf.margin_s(fps)
        rows.append([
            Cell((str(pf.index), ("bold",))),
            Cell((f"{sl.transition_id}@{sl.speed}s" if sl else "-", ("dim",))),
            win, frm,
            Cell((str(pf.target), ("bold",))),
            Cell((fmt_t(float(pf.target / fps)), ())),
            Cell((pf.pick, PICK_STYLE.get(pf.pick, ()))),
            Cell((f"{m:.2f}s", ("yellow",) if m < 2 * margin else ())),
            Cell((name, ("cyan",))),
            src, size, notes,
        ])
    blocks = [max([0] + [len(r[i].plain) for r in rows]) for i in range(len(cols))]
    widths = [max(len(h), b) for (h, _), b in zip(cols, blocks)]
    gap = "  "
    head = gap.join(f"{h:^{w}}" for (h, _), w in zip(cols, widths))
    out("\n" + paint(head, "bold", "under"), VERBOSE)       # underline spans the full table
    for r in rows:
        out(gap.join(c.render(w, b, a)
                     for c, w, b, (_, a) in zip(r, widths, blocks, cols)).rstrip(), VERBOSE)


def print_assessment(out: Output, a: Assessment, stats: dict, project: Project | None,
                     args, title: str) -> None:
    out("\n" + heading(title))
    if project is not None and "expected_frames" in stats:
        out(f"  {paint('Slides:', 'bold')} project declares {project.declared_slides}, "
            f"sections {len(project.slides)}, planned {stats.get('planned', 0)}")
        out(f"  {paint('Timing:', 'bold')} {stats['sum_duration']:g}s stills + "
            f"{stats['sum_transitions']:g}s transitions = "
            f"{stats['expected_seconds']:g}s -> {stats['expected_frames']} frames  "
            f"{fmt_delta(stats['frame_delta'], stats['frame_delta_kind'])}  "
            + paint(f"(video-implied transition {stats['implied_transition']:.3f}s)", "dim"))
    elif "planned" in stats:
        out(f"  {paint('Slides planned:', 'bold')} {stats['planned']}")
    for c in a.checks:
        pen = paint(f" (-{c.penalty})", "yellow", "bold") if c.penalty else ""
        name = paint(c.name, "red", "bold") if c.status == "FAIL" else paint(c.name, "bold")
        out(f"  {tag(c.status)} {name}: {c.detail}{pen}")
    lvl = paint(f"{a.level} ({a.score}/100)", *LEVEL_STYLE[a.level])
    out(f"  {paint('Confidence:', 'bold')} {lvl}"
        + paint(f"  saving requires no FAIL and >= {args.min_confidence}", "dim"))


# --------------------------------------------------------------------------- #
# Video selection: literal paths and patterns  * ? [...] {N-M} {N-}
# (pattern rules live in ../globrange.py, shared with ken_burns --img_path)
# --------------------------------------------------------------------------- #
@dataclass
class VideoEntry:
    label: str                 # video stem, or the spec when it could not be resolved
    path: Path | None = None
    error: str = ""
    code: int = EXIT_OK


def resolve_videos(specs: list[str]) -> list[VideoEntry]:
    """Expand --video arguments (see globrange.expand)."""
    entries = []
    for m in globrange.expand(specs, "file", extensions=(".mp4",), noun="video"):
        if m.error:
            entries.append(VideoEntry(m.spec, error=m.error,
                                      code=EXIT_USAGE if m.usage else EXIT_ERROR))
        else:
            entries.append(VideoEntry(m.path.stem, m.path))
    return entries


# --------------------------------------------------------------------------- #
# Per-video result and summary line
# --------------------------------------------------------------------------- #
RESULT_STYLE = {"PASS": ("green", "bold"), "WARN": ("yellow", "bold"),
                "FAIL": ("red", "bold"), "ERROR": ("red", "bold")}


@dataclass
class Result:
    name: str
    code: int = EXIT_OK
    status: str = "PASS"           # PASS | WARN | FAIL | ERROR
    detail: str = ""
    confidence: int | None = None
    slides: str = ""
    delta: int | None = None

    def fail(self, code: int, detail: str) -> "Result":
        self.code = code
        self.status = "FAIL" if code in (EXIT_BLOCKED, EXIT_NOT_STILL, EXIT_EXISTS) else "ERROR"
        self.detail = detail
        return self


def summary_line(r: Result, name_width: int) -> str:
    conf = f"conf={r.confidence}" if r.confidence is not None else ""
    slides = f"slides={r.slides}" if r.slides else ""
    delta = f"delta={r.delta:+d}" if r.delta is not None else ""
    return (f"{paint(f'{r.status:<5}', *RESULT_STYLE[r.status])}  {r.name:<{name_width}}  "
            f"{conf:<8}  {slides:<14}  {delta:<12}  {r.detail}").rstrip()


def first_line(text: str, limit: int = 110) -> str:
    line = str(text).strip().splitlines()[0] if str(text).strip() else ""
    return line if len(line) <= limit else line[:limit - 3] + "..."


# --------------------------------------------------------------------------- #
# One video
# --------------------------------------------------------------------------- #
def run_video(path: Path, args, out: Output, position: str) -> Result:
    res = Result(path.stem)
    try:
        _run_video(path, args, out, position, res)
    except StillsError as e:
        out(f"\n{paint('ERROR:', 'red', 'bold', err=True)} {e}", NORMAL, err=True)
        res.fail(e.code, e.short or first_line(e))
    return res


def _run_video(path: Path, args, out: Output, position: str, res: Result) -> None:
    names = parse_video_name(path)
    out_dir_name = f"{names.video_basename(args.suffix)}_Extracted"
    ext = "png" if args.png else "jpg"
    size = RESOLUTIONS[args.resolution]

    out.to_log(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] {path}")
    run_kind = (paint(" DRY RUN ", "cyan", "bold") if args.dry_run
                else paint(" LIVE ", "magenta", "bold"))
    out("\n" + paint(f"=== {position} {path.stem} | {args.mode} |", "blue", "bold")
        + run_kind + paint("===", "blue", "bold"))

    video = probe_video(path)
    project = None
    if args.project_dir is not None:
        ppath = find_project(args.project_dir, names.video_basename())
        if ppath is None and args.mode == "auto":
            raise StillsError(f"project '{names.video_basename()}.img' not found under "
                              f"'{args.project_dir}' (required in auto mode)",
                              short=f"project not found: {names.video_basename()}.img")
        project = parse_project(ppath) if ppath else None

    plan = plan_auto(project, video, args) if (args.mode == "auto" and project) else (
        plan_manual(video, args) if args.mode == "manual" else [])
    attach_sources(plan, project)
    pad = max(3, len(str(len(plan))))
    fnames = [f"{names.image_basename}-{pf.index:0{pad}d}.{ext}" for pf in plan]
    final_dir = args.output_base / out_dir_name if args.output_base else None

    out(f"{label('Video')} {video.path}  "
        + paint(f"({video.codec} {video.width}x{video.height} {float(video.fps):.3f}fps, "
                f"{video.frames} frames, {video.duration:.3f}s, "
                f"{len(video.keyframes)} keyframes)", "dim"))
    out(f"{label('Project')} "
        + (str(project.path) if project else paint("(not found)", "red", "bold")))
    out(f"{label('Names')} model {paint(names.model, 'cyan')}, set "
        f"{paint(names.set_num, 'cyan')}, suffix {paint(repr(names.suffix), 'cyan')} -> "
        f"folder {paint(out_dir_name, 'cyan')}, images "
        f"{paint(fnames[0] if fnames else '-', 'cyan')}", VERBOSE)
    out(f"{label('Output')} {args.resolution} {size[0]}x{size[1]} .{ext}  ->  "
        + (str(final_dir) if final_dir else paint("(dry run)", "dim")))
    if args.mode == "manual":
        out(f"{label('Manual')} first target {args.transition + args.still / 2:g}s, "
            f"step {args.transition + args.still + args.drift:g}s", VERBOSE)

    a, stats = assess(video, project, plan, args)
    res.confidence = a.score
    res.slides = f"{len(plan)}/{project.declared_slides}" if project else str(len(plan))
    res.delta = stats.get("frame_delta")
    print_plan(out, plan, video.fps, fnames, args.margin)
    print_assessment(out, a, stats, project, args, "Pre-decode assessment")
    warnings = [c.name for c in a.checks if c.status == "WARN"]

    if a.failed or a.score < args.min_confidence:
        fails = [c.name for c in a.checks if c.status == "FAIL"]
        reason = fails[0] if fails else f"confidence {a.score} < {args.min_confidence}"
        out(f"\n{paint('[BLOCKED]', 'red', 'bold')} Confidence gate not met - "
            "nothing decoded or saved.")
        res.fail(EXIT_BLOCKED, f"blocked: {reason}")
        return

    if final_dir is not None:
        collisions = [f for f in fnames if (final_dir / f).exists()]
        if collisions and not args.overwrite:
            raise StillsError(
                f"'{final_dir}' already contains {len(collisions)} of the target files "
                f"(e.g. '{collisions[0]}'); use --overwrite to replace them.",
                EXIT_EXISTS, short=f"output exists: {final_dir}")

    def finish(action: str) -> None:
        res.status = "WARN" if warnings else "PASS"
        res.detail = action + (f"  warn: {', '.join(warnings)}" if warnings else "")

    if args.dry_run and not args.verify:
        out(f"\n{paint('[DRY-RUN]', 'cyan', 'bold')} {len(plan)} frames planned; "
            f"{paint('would SAVE', 'green', 'bold')}. Nothing written.")
        finish(f"would save {len(plan)}")
        return

    if args.dry_run:                                     # --dryRun --verify
        results = process(video, plan, args,
                          progress=lambda d, t: out.progress("Verifying ", d, t))
        warnings += report_verify(out, results)
        out(f"\n{paint('[DRY-RUN]', 'cyan', 'bold')} "
            f"{paint(f'Verification passed for {len(results)} frames.', 'green', 'bold')}"
            " Nothing written.")
        finish(f"verified {len(results)}, would save")
        return

    # ---- live: stage, verify, then commit --------------------------------
    stage = args.output_base / f".{out_dir_name}.partial-{os.getpid()}"
    stage.mkdir(parents=True, exist_ok=False)
    resize = Resizer(size, args.cpu)
    params = ([cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION] if args.png
              else [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    written: list[Path] = []

    def writer(pf: PlannedFrame, img: np.ndarray) -> None:
        dst = stage / fnames[pf.index - 1]
        if not cv2.imwrite(str(dst), resize(img), params):
            raise StillsError(f"cv2.imwrite failed for '{dst}'")
        written.append(dst)
        out(f"  {paint('->', 'green')} {dst.name}  "
            + paint(f"frame {pf.target} ({fmt_t(float(pf.target / video.fps))})", "dim"),
            VERBOSE)

    try:
        out("\n" + paint(f"Decoding {len(plan)} targets "
                         f"({'CPU' if args.cpu else 'NVDEC'})...", "blue", "bold"), VERBOSE)
        results = process(video, plan, args, writer,
                          progress=lambda d, t: out.progress("Extracting", d, t))
        if len(written) != len(plan):
            raise StillsError(f"only {len(written)}/{len(plan)} frames produced")
    except BaseException:
        for f in written:
            f.unlink(missing_ok=True)
        shutil.rmtree(stage, ignore_errors=True)   # our own staging dir only
        raise
    warnings += report_verify(out, results)

    final_dir.mkdir(parents=True, exist_ok=True)
    for f in written:
        os.replace(f, final_dir / f.name)
    stage.rmdir()
    out(f"\n{paint('[SUCCESS]', 'green', 'bold')} {len(written)} frames "
        f"({args.resolution}, .{ext}) -> {paint(f'{final_dir}/', 'cyan')}")
    finish(f"saved {len(written)} -> {final_dir}/")


def report_verify(out: Output, results: list[VerifyResult]) -> list[str]:
    """Print the verification block; return names of warnings raised."""
    worst = max(results, key=lambda r: r.mad)
    out("\n" + heading("Frame verification"))
    out(f"  {tag('PASS')} {paint('stillness', 'bold')}: all {len(results)} targets static "
        f"(max neighbour MAD {worst.mad:.3f} on slide {worst.pf.index:03d}, "
        f"threshold {MOTION_MAD_THRESHOLD})")
    bars = [r.pf.index for r in results if r.pillarbars]
    out(f"  {tag('INFO')} {paint('pillarboxed frames (portrait)', 'bold')}: "
        f"{len(bars)} {paint(str(bars), 'dim') if bars else ''}")
    mism = [r.pf.index for r in results
            if r.pf.expect_portrait is not None and r.pf.expect_portrait != r.pillarbars]
    if mism:
        out(f"  {tag('WARN')} {paint('orientation vs pillarbars', 'bold')}: disagree on "
            f"slides {mism} (source image/angle vs frame; check slide alignment or "
            f"dark image edges)")
        return ["orientation vs pillarbars"]
    return []


# --------------------------------------------------------------------------- #
# Batch summary
# --------------------------------------------------------------------------- #
FAIL_LABELS = {EXIT_BLOCKED: "blocked by confidence gate",
               EXIT_NOT_STILL: "target not a clean still",
               EXIT_EXISTS: "output already exists"}


def batch_exit_code(results: list[Result], interrupted: bool) -> int:
    if interrupted:
        return EXIT_INTERRUPTED
    codes = {r.code for r in results}
    return next((c for c in EXIT_PRECEDENCE if c in codes), EXIT_OK)


def fmt_elapsed(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    return f"{m}m{s:02d}s" if m else f"{sec:.1f}s"


def print_batch_summary(out: Output, results: list[Result], total: int, elapsed: float,
                        code: int, logpath: Path | None, interrupted: bool) -> None:
    ok = [r for r in results if r.code == EXIT_OK]
    warned = [r for r in ok if r.status == "WARN"]
    bad = [r for r in results if r.code != EXIT_OK]
    width = max((len(r.name) for r in bad), default=0)

    out("\n" + paint(f"=== Batch summary: {len(results)} of {total} videos in "
                     f"{fmt_elapsed(elapsed)} ===", "blue", "bold"), SUMMARY)
    out(f"  {paint('PASS ', *RESULT_STYLE['PASS'])} {len(ok):4d}"
        + (paint(f"   ({len(warned)} with warnings)", "yellow") if warned else ""), SUMMARY)
    for c, text in FAIL_LABELS.items():
        n = sum(r.code == c for r in results)
        if n:
            out(f"  {paint('FAIL ', *RESULT_STYLE['FAIL'])} {n:4d}   {text} "
                + paint(f"(exit {c})", "dim"), SUMMARY)
    n_err = sum(r.status == "ERROR" for r in results)
    if n_err:
        out(f"  {paint('ERROR', *RESULT_STYLE['ERROR'])} {n_err:4d}   input / runtime errors "
            + paint("(exit 1 or 2)", "dim"), SUMMARY)
    if interrupted:
        out(f"  {paint('STOP ', 'yellow', 'bold')} {total - len(results):4d}   not processed "
            "(interrupted)", SUMMARY)

    if bad:
        out("\n  " + paint("Needs attention:", "bold"), SUMMARY)
        for r in bad:
            out(f"    {paint(f'{r.status:<5}', *RESULT_STYLE[r.status])}  {r.name:<{width}}  "
                f"{paint(f'exit {r.code}', 'dim')}  {r.detail}", SUMMARY)

    if bad or warned:
        hints = []
        if out.level < VERBOSE:
            hints.append("re-run the affected sets with -v for full diagnostics")
        hints.append(f"full diagnostics for every set are in {logpath}" if logpath else
                     "add --logfile to keep full diagnostics while the console stays quiet")
        out("\n  " + paint("Hint:", "cyan", "bold") + " " + "; ".join(hints) + ".", SUMMARY)
    out(f"\n  Exit code: {paint(str(code), 'green' if code == 0 else 'red', 'bold')}", SUMMARY)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
# Help colour scheme - aligned with Python 3.14 argparse's own theme so the text
# and argparse's option column read as one: option names bold cyan, metavars
# bold yellow, headings bold blue.
def h_opt(text: str) -> str:            # reference to a CLI option
    return paint(text, "cyan", "bold")


def h_var(text: str) -> str:            # placeholder / metavar
    return paint(text, "yellow", "bold")


def h_val(text: str) -> str:            # literal value: choice, default, pattern, code
    return paint(text, "green")


def h_lab(text: str) -> str:            # label inside a help entry
    return paint(text, "magenta")


def h_head(text: str) -> str:           # section heading (same as argparse)
    return paint(text, "blue", "bold")


def h_text(*lines: str, choices=None, default: str | None = None) -> str:
    """Help entry: body lines, then 'Choices:' and 'Default:' as the last lines."""
    out = list(lines)
    if choices:
        out.append(f"{h_lab('Choices:')} " + ", ".join(h_val(c) for c in choices))
    if default is not None:
        out.append(f"{h_lab('Default:')} {default}")
    return "\n".join(out)


def extended_help(prog: str) -> str:
    """Text for --help-all, appended to the normal help."""
    o, val, hd = h_opt, h_val, h_head
    p = paint(prog, "magenta", "bold")

    def rows(pairs, width):
        return "\n".join(f"  {k}{' ' * (width - len(ANSI_RE.sub('', k)))}{d}"
                         for k, d in pairs)

    patterns = rows([
        (val("*  ?  [abc]  [!abc]"), "as in the shell"),
        (val("{N-M}"), f"a whole number from N to M, e.g. {val('{001-025}')}"),
        (val("{N-}"), f"from N to {globrange.RANGE_OPEN_MAX}"),
    ], 23)
    verbosity = rows([
        (o("-qq"), "silent; exit code only (command-line errors still print)"),
        (o("-q"), "one summary line per video, plus the batch summary"),
        ("(default)", "per-video checks and verdict"),
        (o("-v"), "everything, including the per-slide plan table"),
    ], 11)
    codes = rows([
        (val(str(EXIT_OK)), "success (including passes with warnings)"),
        (val(str(EXIT_ERROR)), "runtime or input error"),
        (val(str(EXIT_USAGE)), "bad command line, including malformed patterns"),
        (val(str(EXIT_BLOCKED)), "blocked by the confidence gate"),
        (val(str(EXIT_NOT_STILL)), "a target frame is not a clean still"),
        (val(str(EXIT_EXISTS)), f"output already exists (see {o('--overwrite')})"),
        (val(str(EXIT_INTERRUPTED)), "interrupted"),
    ], 6)
    order = " > ".join(val(str(c)) for c in EXIT_PRECEDENCE)
    return f"""
{hd('Video patterns')} (quote them so the shell passes them through unchanged):
{patterns}
  Without a directory the current directory is searched; only {val('.mp4')} files match.

{hd('Verbosity')} (console only; {o('--logfile')} always receives full detail):
{verbosity}

{hd('Exit codes')}:
{codes}
  A batch returns the most serious code among its videos: {order}.

{hd('Examples')}:
  {p} {o('--dryRun')} {o('-q')} {o('--project-dir')} {val('projects')} \\
      {o('--video')} {val("'Nikki Cooper-Slideshow-{001-025}*'")}
  {p} {o('--project-dir')} {val('projects')} {o('--output_base')} {val('stills')} \\
      {o('--logfile')} {o('--video')} {val('~/Videos/*.mp4')}
"""


class HelpAllAction(argparse.Action):
    def __init__(self, option_strings, dest=argparse.SUPPRESS,
                 default=argparse.SUPPRESS, help=None):
        super().__init__(option_strings, dest, nargs=0, default=default, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.epilog = None                 # the pointer to --help-all is redundant here
        parser.print_help()
        print(extended_help(parser.prog))
        parser.exit()


def color_mode_from_argv(argv: list[str]) -> str:
    """--color must be known before the parser is built (help text is coloured)."""
    mode = "auto"
    for i, a in enumerate(argv):
        name, _, value = a.partition("=")
        if len(name) >= 4 and "--color".startswith(name):     # argparse allows --col
            mode = value or (argv[i + 1] if i + 1 < len(argv) else mode)
    return mode if mode in ("auto", "always", "never") else "auto"


def build_parser(color_mode: str = "auto") -> argparse.ArgumentParser:
    """Call after Color.setup(): help strings are coloured when they are built."""
    o, v, val, lab, H = h_opt, h_var, h_val, h_lab, h_text
    kwargs = {}
    if sys.version_info >= (3, 14):          # argparse's own colouring follows ours
        kwargs["color"] = Color.out
        if color_mode == "always":              # argparse also checks for a terminal
            os.environ.setdefault("FORCE_COLOR", "1")
    p = argparse.ArgumentParser(
        prog="extractStills.py", add_help=False,
        formatter_class=argparse.RawTextHelpFormatter,
        description=(f"Extract one transition-free still per slide from Imagination HEVC\n"
                     f"slideshow videos, timed from each video's .img project file\n"
                     f"({o('--mode')} {val('auto')}) or from fixed timing "
                     f"({o('--mode')} {val('manual')})."),
        epilog=(f"Run {o('--help-all')} for video patterns, verbosity levels, exit codes "
                f"and examples."),
        **kwargs)
    p._optionals.title = "Options"

    p.add_argument("-h", "--help", action="help",
                   help="Show this help and exit.")
    p.add_argument("--help-all", action=HelpAllAction,
                   help=H("Show this help plus video patterns, verbosity",
                          "levels, exit codes and examples, then exit."))
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}",
                   help="Show the version and exit.")

    g = p.add_argument_group("Input")
    g.add_argument("--video", required=True, nargs="+", metavar="VIDEO",
                   help=H("Video files or glob / numeric-range patterns.",
                          f"{lab('Supports:')} {val('*  ?  [abc]  [!abc]  {N-M}  {N-}')}",
                          f"  {val('{N-M}')} a whole number from N to M,",
                          f"        e.g. {val('Set-{001-025}*')}",
                          f"  {val('{N-}')}  from N to {globrange.RANGE_OPEN_MAX}",
                          "Without a directory the current directory is",
                          f"searched; only {val('.mp4')} files match. Quote patterns.",
                          f"{lab('Specify:')} {v('VIDEO')} [{v('VIDEO')} ...]"))
    g.add_argument("--project-dir", "--project_dir", dest="project_dir", type=Path,
                   metavar="DIR",
                   help=H(f"Directory holding {v('<video_basename>')}.img project",
                          "files; searched recursively.",
                          f"Required with {o('--mode')} {val('auto')}."))

    g = p.add_argument_group("Output")
    g.add_argument("--output_base", "--output-base", dest="output_base", type=Path,
                   metavar="DIR",
                   help=H("Root of the output tree; frames are written to",
                          f"{v('DIR')}/{v('<video_basename>')}_Extracted/"
                          f"{v('<model>')}-{v('<set>')}-{v('NNN')}.{v('<ext>')}",
                          f"Required unless {o('--dryRun')}."))
    g.add_argument("--resolution", choices=RESOLUTIONS, default="2160p",
                   help=H(f"Output size: {val('2160p')} = 3840x2160, "
                          f"{val('1080p')} = 1920x1080.",
                          choices=RESOLUTIONS, default=val("2160p")))
    g.add_argument("--png", action="store_true",
                   help=H("Write lossless PNG instead of JPEG.",
                          default=f"JPEG, quality {val(str(JPEG_QUALITY))}"))
    g.add_argument("--suffix", metavar="SUFFIX", default=None,
                   help=H(f"Replace the video-name suffix (e.g. {val(repr('_AI'))}) in",
                          f"the output folder name; may be {val(repr(''))}.",
                          default="the suffix from the video name"))
    g.add_argument("--overwrite", action="store_true",
                   help=H("Allow replacing images in an existing output folder.",
                          default="existing images are never replaced"))

    g = p.add_argument_group("Timing")
    g.add_argument("--mode", choices=("auto", "manual"), default="auto",
                   help=H(f"{val('auto')}:   timing from the .img project file.",
                          f"{val('manual')}: timing from {o('--transition')}, "
                          f"{o('--still')}, {o('--drift')}.",
                          choices=("auto", "manual"), default=val("auto")))
    g.add_argument("-t", "--transition", type=float, metavar="SEC",
                   help=H("Manual mode: time from the start of a slide to",
                          "the start of its still (the transition length).",
                          f"First target: {o('--transition')} + {o('--still')} / 2;",
                          f"then every {o('--transition')} + {o('--still')} + "
                          f"{o('--drift')}."))
    g.add_argument("-s", "--still", type=float, metavar="SEC",
                   help=H("Manual mode: still length of each slide, in seconds."))
    g.add_argument("--drift", type=float, default=0.0, metavar="SEC",
                   help=H("Manual mode: correction added to every slide step,",
                          f"in seconds (e.g. {val('0.01')} or {val('-0.01')}).",
                          default=val("0")))
    g.add_argument("--frame-pick", choices=("keyframe", "center"), default="keyframe",
                   help=H(f"{val('keyframe')}: the I-frame inside the safe still window",
                          "          (best-coded frame), else the window centre.",
                          f"{val('center')}:   always the window centre.",
                          choices=("keyframe", "center"), default=val("keyframe")))
    g.add_argument("--margin", type=float, default=0.5, metavar="SEC",
                   help=H("Minimum distance in seconds between a target frame",
                          "and any transition.",
                          default=val("0.5")))

    g = p.add_argument_group("Checks")
    g.add_argument("--dryRun", "--dry-run", dest="dry_run", action="store_true",
                   help=H("Plan and check only; nothing is written."))
    g.add_argument("--verify", action="store_true",
                   help=H(f"With {o('--dryRun')}: also decode and run the stillness",
                          "check (always done when saving)."))
    g.add_argument("--min-confidence", type=int, default=90, metavar="N",
                   help=H("Minimum confidence score required to save.",
                          choices=(f"{val('0')}-{val('100')}",), default=val("90")))

    g = p.add_argument_group("Hardware")
    g.add_argument("--cpu", action="store_true",
                   help=H("Decode and resize on the CPU instead of the GPU.",
                          default="GPU (NVDEC decode, cv2.cuda resize)"))

    g = p.add_argument_group("Console and logging")
    q = g.add_mutually_exclusive_group()
    q.add_argument("-q", "--quiet", action="count", default=0,
                   help=H(f"{o('-q')}:  one summary line per video.",
                          f"{o('-qq')}: silent; exit code only.",
                          default="per-video checks and verdict"))
    q.add_argument("-v", "--verbose", action="count", default=0,
                   help=H("Full detail, including the per-slide plan table.",
                          default="per-video checks and verdict"))
    g.add_argument("--logfile", nargs="?", const=DEFAULT_LOGFILE, metavar="FILE",
                   help=H(f"Append full ({o('-v')}) detail to {v('FILE')}, regardless",
                          f"of {o('-q')} / {o('-v')}.",
                          default=f"{val(DEFAULT_LOGFILE)} when {v('FILE')} is omitted"))
    g.add_argument("--color", choices=("auto", "always", "never"), default="auto",
                   help=H(f"Console colour; {val('auto')} colours terminals only and",
                          f"honours {val('NO_COLOR')} / {val('FORCE_COLOR')}.",
                          choices=("auto", "always", "never"), default=val("auto")))
    return p


def validate_args(p: argparse.ArgumentParser, args) -> None:
    if args.mode == "auto" and args.project_dir is None:
        p.error("--project-dir is required in auto mode")
    if args.project_dir is not None and not args.project_dir.is_dir():
        p.error(f"project dir not found: {args.project_dir}")
    if args.mode == "manual":
        if args.transition is None or args.still is None:
            p.error("manual mode requires --transition and --still")
        if args.still <= 0 or args.transition < 0 or args.transition + args.still + args.drift <= 0:
            p.error("manual timing must give a positive slide period")
    elif args.transition is not None or args.still is not None or args.drift:
        p.error("--transition/--still/--drift are only valid with --mode manual")
    if not args.dry_run and args.output_base is None:
        p.error("--output_base is required unless --dryRun")
    if args.margin < 0:
        p.error("--margin must be >= 0")
    if not 0 <= args.min_confidence <= 100:
        p.error("--min-confidence must be 0-100")


def main() -> int:
    color_mode = color_mode_from_argv(sys.argv[1:])
    Color.setup(color_mode)
    parser = build_parser(color_mode)
    args = parser.parse_args()
    Color.setup(args.color)
    validate_args(parser, args)
    level = max(QUIETEST, min(VERBOSE, NORMAL + args.verbose - args.quiet))

    log, logpath = None, None
    if args.logfile:
        logpath = Path(os.path.expanduser(args.logfile))
        try:
            log = open(logpath, "a", encoding="utf-8")
        except OSError as e:
            print(f"{paint('ERROR:', 'red', 'bold', err=True)} cannot open log file "
                  f"'{logpath}': {e.strerror}", file=sys.stderr)
            return EXIT_ERROR

    out = Output(level, log)
    started = time.monotonic()
    results: list[Result] = []
    interrupted = False
    with log if log else contextlib.nullcontext():
        out.to_log("=" * 78)
        out.to_log(f"extractStills {__version__}  {datetime.now():%Y-%m-%d %H:%M:%S}  "
                   f"cwd={os.getcwd()}")
        out.to_log(f"command: {shlex.join(sys.argv)}")

        entries = resolve_videos(args.video)
        width = max(len(e.label) for e in entries)
        try:
            for i, entry in enumerate(entries, 1):
                position = f"[{i}/{len(entries)}]" if len(entries) > 1 else ""
                if entry.error:
                    out(f"\n{paint('ERROR:', 'red', 'bold', err=True)} {entry.label}: "
                        f"{entry.error}", NORMAL, err=True)
                    r = Result(entry.label).fail(entry.code, entry.error)
                else:
                    try:
                        r = run_video(entry.path, args, out, position)
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except Exception as e:           # unexpected: report, keep going
                        out(traceback.format_exc().rstrip(), VERBOSE, err=True)
                        out(f"\n{paint('ERROR:', 'red', 'bold', err=True)} {entry.label}: "
                            f"unexpected {type(e).__name__}: {e}", NORMAL, err=True)
                        r = Result(entry.label).fail(
                            EXIT_ERROR, first_line(f"unexpected {type(e).__name__}: {e}"))
                results.append(r)
                out(summary_line(r, width), SUMMARY, exact=True)
        except KeyboardInterrupt:
            interrupted = True
            out(f"\n{paint('Interrupted.', 'yellow', 'bold', err=True)}", SUMMARY, err=True)

        code = batch_exit_code(results, interrupted)
        if len(entries) > 1 or interrupted:
            print_batch_summary(out, results, len(entries), time.monotonic() - started,
                                code, logpath, interrupted)
        out.to_log(f"exit {code}  ({fmt_elapsed(time.monotonic() - started)})")
    return code


if __name__ == "__main__":
    sys.exit(main())
