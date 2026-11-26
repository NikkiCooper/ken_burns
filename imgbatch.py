#  imgbatch.py Copyright (c) 2026 Nikki Cooper
#
#  This program and the accompanying materials are made available under the
#  terms of the GNU Lesser General Public License, version 3.0 which is available at
#  https://www.gnu.org/licenses/gpl-3.0.html#license-text
#
"""Resolve ``--img_path`` folders and their output video names.

Folder names
------------
Extraction folders are named ``<video_basename><ENDING>``:

    Bobbie-Slideshow-077_AI_Extracted            (extractStills.py)
    Cristina-Slideshow-001_AI_extracted_frames   (getFrames.py)

``<video_basename>`` is ``<model>-Slideshow-<set>[_SUFFIX]``.  The model name may
contain spaces; SUFFIX may not.

Output names (``--auto_filename``)
----------------------------------
The output is ``<model>-Slideshow-<set>_<--suffix>.mp4``: ``--suffix`` replaces the
folder's SUFFIX, or is appended when there is none.  A leading ``_`` on
``--suffix`` is optional (``AI_Kenburns`` and ``_AI_Kenburns`` are the same).

``--suffix`` is rejected when it

* contains whitespace or a path separator,
* is empty or ``AI`` - by convention a video with no suffix, or with ``_AI``,
  was *not* made by ken_burns, so ken_burns may never produce one,
* equals the folder's own SUFFIX (case-insensitive) - the result would carry the
  name of the very video the images were extracted from.

An output file that already exists is always an error; nothing is overwritten.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import globrange

EXTRACT_ENDINGS = ("_Extracted", "_extracted_frames")
NAME_RE = re.compile(r"^(?P<model>.+?)-Slideshow-(?P<set>\d+)(?P<rest>.*)$")
NON_KB_SUFFIXES = {"", "ai"}          # case-insensitive; never produced by ken_burns


class NamingError(ValueError):
    """A folder name or --suffix that breaks the naming rules."""


def normalize_suffix(suffix: str) -> str:
    """Validate --suffix and return it without leading underscores."""
    if any(c.isspace() for c in suffix):
        raise NamingError(f"--suffix '{suffix}' may not contain spaces")
    if "/" in suffix or "\\" in suffix:
        raise NamingError(f"--suffix '{suffix}' may not contain a path separator")
    core = suffix.lstrip("_")
    if core.lower() in NON_KB_SUFFIXES:
        shown = "''" if not core else f"'{suffix}'"
        raise NamingError(f"--suffix {shown} is not allowed: a video with "
                         f"{'no suffix' if not core else 'the _AI suffix'} is, by convention, "
                         f"not a ken_burns video")
    return core


def parse_folder(folder: Path) -> tuple[str, str]:
    """Return (``<model>-Slideshow-<set>``, existing SUFFIX without '_' or '')."""
    name = folder.name
    base = next((name[:-len(e)] for e in EXTRACT_ENDINGS if name.endswith(e)), None)
    if base is None:
        raise NamingError(f"folder '{name}' does not end in "
                         f"{' or '.join(EXTRACT_ENDINGS)}; cannot derive a video name")
    m = NAME_RE.match(base)
    if not m:
        raise NamingError(f"folder '{name}' is not named "
                         f"<model>-Slideshow-<set>[_SUFFIX]<ending>")
    rest = m["rest"]
    if rest and not rest.startswith("_"):
        raise NamingError(f"folder '{name}': text after the set number must start with '_'")
    existing = rest.lstrip("_")
    if any(c.isspace() for c in existing):
        raise NamingError(f"folder '{name}': SUFFIX '{existing}' contains spaces")
    return f"{m['model']}-Slideshow-{m['set']}", existing


def derive_filename(folder: Path, suffix_core: str) -> str:
    stem, existing = parse_folder(folder)
    if existing.lower() == suffix_core.lower():
        raise NamingError(f"--suffix '_{suffix_core}' is the same as the folder's SUFFIX "
                         f"'_{existing}': the output would carry the source video's name")
    return f"{stem}_{suffix_core}.mp4"


@dataclass
class Job:
    label: str                       # folder name, or the spec when unresolved
    img_dir: Path | None = None
    output: Path | None = None
    error: str = ""


@dataclass
class Plan:
    jobs: list[Job]
    fatal: str = ""                  # stops everything (bad --output, batch without names)

    @property
    def batch(self) -> bool:
        return len(self.jobs) > 1


def resolve_dirs(specs: list[str]) -> list[Job]:
    """--img_path specs -> jobs with img_dir (or error); no output names yet."""
    return [Job(m.label, img_dir=m.path, error=m.error)
            for m in globrange.expand(specs, "dir", noun="image folder")]


def plan_render(specs: list[str], output: str | None, auto_filename: bool,
                suffix: str | None) -> Plan:
    """Resolve folders and output paths for rendering."""
    jobs = resolve_dirs(specs)
    plan = Plan(jobs)

    if suffix is None:
        plan.fatal = "--suffix is required for rendering"
        return plan
    try:
        suffix_core = normalize_suffix(suffix)
    except NamingError as e:
        plan.fatal = str(e)
        return plan
    if output is None:
        plan.fatal = "--output is required for rendering"
        return plan
    out = Path(os.path.expanduser(output))

    if plan.batch and not auto_filename:
        plan.fatal = (f"{len(jobs)} image folders given: batch processing needs "
                      f"--auto_filename to name each output video")
        return plan

    if auto_filename:
        if not out.is_dir():
            plan.fatal = (f"with --auto_filename, --output must be an existing directory: "
                          f"{out}")
            return plan
        claimed: dict[Path, str] = {}
        for job in jobs:
            if job.error:
                continue
            try:
                job.output = out / derive_filename(job.img_dir, suffix_core)
            except NamingError as e:
                job.error = str(e)
                continue
            if job.output in claimed:
                job.error = (f"output {job.output.name} is also produced by "
                             f"'{claimed[job.output]}'")
            else:
                claimed[job.output] = job.label
    else:
        if out.is_dir():
            plan.fatal = (f"--output is a directory: {out}\n"
                          f"Give a file name, or add --auto_filename to derive one")
            return plan
        if not out.parent.is_dir():
            plan.fatal = f"output directory does not exist: {out.parent}"
            return plan
        for job in jobs:
            if not job.error:
                job.output = out

    for job in jobs:                       # number one rule: never overwrite
        if not job.error and job.output.exists():
            job.error = f"output already exists: {job.output}"
    return plan
