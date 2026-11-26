#  globrange.py Copyright (c) 2026 Nikki Cooper
#
#  This program and the accompanying materials are made available under the
#  terms of the GNU Lesser General Public License, version 3.0 which is available at
#  https://www.gnu.org/licenses/gpl-3.0.html#license-text
#
"""Shell-style path patterns with numeric ranges.

Shared by ``ken_burns.py`` (``--img_path``) and ``work_scripts/extractStills.py``
(``--video``) so both tools match paths identically.

Patterns apply to the last path component only:

    *          any run of characters
    ?          exactly one character
    [abc]      one character from the set   ([!abc] / [^abc]: not from the set)
    {N-M}      a whole number (leading zeros allowed) whose value is N..M
    {N-}       same, N..999

A spec without a directory is resolved against the current directory.  Specs
the shell has already expanded arrive as literal paths and are used as-is.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

RANGE_RE = re.compile(r"\{(\d+)-(\d*)\}")
RANGE_OPEN_MAX = 999


class PatternError(ValueError):
    """Malformed pattern (a command-line mistake, not a missing file)."""


def is_pattern(text: str) -> bool:
    return any(c in text for c in "*?[") or bool(RANGE_RE.search(text))


def pattern_regex(pattern: str) -> re.Pattern:
    """Compile a single-component shell-style pattern (see module docstring)."""
    parts, i = [], 0
    while i < len(pattern):
        m = RANGE_RE.match(pattern, i)
        if m:
            lo, hi = int(m[1]), int(m[2]) if m[2] else RANGE_OPEN_MAX
            if lo > hi:
                raise PatternError(f"empty numeric range '{m[0]}' in '{pattern}'")
            alts = "|".join(str(n) for n in range(hi, lo - 1, -1))
            parts.append(rf"(?<!\d)0*(?:{alts})(?!\d)")
            i = m.end()
            continue
        c = pattern[i]
        if c == "*":
            parts.append(".*")
        elif c == "?":
            parts.append(".")
        elif c == "[" and (j := pattern.find("]", i + 2)) != -1:
            body = pattern[i + 1:j]
            if body[0] in "!^":
                body = "^" + body[1:]
            parts.append("[" + body.replace("\\", "\\\\") + "]")
            i = j + 1
            continue
        else:
            parts.append(re.escape(c))
        i += 1
    return re.compile("".join(parts) + r"\Z")


@dataclass
class Match:
    """One resolved path, or the reason a spec could not be resolved."""
    spec: str                  # the argument as given
    path: Path | None = None
    error: str = ""
    usage: bool = False        # True when the error is a malformed argument

    @property
    def label(self) -> str:
        return self.path.name if self.path else self.spec


def expand(specs: list[str], kind: str, extensions: tuple[str, ...] = (),
           noun: str = "") -> list[Match]:
    """Resolve literal paths and patterns.

    kind        "file" or "dir": what a spec must resolve to.
    extensions  for patterns only: accepted suffixes, e.g. (".mp4",); empty = any.
    noun        used in messages, e.g. "video" -> "video not found".

    Results keep argument order (pattern matches sorted by name); duplicates are
    dropped.  Distinct symlinks to the same target stay distinct.
    """
    if kind not in ("file", "dir"):
        raise ValueError(f"kind must be 'file' or 'dir', not {kind!r}")
    noun = noun or kind
    is_kind = Path.is_dir if kind == "dir" else Path.is_file
    exts = tuple(e.lower() for e in extensions)
    what = (" / ".join(f"{e} files" for e in exts) if exts and kind == "file"
            else "directories" if kind == "dir" else "files")
    results: list[Match] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        key = os.path.abspath(path)
        if key not in seen:
            seen.add(key)
            results.append(Match(str(path), path))

    for spec in specs:
        path = Path(os.path.expanduser(spec))
        if is_kind(path):
            add(path)
            continue
        if path.exists():
            results.append(Match(spec, error=f"not a {'directory' if kind == 'dir' else 'file'}"))
            continue
        if not is_pattern(path.name):
            results.append(Match(spec, error=f"{noun} not found"))
            continue
        folder = path.parent
        if is_pattern(str(folder)):
            results.append(Match(spec, error="wildcards are only supported in the last "
                                             "path component", usage=True))
            continue
        if not folder.is_dir():
            results.append(Match(spec, error=f"directory not found: {folder}"))
            continue
        try:
            rx = pattern_regex(path.name)
        except PatternError as e:
            results.append(Match(spec, error=str(e), usage=True))
            continue
        hidden_ok = path.name.startswith(".")
        hits = sorted(p for p in folder.iterdir()
                      if rx.match(p.name) and is_kind(p)
                      and (not exts or p.suffix.lower() in exts)
                      and (hidden_ok or not p.name.startswith(".")))
        if not hits:
            results.append(Match(spec, error=f"pattern matched no {what}"))
        for p in hits:
            add(p)
    return results
