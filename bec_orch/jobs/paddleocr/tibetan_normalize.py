"""Canonical Tibetan Unicode normalization.

Vendored verbatim from ``bec-ocr-training/bdrc_ocr/tibetan_normalize.py`` (itself
vendored from botok's ``unicode_normalization`` / ``lenient_normalization``) so
the OCR worker's post-processing matches the training/eval scorer **exactly**.

``normalize_unicode_text`` is the single entry point: it applies
``normalize_unicode`` (NFD unification + canonical combining-mark reorder)
followed by ``normalize_graphical`` (fold graphically-identical codepoints such
as nyis-shad ``༎`` -> ``།།`` and ``0f0c`` -> ``0f0b``).
"""

from __future__ import annotations

import re
from enum import Enum


class Cats(Enum):
    Other = 0
    Base = 1
    Subscript = 2
    BottomVowel = 3
    BottomMark = 4
    TopVowel = 5
    TopMark = 6
    RightMark = 7


CATEGORIES = (
    [Cats.Other]  # 0F00
    + [Cats.Base]  # 0F01
    + [Cats.Other] * 22  # 0F02-0F17
    + [Cats.BottomVowel] * 2  # 0F18-0F19
    + [Cats.Other] * 6  # 0F1A-0F1F
    + [Cats.Base] * 20  # 0F20-0F33
    + [Cats.Other]  # 0F34
    + [Cats.BottomMark]  # 0F35
    + [Cats.Other]  # 0F36
    + [Cats.BottomMark]  # 0F37
    + [Cats.Other]  # 0F38
    + [Cats.Subscript]  # 0F39
    + [Cats.Other] * 4  # 0F3A-0F3D
    + [Cats.RightMark]  # 0F3E
    + [Cats.Other]  # 0F3F
    + [Cats.Base] * 45  # 0F40-0F6C
    + [Cats.Other] * 4  # 0F6D-0F70
    + [Cats.BottomVowel]  # 0F71
    + [Cats.TopVowel]  # 0F72
    + [Cats.TopVowel]  # 0F73
    + [Cats.BottomVowel] * 2  # 0F74-0F75
    + [Cats.TopVowel] * 8  # 0F76-0F7D
    + [Cats.TopMark]  # 0F7E
    + [Cats.RightMark]  # 0F7F
    + [Cats.TopVowel] * 2  # 0F80-0F81
    + [Cats.TopMark] * 2  # 0F82-0F83
    + [Cats.BottomMark]  # 0F84
    + [Cats.Other]  # 0F85
    + [Cats.TopMark] * 2  # 0F86-0F87
    + [Cats.Base] * 2  # 0F88-0F89
    + [Cats.Base]  # 0F8A
    + [Cats.Other]  # 0F8B
    + [Cats.Base]  # 0F8C
    + [Cats.Subscript] * 48  # 0F8D-0FBC
)


def charcat(c: str) -> Cats:
    """Return the category for a single-char string."""
    o = ord(c)
    if 0x0F00 <= o <= 0x0FBC:
        return CATEGORIES[o - 0x0F00]
    return Cats.Other


def unicode_reorder(txt: str):
    charcats = [charcat(c) for c in txt]
    i = 0
    res = []
    valid = True
    while i < len(charcats):
        c = charcats[i]
        if c != Cats.Base:
            if c.value > Cats.Base.value:
                valid = False
            res.append(txt[i])
            i += 1
            continue
        j = i + 1
        while j < len(charcats) and charcats[j].value > Cats.Base.value:
            j += 1
        newindices = sorted(range(i, j), key=lambda e: (charcats[e].value, e))
        res.append("".join(txt[n] for n in newindices))
        i = j
    return "".join(res), valid


def _is_vowel(char: str) -> bool:
    return bool(re.search(r"[\u0f71-\u0f84]", char))


def _is_suffix(char: str) -> bool:
    return bool(re.search(r"[\u0f90-\u0fbc]", char))


def normalize_invalid_start_string(s: str) -> str:
    if len(s) < 2:
        return s
    if _is_vowel(s[0]) and not _is_vowel(s[1]) and not _is_suffix(s[1]):
        return s[1] + s[0] + (s[2:] if len(s) > 2 else "")
    if _is_suffix(s[0]):
        return s[1:]
    return s


def normalize_unicode(s: str, form: str = "nfd") -> str:
    # deprecated or discouraged characters
    s = s.replace("\u0f73", "\u0f71\u0f72")
    s = s.replace("\u0f75", "\u0f71\u0f74")
    s = s.replace("\u0f77", "\u0fb2\u0f71\u0f80")
    s = s.replace("\u0f79", "\u0fb3\u0f71\u0f80")
    s = s.replace("\u0f81", "\u0f71\u0f80")
    if form == "nfd":
        s = s.replace("\u0f43", "\u0f42\u0fb7")
        s = s.replace("\u0f48", "\u0f47\u0fb7")
        s = s.replace("\u0f4d", "\u0f4c\u0fb7")
        s = s.replace("\u0f52", "\u0f51\u0fb7")
        s = s.replace("\u0f57", "\u0f56\u0fb7")
        s = s.replace("\u0f5c", "\u0f5b\u0fb7")
        s = s.replace("\u0f69", "\u0f40\u0fb5")
        s = s.replace("\u0f76", "\u0fb2\u0f80")
        s = s.replace("\u0f78", "\u0fb3\u0f80")
        s = s.replace("\u0f93", "\u0f92\u0fb7")
        s = s.replace("\u0f98", "\u0f97\u0fb7")
        s = s.replace("\u0f9d", "\u0f9c\u0fb7")
        s = s.replace("\u0fa2", "\u0fa1\u0fb7")
        s = s.replace("\u0fa7", "\u0fa6\u0fb7")
        s = s.replace("\u0fac", "\u0fab\u0fb7")
        s = s.replace("\u0fb9", "\u0f90\u0fb5")
    else:
        s = s.replace("\u0f42\u0fb7", "\u0f43")
        s = s.replace("\u0f4c\u0fb7", "\u0f4d")
        s = s.replace("\u0f51\u0fb7", "\u0f52")
        s = s.replace("\u0f56\u0fb7", "\u0f57")
        s = s.replace("\u0f5b\u0fb7", "\u0f5c")
        s = s.replace("\u0f40\u0fb5", "\u0f69")
        s = s.replace("\u0fb2\u0f80", "\u0f76")
        s = s.replace("\u0fb3\u0f80", "\u0f78")
        s = s.replace("\u0f92\u0fb7", "\u0f93")
        s = s.replace("\u0f9c\u0fb7", "\u0f9d")
        s = s.replace("\u0fa1\u0fb7", "\u0fa2")
        s = s.replace("\u0fa6\u0fb7", "\u0fa7")
        s = s.replace("\u0fab\u0fb7", "\u0fac")
        s = s.replace("\u0f90\u0fb5", "\u0fb9")
    # 0f00 is not marked as composed in Unicode (a known spec mistake)
    s = s.replace("\u0f00", "\u0f68\u0f7c\u0f7e")
    s = s.replace("ཅ༹", "ཙ")
    s = s.replace("ཆ༹", "ཚ")
    s = s.replace("ཇ༹", "ཛ")
    s, _valid = unicode_reorder(s)
    # ra -> non-small rago (0f6a -> 0f62) unless it sits above a subjoined letter
    s = re.sub("\u0f6a(?![\u0f90-\u0f97\u0f9a-\u0fac\u0fae\u0faf\u0fb4-\u0fbc])", "ར", s)
    s = normalize_invalid_start_string(s)
    return s


def normalize_graphical(s: str) -> str:
    """Fold codepoints that share the same graphical representation."""
    # no graphical distinction between 0f0c and 0f0b
    s = s.replace("\u0f0c", "\u0f0b")
    # double shad is just two shad
    s = s.replace("\u0f0e", "\u0f0d\u0f0d")
    # 0f38 vs 0f27 is semantic but rarely distinguished graphically
    s = s.replace("\u0f38", "\u0f27")
    s = s.replace("\u0f7a\u0f7a", "\u0f7b")
    s = s.replace("\u0f7c\u0f7c", "\u0f7d")
    s = s.replace("༇", "࿓།࿒།")
    # no 0f71 in the middle of stacks, only 0fb0
    s = re.sub(r"[\u0f71]([\u0f8d-\u0fac\u0fae\u0fb0\u0fb3-\u0fbc])", "\u0fb0\\1", s)
    # no 0fb0 at the end of stacks, only 0f71
    s = re.sub(r"[\u0fb0]([^\u0f8d-\u0fac\u0fae\u0fb0\u0fb3-\u0fbc]|$)", "\u0f71\\1", s)
    return s


def normalize_unicode_text(text: str | None) -> str:
    """Canonical Tibetan Unicode form used across the OCR pipeline.

    Applies ``normalize_unicode`` (NFD + reorder) then ``normalize_graphical``
    so predictions are always in the same normal form used for scoring (e.g.
    nyis-shad ``༎`` folded to ``།།``). Empty/None -> "".
    """
    if not text:
        return ""
    return normalize_graphical(normalize_unicode(text))
