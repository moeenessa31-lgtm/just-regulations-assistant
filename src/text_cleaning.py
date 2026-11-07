"""Clean the Arabic text extracted from the PDFs before training.

pdftotext and OCR leave three kinds of noise that hurt training:
1. Invisible direction marks (RTL/LTR embeddings) around numbers and brackets.
2. Arabic "presentation forms": the same letter stored as a different character
   depending on its position in the word. NFKC normalization maps them back to
   the normal letters, so the model sees one character per letter.
3. Empty lines, page-number-only lines and repeated spaces.
"""
import re
import unicodedata

# U+200E/200F marks, U+202A-202E embeddings, U+2066-2069 isolates, BOM, zero-width chars
_INVISIBLE = re.compile("[​-‏‪-‮⁦-⁩﻿]")
_TATWEEL = "ـ"  # decorative stretching of Arabic letters (ـــ)
_PAGE_NUMBER_LINE = re.compile(r"^\s*[\d٠-٩]{1,3}\s*$")
_SPACES = re.compile(r"[ \t ]+")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE.sub("", text)
    text = text.replace(_TATWEEL, "")
    lines = []
    for line in text.splitlines():
        line = _SPACES.sub(" ", line).strip()
        if not line or _PAGE_NUMBER_LINE.match(line):
            continue
        lines.append(line)
    return "\n".join(lines)
