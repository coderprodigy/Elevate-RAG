"""Text normalisation and content-derived IDs."""

from __future__ import annotations

import re
import unicodedata

from hgrag.core.hashing import stable_hash

_SPACES = re.compile(r"[ \t\f\v]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_NON_WORD = re.compile(r"[^\w\s]")
_ANY_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """NFC-normalise, collapse runs of spaces, keep paragraph breaks (at most one blank line)."""
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = (_SPACES.sub(" ", line).strip() for line in text.split("\n"))
    return _MANY_NEWLINES.sub("\n\n", "\n".join(lines)).strip()


def normalize_question(question: str) -> str:
    """Lowercase, drop punctuation and collapse whitespace. Used to find duplicate questions."""
    lowered = unicodedata.normalize("NFKC", question).lower()
    return _ANY_WHITESPACE.sub(" ", _NON_WORD.sub(" ", lowered)).strip()


def doc_id_for(title: str, text: str) -> str:
    """Stable ID from content, so the same passage gets the same ID in every dataset build."""
    return "d_" + stable_hash({"title": title, "text": text})[:16]


def qid_for(prefix: str, question: str) -> str:
    """Stable query ID for sources that have no native question ID."""
    return f"{prefix}-{stable_hash(question)[:12]}"
