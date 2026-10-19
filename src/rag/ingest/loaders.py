"""Read source documents into plain text plus the metadata needed for citations."""

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RawPage:
    doc: str  # file name, e.g. "guide.pdf"
    page: int | None  # 1-based page number; None for formats without pages
    text: str


def file_sha256(path: Path) -> str:
    """Hash of the file content, used to skip documents that were already indexed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_document(path: Path) -> list[RawPage]:
    """Return one RawPage per PDF page, or a single RawPage for .md / .txt files.

    TODO(v0.1):
      - .pdf  -> pypdf, one RawPage per page (keep the page number for citations)
      - .md / .txt -> read as UTF-8
      - raise a clear error for unsupported extensions
    """
    raise NotImplementedError("load_document is the first function to implement")
