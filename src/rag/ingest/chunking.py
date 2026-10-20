"""Split pages into overlapping chunks. This is where most retrieval quality is won or lost."""

import re

import tiktoken

from rag.ingest.loaders import RawPage
from rag.models import Chunk

_ENC = tiktoken.get_encoding("cl100k_base")
_SEP = _ENC.encode("\n\n")
_HEADING_TOKENS = 12  # un párrafo más corto que esto y sin punto final se trata como título

# funcion auxiliar para contar tokens
def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))

    

def chunk_pages(
    pages: list[RawPage], max_tokens: int, overlap: int, doc_hash: str = ""
) -> list[Chunk]:
    """Turn pages into chunks of at most ``max_tokens`` tokens with ``overlap`` shared tokens.
    
        (v0.1):
          - count tokens with tiktoken (an approximation of the embedding model's tokenizer)
          - split on paragraph boundaries first, and only cut inside a paragraph if it is too long
          - build ids like ``{doc}::p{page}::c{index}`` and fill n_tokens and doc_hash
          - read the chunks by eye afterwards: are sentences cut? are headings lost?
    
        The invariants in tests/test_chunking.py describe what "correct" means.
    """
    if overlap >= max_tokens:
        raise ValueError("overlap must be smaller than max_tokens")
    step = max_tokens - overlap

    chunks: list[Chunk] = []
    index = 0  # contador global: único dentro del documento

    for page in pages:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page.text) if p.strip()]

        # Fase A: pegar los títulos cortos al párrafo siguiente.
        merged: list[str] = []
        carry = ""
        for para in paragraphs:
            if carry:
                para = carry + "\n" + para
                carry = ""
            if count_tokens(para) < _HEADING_TOKENS and not para.endswith((".", ":", "?", "!")):
                carry = para
            else:
                merged.append(para)
        if carry:
            merged.append(carry)

        # Fases B y C: empaquetar párrafos y cortar los demasiado largos.
        ids_per_chunk: list[list[int]] = []
        cur: list[int] = []

        for para in merged:
            ids = _ENC.encode(para)

            if len(ids) > max_tokens:
                if cur:
                    ids_per_chunk.append(cur)
                windows, start = [], 0
                while True:
                    windows.append(ids[start:start + max_tokens])
                    if start + max_tokens >= len(ids):
                        break
                    start += step
                ids_per_chunk.extend(windows[:-1])
                cur = windows[-1]
                continue

            sep = _SEP if cur else []
            if len(cur) + len(sep) + len(ids) <= max_tokens:
                cur = cur + sep + ids
            else:
                ids_per_chunk.append(cur)
                tail = cur[-overlap:] if overlap else []
                tail = tail[: max(0, max_tokens - len(ids) - len(_SEP))]
                cur = tail + (_SEP if tail else []) + ids

        if cur:
            ids_per_chunk.append(cur)

        # Construir los Chunk.
        page_no = page.page if page.page is not None else 0
        for ids in ids_per_chunk:
            text = _ENC.decode(ids).strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    id=f"{page.doc}::p{page_no}::c{index:03d}",
                    doc=page.doc,
                    page=page.page,
                    text=text,
                    n_tokens=count_tokens(text),
                    doc_hash=doc_hash,
                )
            )
            index += 1

    return chunks


r"""VERSION A COMPARAR AL FINAL
import re

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")
_SEP = _ENC.encode("\n\n")
_SPACE = _ENC.encode(" ")
_HEADING_TOKENS = 12
_SENTENCE_END = re.compile(r"(?<=[.;:?!])\s+")


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def _join(units: list[tuple[list[int], list[int]]]) -> list[int]:
    #Concatenate units (ids, separator_before) into one token list.#
    out: list[int] = []
    for i, (ids, sep) in enumerate(units):
        if i:
            out += sep
        out += ids
    return out


def chunk_pages_v1_1(
    pages: list[RawPage], max_tokens: int, overlap: int, doc_hash: str = ""
) -> list[Chunk]:
    #Sentence-aware version of ``chunk_pages``.

    #Paragraphs are packed first. A paragraph longer than ``max_tokens`` is split into
    #sentences, and those are packed whole. Overlap is made of whole sentences (or
    #paragraphs) that fit in ``overlap`` tokens, so chunks start at a sentence boundary.
    #Only a single sentence longer than ``max_tokens`` falls back to a sliding window.
    
    if overlap >= max_tokens:
        raise ValueError("overlap must be smaller than max_tokens")
    step = max_tokens - overlap

    chunks: list[Chunk] = []
    index = 0

    for page in pages:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page.text) if p.strip()]

        # Fase A: pegar los títulos cortos al párrafo siguiente.
        merged: list[str] = []
        carry = ""
        for para in paragraphs:
            if carry:
                para = carry + "\n" + para
                carry = ""
            if count_tokens(para) <= _HEADING_TOKENS and not para.endswith(
                (".", ":", "?", "!")
            ):
                carry = para
            else:
                merged.append(para)
        if carry:
            merged.append(carry)

        # Fase B: convertir a unidades (ids, separador previo), cada una <= max_tokens.
        units: list[tuple[list[int], list[int]]] = []
        for para in merged:
            ids = _ENC.encode(para)
            if len(ids) <= max_tokens:
                units.append((ids, _SEP))
                continue
            first = True
            for sentence in _SENTENCE_END.split(para):
                sentence = sentence.strip()
                if not sentence:
                    continue
                s_ids = _ENC.encode(sentence)
                sep = _SEP if first else _SPACE
                first = False
                if len(s_ids) <= max_tokens:
                    units.append((s_ids, sep))
                    continue
                # Frase gigante: ventana deslizante como último recurso.
                start, k = 0, 0
                while True:
                    units.append((s_ids[start:start + max_tokens], sep if k == 0 else _SPACE))
                    if start + max_tokens >= len(s_ids):
                        break
                    start += step
                    k += 1

        # Fase C: empaquetar unidades; el solapamiento son unidades completas.
        ids_per_chunk: list[list[int]] = []
        cur: list[tuple[list[int], list[int]]] = []
        for unit in units:
            if not cur or len(_join(cur + [unit])) <= max_tokens:
                cur = cur + [unit]
                continue
            ids_per_chunk.append(_join(cur))
            tail: list[tuple[list[int], list[int]]] = []
            total = 0
            for u in reversed(cur):
                size = len(u[0]) + len(u[1])
                if total + size > overlap:
                    break
                tail.insert(0, u)
                total += size
            cur = tail + [unit]
            while len(cur) > 1 and len(_join(cur)) > max_tokens:
                cur.pop(0)
        if cur:
            ids_per_chunk.append(_join(cur))

        # Construir los Chunk.
        page_no = page.page if page.page is not None else 0
        for ids in ids_per_chunk:
            text = _ENC.decode(ids).strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    id=f"{page.doc}::p{page_no}::c{index:03d}",
                    doc=page.doc,
                    page=page.page,
                    text=text,
                    n_tokens=count_tokens(text),
                    doc_hash=doc_hash,
                )
            )
            index += 1

    return chunks
"""