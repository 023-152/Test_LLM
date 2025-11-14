"""文本切分逻辑。"""

from __future__ import annotations

from typing import Iterable, List

from .utils_text import normalize_whitespace


def _split_paragraphs(text: str) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs or [text.strip()]


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> Iterable[str]:
    """将长文本切分为带重叠的片段。"""
    if chunk_size <= overlap:
        raise ValueError("chunk_size 必须大于 overlap")

    clean_text = normalize_whitespace(text)
    paragraphs = _split_paragraphs(clean_text)

    chunks: List[str] = []
    buffer: List[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len + 1 <= chunk_size:
            buffer.append(para)
            current_len += para_len + 1
        else:
            if buffer:
                chunks.append("\n\n".join(buffer))
            # 重叠部分
            overlap_text = "\n\n".join(buffer)[-overlap:] if overlap and buffer else ""
            buffer = [overlap_text] if overlap_text else []
            buffer.append(para)
            current_len = sum(len(x) + 2 for x in buffer)

    if buffer:
        chunks.append("\n\n".join(buffer))

    return [chunk.strip() for chunk in chunks if chunk.strip()]
