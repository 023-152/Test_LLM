"""问答生成主流程。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from tqdm import tqdm

from .chunking import chunk_text
from .ingest import LoadedDocument, load_documents
from .llm import BaseProvider
from .utils_text import normalize_whitespace


@dataclass
class QAPair:
    input: str
    ground_truth: str


class QAGenerator:
    def __init__(self, provider: BaseProvider) -> None:
        self.provider = provider

    def generate_from_paths(
        self,
        paths: Sequence[str | Path],
        chunk_size: int,
        overlap: int,
        per_chunk: int,
        limit: int | None = None,
    ) -> List[QAPair]:
        documents = load_documents(paths)
        return self.generate_from_documents(
            documents, chunk_size=chunk_size, overlap=overlap, per_chunk=per_chunk, limit=limit
        )

    def generate_from_documents(
        self,
        documents: Sequence[LoadedDocument],
        chunk_size: int,
        overlap: int,
        per_chunk: int,
        limit: int | None = None,
    ) -> List[QAPair]:
        pairs: List[QAPair] = []
        progress = tqdm(documents, desc=f"Provider: {self.provider.name()}", unit="doc")
        for doc in progress:
            chunks = chunk_text(doc.text, chunk_size=chunk_size, overlap=overlap)
            for chunk in chunks:
                raw_pairs = self.provider.generate(chunk, per_chunk)
                for raw in raw_pairs:
                    question = normalize_whitespace(raw.get("input", ""))
                    answer = normalize_whitespace(raw.get("ground_truth", ""))
                    if question and answer:
                        pairs.append(QAPair(input=question, ground_truth=answer))
                        if limit and len(pairs) >= limit:
                            progress.set_postfix_str("达到数量上限")
                            return pairs
        return pairs


def export_to_csv(pairs: Iterable[QAPair], output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["input", "ground_truth"])
        for pair in pairs:
            writer.writerow([pair.input, pair.ground_truth])
    return path
