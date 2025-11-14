"""命令行入口"""

from __future__ import annotations

import argparse
import os
from typing import Sequence

from .llm import create_provider
from .qa_generator import QAGenerator, export_to_csv


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取文件并生成问答对数据集 (input, ground_truth)",
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help="待处理的文件路径（支持 PDF/DOCX/TXT/MD）",
    )
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "qwen", "dummy"],
        help="选择 LLM 提供方，默认 openai",
    )
    parser.add_argument(
        "--output",
        default="dataset.csv",
        help="输出 CSV 文件路径，默认 dataset.csv",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=900,
        help="文本分块长度，默认 900",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="分块重叠长度，默认 200",
    )
    parser.add_argument(
        "--per-chunk",
        type=int,
        default=2,
        help="每个文本块的问答对数量上限，默认 2",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="总体问答对数量上限，默认无限制",
    )
    parser.add_argument(
        "--openai-model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI 模型名称，默认 gpt-4o-mini",
    )
    parser.add_argument(
        "--qwen-model",
        default=os.getenv("QWEN_MODEL", "qwen-plus"),
        help="通义千问模型名称，默认 qwen-plus",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="采样温度，默认 0.2",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    provider_kwargs = {}
    if args.provider == "openai":
        provider_kwargs = {"model": args.openai_model, "temperature": args.temperature}
    elif args.provider == "qwen":
        provider_kwargs = {"model": args.qwen_model, "temperature": args.temperature}
    provider = create_provider(args.provider, **provider_kwargs)
    generator = QAGenerator(provider)
    pairs = generator.generate_from_paths(
        paths=args.paths,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
        per_chunk=args.per_chunk,
        limit=args.limit,
    )
    output_path = export_to_csv(pairs, args.output)
    print(f"生成完成，共 {len(pairs)} 条，文件已写入：{output_path}")


if __name__ == "__main__":
    main()

