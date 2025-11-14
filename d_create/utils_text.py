"""文本辅助函数。"""

from __future__ import annotations

import re


def normalize_whitespace(text: str) -> str:
    """移除多余空白并保持段落分隔。"""
    if not text:
        return ""
    # 保留段落的空行，先临时替换
    placeholder = "\uFFFF"
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n" + placeholder, text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = text.replace(placeholder, "\n")
    return text.strip()


def sanitize_for_prompt(text: str, limit: int = 2000) -> str:
    """限制文本长度，避免超出模型上下文。"""
    text = normalize_whitespace(text)
    if len(text) <= limit:
        return text
    return text[:limit] + "..."
