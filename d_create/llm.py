"""LLM 提供方实现"""

from __future__ import annotations

import json
import os
import re
import textwrap
import time
from abc import ABC, abstractmethod
from typing import List

try:  # pragma: no cover - 仅在缺少 openai 库时触发
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

try:  # pragma: no cover - 仅在缺少 dashscope 库时触发
    import dashscope
    from dashscope import Generation
except ImportError:  # pragma: no cover
    dashscope = None  # type: ignore
    Generation = None  # type: ignore

from .utils_text import sanitize_for_prompt


class BaseProvider(ABC):
    """统一的问答生成接口"""

    @abstractmethod
    def name(self) -> str:  # noqa: D401
        """返回提供方名称"""

    @abstractmethod
    def generate(self, chunk: str, max_questions: int) -> List[dict]:
        """输入文本片段，返回问答对列表"""


class OpenAIProvider(BaseProvider):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.2,
        max_retries: int | None = None,
        retry_initial_delay: float | None = None,
        backoff_factor: float | None = None,
        retry_max_delay: float | None = None,
        ratelimit_min_interval: float | None = None,
    ) -> None:
        if OpenAI is None:
            raise RuntimeError("未安装 openai 库，请先 pip install openai")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("缺少 OPENAI_API_KEY，无法调用 OpenAI 模型")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_retries = (
            max_retries if max_retries is not None else int(os.getenv("OPENAI_MAX_RETRIES", "5"))
        )
        self.retry_initial_delay = (
            retry_initial_delay
            if retry_initial_delay is not None
            else float(os.getenv("OPENAI_RETRY_INITIAL_DELAY", "20"))
        )
        self.backoff_factor = (
            backoff_factor if backoff_factor is not None else float(os.getenv("OPENAI_BACKOFF_FACTOR", "2"))
        )
        self.retry_max_delay = (
            retry_max_delay if retry_max_delay is not None else float(os.getenv("OPENAI_RETRY_MAX_DELAY", "60"))
        )
        self.min_interval = (
            ratelimit_min_interval
            if ratelimit_min_interval is not None
            else float(os.getenv("OPENAI_RATELIMIT_MIN_INTERVAL", "0"))
        )
        self._last_request_ts: float = 0.0

    def name(self) -> str:
        return "openai"

    def generate(self, chunk: str, max_questions: int) -> List[dict]:
        def _extract_retry_after_seconds(msg: str) -> float:
            m = re.search(r"try again in (\d+)s", msg)
            return float(m.group(1)) if m else 0.0

        def _maybe_sleep_to_throttle() -> None:
            if self.min_interval <= 0 or self._last_request_ts <= 0:
                return
            elapsed = time.monotonic() - self._last_request_ts
            remaining = self.min_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

        prompt = _build_prompt(chunk, max_questions)
        attempt = 0
        while True:
            _maybe_sleep_to_throttle()
            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                self._last_request_ts = time.monotonic()
                return _parse_json_pairs(response.output_text)
            except Exception as exc:  # noqa: BLE001 - 仅针对 429 做处理
                is_rate_limit = exc.__class__.__name__ == "RateLimitError"
                if is_rate_limit and attempt < self.max_retries:
                    hinted = _extract_retry_after_seconds(str(exc))
                    base = max(self.retry_initial_delay, self.min_interval, hinted)
                    wait = min(self.retry_max_delay, base * (self.backoff_factor**attempt))
                    time.sleep(wait)
                    attempt += 1
                    continue
                raise


class QwenProvider(BaseProvider):
    """接入通义千问（DashScope）的实现"""

    def __init__(
        self,
        model: str = os.getenv("QWEN_MODEL", "qwen-plus"),
        api_key: str | None = None,
        temperature: float = 0.2,
    ) -> None:
        if Generation is None or dashscope is None:
            raise RuntimeError("未安装 dashscope 库，请先 pip install dashscope")
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        if not self.api_key:
            raise RuntimeError("缺少 DASHSCOPE_API_KEY（或 QWEN_API_KEY），无法调用千问模型")
        dashscope.api_key = self.api_key
        self._generation = Generation
        self.model = model
        self.temperature = temperature

    def name(self) -> str:
        return "qwen"

    def generate(self, chunk: str, max_questions: int) -> List[dict]:
        prompt = _build_prompt(chunk, max_questions)
        response = self._generation.call(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            result_format="json",
            temperature=self.temperature,
        )
        text = _extract_text_from_dashscope(response)
        if not text:
            raise RuntimeError("千问接口返回内容为空，无法解析")
        return _parse_json_pairs(text)


class DummyProvider(BaseProvider):
    """离线占位实现，便于无 Key 环境测试"""

    def name(self) -> str:
        return "dummy"

    def generate(self, chunk: str, max_questions: int) -> List[dict]:
        chunk = sanitize_for_prompt(chunk, 600)
        summary = chunk[:200].replace("\n", " ")
        pairs: List[dict] = []
        for idx in range(max_questions):
            question = f"根据文档内容，第 {idx + 1} 条要点是什么？"
            answer = f"第 {idx + 1} 条要点概述：{summary}"
            pairs.append({"input": question, "ground_truth": answer})
        return pairs


def create_provider(name: str, **kwargs) -> BaseProvider:
    name = name.lower()
    if name == "openai":
        return OpenAIProvider(**kwargs)
    if name == "qwen":
        return QwenProvider(**kwargs)
    if name == "dummy":
        return DummyProvider()
    raise ValueError(f"未知 provider: {name}")


def _build_prompt(chunk: str, max_questions: int) -> str:
    instructions = textwrap.dedent(
        f"""
        你是资深数据标注员，请基于以下文档内容生成高质量问答对：
        - 输出 JSON 数组，每个元素包含 input 和 ground_truth 字段；
        - 所有回答用中文；
        - 问题要具体、能够通过给定文本回答；
        - 至多生成 {max_questions} 条；
        文档内容：
        {sanitize_for_prompt(chunk, 3200)}
        """
    ).strip()
    return instructions


def _extract_text_from_dashscope(response) -> str | None:
    """兼容 DashScope 各版本 response 结构，避免 KeyError"""
    text = None
    try:
        text = response.output_text
    except (AttributeError, KeyError):
        text = None
    if text:
        return text
    output = None
    try:
        output = response.output
    except (AttributeError, KeyError):
        output = None
    if isinstance(output, dict):
        text = output.get("text")
        if text:
            return text
        choices = output.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") or {}
            text = message.get("content")
            if text:
                return text
    messages = getattr(response, "messages", None)
    if isinstance(messages, list) and messages:
        last = messages[-1]
        if isinstance(last, dict):
            text = last.get("content")
    return text


def _parse_json_pairs(raw: str) -> List[dict]:
    """尝试解析 JSON 数组，自动清理围栏符号"""
    candidate = raw.strip()
    if candidate.startswith("```"):
        candidate = "\n".join(line for line in candidate.splitlines() if not line.startswith("```"))
    start = candidate.find("[")
    end = candidate.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = candidate[start : end + 1]
    try:
        data = json.loads(candidate)
        if isinstance(data, list):
            return [
                {
                    "input": (item.get("input") or "").strip(),
                    "ground_truth": (item.get("ground_truth") or "").strip(),
                }
                for item in data
                if isinstance(item, dict)
            ]
    except json.JSONDecodeError:
        pass
    return []
