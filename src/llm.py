"""
llm.py: one small chat-client interface over two providers. The provider is a setting.

    client.add_user_text(text)
    turn = client.step()                 # one model call -> Turn (text, tool calls, usage, cost)
    client.add_tool_results(results)     # answer every tool call of the last turn

AnthropicClient: official SDK, prompt caching (automatic + a breakpoint after the system prompt).
No refusal fallback: every call is served by the requested model, and the model the API reports is logged.
OpenRouterClient: OpenAI-compatible chat API over urllib, with retries on 429 and provider errors.
Tools, tool_choice and effort are fixed when the client is created and never change during a run,
so the cached prefix stays valid.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5-5",
    "openrouter": "qwen/qwen3.8-27b:free",
}

# USD per million tokens: (input, cache write 5 min, cache read, output). Used by the cost guard.
PRICES = {
    "claude-sonnet-5-5": (2.00, 2.50, 0.20, 10.00),
    "claude-opus-5-5": (4.00, 5.00, 0.20, 20.00),
    "claude-opus-5": (5.00, 6.25, 0.50, 25.00),
    "claude-opus-4-8": (5.00, 6.25, 0.50, 25.00),
    "claude-fable-5-1": (10.00, 12.50, 0.25, 50.00),
    "claude-haiku-4-5": (1.00, 1.25, 0.10, 5.00),
}

# Models that reject the effort setting (it is an error on Haiku 4.5); they run without it.
NO_EFFORT_PREFIXES = ("claude-haiku-4",)


def effort_options(model: str, effort: str) -> dict:
    """The output_config to send: the effort level, unless the model does not take one."""
    return {} if model.startswith(NO_EFFORT_PREFIXES) else {"output_config": {"effort": effort}}
# Unknown model (one we have no price row for): assume the most expensive tier.
CONSERVATIVE_PRICE = (10.00, 12.50, 1.00, 50.00)


# --------------------------------------------------------------------------
# Provider-neutral types
# --------------------------------------------------------------------------

@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict


@dataclass
class ToolCall:
    id: str
    name: str
    input: Optional[dict]          # None if the model sent arguments that are not valid JSON
    raw_arguments: str = ""


@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class Usage:
    input_tokens: int = 0          # uncached
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_input(self) -> int:
        return self.input_tokens + self.cache_write_tokens + self.cache_read_tokens


@dataclass
class Turn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    model: str = ""
    stop_reason: Optional[str] = None
    seconds: float = 0.0
    error: Optional[str] = None


class ChatClient(Protocol):
    provider: str
    model: str
    input_price_per_token: float
    output_price_per_token: float

    def add_user_text(self, text: str) -> None: ...
    def add_tool_results(self, results: list[ToolResult]) -> None: ...
    def step(self) -> Turn: ...


def price_of(model: str) -> tuple[float, float, float, float]:
    """The price row for a model. The API can report a dated ID (claude-haiku-4-5-20251001) for a model we
    requested by its alias, so a trailing -YYYYMMDD is dropped before the lookup."""
    return PRICES.get(model) or PRICES.get(re.sub(r"-\d{8}$", "", model), CONSERVATIVE_PRICE)


def cost_usd(model: str, usage: Usage) -> float:
    if model.endswith(":free"):
        return 0.0
    p_in, p_write, p_read, p_out = price_of(model)
    return (usage.input_tokens * p_in + usage.cache_write_tokens * p_write
            + usage.cache_read_tokens * p_read + usage.output_tokens * p_out) / 1_000_000


def read_env_key(name: str, env_path: Optional[Path] = None) -> str:
    """Read one key from .env (or the environment). Never printed. The exact name wins; otherwise a name that
    differs only in case is accepted (e.g. tavily_api_key for TAVILY_API_KEY)."""
    env_path = env_path or PROJECT_ROOT / ".env"
    if env_path.exists():
        pairs = [line.strip().partition("=") for line in env_path.read_text().splitlines()]
        pairs = [(k.strip(), v.strip().strip("'\"")) for k, sep, v in pairs if sep]
        for key, value in pairs:
            if key == name:
                return value
        for key, value in pairs:
            if key.lower() == name.lower():
                return value
    if os.environ.get(name):
        return os.environ[name]
    raise RuntimeError(f"{name} not found in .env")


def inline_refs(schema: dict) -> dict:
    """Replace pydantic's '$ref': '#/$defs/X' with the definition itself, so the tool schema is
    self-contained for every provider."""
    defs = schema.get("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs[node["$ref"].split("/")[-1]])
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


def make_client(provider: str, system: str, tools: list[ToolSpec], model: Optional[str] = None) -> ChatClient:
    model = model or DEFAULT_MODELS[provider]
    if provider == "anthropic":
        return AnthropicClient(model, system, tools)
    if provider == "openrouter":
        return OpenRouterClient(model, system, tools)
    raise ValueError(f"unknown provider: {provider}")


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------

class AnthropicClient:
    provider = "anthropic"
    EFFORT = "medium"               # constant for the whole run (changing it would break the cache)
    MAX_TOKENS = 32_000

    def __init__(self, model: str, system: str, tools: list[ToolSpec]):
        import anthropic
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=read_env_key("ANTHROPIC_API_KEY"))
        self.model = model
        p_in, _, _, p_out = price_of(model)
        self.input_price_per_token, self.output_price_per_token = p_in / 1e6, p_out / 1e6
        # Explicit breakpoint after the system prompt: tools + system are cached as one prefix.
        self.system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        self.tools = [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in tools]
        self.messages: list[dict] = []

    def add_user_text(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_tool_results(self, results: list[ToolResult]) -> None:
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": r.tool_call_id, "content": r.content, "is_error": r.is_error}
            for r in results]})

    def step(self) -> Turn:
        start = time.monotonic()
        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.MAX_TOKENS,
                system=self.system,
                tools=self.tools,
                tool_choice={"type": "auto"},
                cache_control={"type": "ephemeral"},     # automatic caching of the growing conversation
                messages=self.messages,
                **effort_options(self.model, self.EFFORT),
            ) as stream:
                message = stream.get_final_message()
        except self._anthropic.APIError as e:
            return Turn(error=f"{type(e).__name__}: {e}", seconds=time.monotonic() - start)   # no model answered

        # Append the full content (thinking blocks included) so the next turn sees it unchanged.
        self.messages.append({"role": "assistant", "content": message.content})
        u = message.usage
        usage = Usage(
            input_tokens=u.input_tokens or 0,
            cache_write_tokens=u.cache_creation_input_tokens or 0,
            cache_read_tokens=u.cache_read_input_tokens or 0,
            output_tokens=u.output_tokens or 0,
        )
        turn = Turn(
            text="".join(b.text for b in message.content if b.type == "text"),
            tool_calls=[ToolCall(id=b.id, name=b.name, input=b.input if isinstance(b.input, dict) else None)
                        for b in message.content if b.type == "tool_use"],
            usage=usage,
            cost_usd=cost_usd(message.model, usage),
            model=message.model,
            stop_reason=message.stop_reason,
            seconds=time.monotonic() - start,
        )
        if message.stop_reason == "refusal":
            category = getattr(message.stop_details, "category", None) if message.stop_details else None
            turn.error = f"refusal (category: {category})"
        elif message.stop_reason == "max_tokens":
            turn.error = "hit max_tokens"
        return turn


# --------------------------------------------------------------------------
# OpenRouter (OpenAI-compatible chat API)
# --------------------------------------------------------------------------

class OpenRouterClient:
    provider = "openrouter"
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    TIMEOUT_SECONDS = 180
    RETRY_WAITS_SECONDS = [10, 30, 60]
    MAX_RETRY_AFTER_SECONDS = 120

    def __init__(self, model: str, system: str, tools: list[ToolSpec]):
        self.key = read_env_key("OPENROUTER_API_KEY")
        self.model = model
        p_in, _, _, p_out = (0, 0, 0, 0) if model.endswith(":free") else price_of(model)
        self.input_price_per_token, self.output_price_per_token = p_in / 1e6, p_out / 1e6
        self.tools = [{"type": "function", "function": {
            "name": t.name, "description": t.description, "parameters": t.input_schema}} for t in tools]
        self.messages: list[dict] = [{"role": "system", "content": system}]

    def add_user_text(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_tool_results(self, results: list[ToolResult]) -> None:
        for r in results:
            content = f"ERROR: {r.content}" if r.is_error else r.content
            self.messages.append({"role": "tool", "tool_call_id": r.tool_call_id, "content": content})

    def step(self) -> Turn:
        body = {"model": self.model, "messages": self.messages, "tools": self.tools, "tool_choice": "auto"}
        start = time.monotonic()
        for attempt in range(len(self.RETRY_WAITS_SECONDS) + 1):
            data, error, retryable, retry_after = self._post(body)
            if not error or not retryable or attempt == len(self.RETRY_WAITS_SECONDS):
                break
            wait = self.RETRY_WAITS_SECONDS[attempt]
            if retry_after:
                try:
                    wait = min(max(float(retry_after), 0), self.MAX_RETRY_AFTER_SECONDS)
                except ValueError:
                    pass
            print(f"  retry {attempt + 1}/{len(self.RETRY_WAITS_SECONDS)} in {wait:.0f}s after: {error}", flush=True)
            time.sleep(wait)
        seconds = time.monotonic() - start
        if error:
            return Turn(error=error, seconds=seconds)   # no model answered

        turn = parse_openrouter_response(data, self.model)
        turn.seconds = seconds
        message = data["choices"][0]["message"]
        self.messages.append({"role": "assistant", "content": message.get("content") or "",
                              **({"tool_calls": message["tool_calls"]} if message.get("tool_calls") else {})})
        return turn

    def _post(self, body: dict) -> tuple[Optional[dict], Optional[str], bool, Optional[str]]:
        """Returns (data, error, retryable, Retry-After)."""
        request = urllib.request.Request(
            self.API_URL, data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                     "X-Title": "Founder Dossiers"})
        try:
            with urllib.request.urlopen(request, timeout=self.TIMEOUT_SECONDS) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace")
            retry_after = e.headers.get("Retry-After") if e.headers else None
            e.close()
            try:
                raw = json.loads(raw).get("error", {}).get("message") or raw
            except (json.JSONDecodeError, AttributeError):
                pass
            return None, f"HTTP {e.code}: {raw}", e.code == 429 or e.code >= 500, retry_after
        except Exception as e:
            return None, f"{type(e).__name__}: {e}", False, None
        if data.get("error"):
            return None, f"provider error: {data['error'].get('message')}", True, None
        if not data.get("choices"):
            return None, "response has no choices", False, None
        return data, None, False, None


def parse_openrouter_response(data: dict, requested_model: str) -> Turn:
    message = data["choices"][0]["message"]
    calls = []
    for c in message.get("tool_calls") or []:
        raw = (c.get("function") or {}).get("arguments") or ""
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = None
        calls.append(ToolCall(id=c.get("id", ""), name=(c.get("function") or {}).get("name", ""),
                              input=parsed if isinstance(parsed, dict) else None, raw_arguments=raw))
    u = data.get("usage") or {}
    cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
    usage = Usage(input_tokens=(u.get("prompt_tokens") or 0) - cached, cache_read_tokens=cached,
                  output_tokens=u.get("completion_tokens") or 0)
    model = data.get("model") or requested_model
    # OpenRouter reports the actual charge when it has one; otherwise use the price table (0 for :free).
    cost = u.get("cost") if isinstance(u.get("cost"), (int, float)) else cost_usd(requested_model, usage)
    return Turn(text=message.get("content") or "", tool_calls=calls, usage=usage, cost_usd=float(cost),
                model=model, stop_reason=data["choices"][0].get("finish_reason"))
