"""LLM call wrapper supporting OpenAI and Anthropic."""

import contextvars
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from app.observability import get_tracer


class LLMConfig(BaseModel):
    """Configuration for an LLM call."""

    model: str = Field(default="gpt-4o")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    provider: str = Field(default="openai")
    base_url: str = Field(default="https://api.deepseek.com")


class LLMProvider:
    """Provider enum values."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


_usage_meter: "contextvars.ContextVar[Optional[List[Dict[str, Any]]]]" = contextvars.ContextVar(
    "deep_research_usage_meter", default=None
)

_openai_clients: Dict[Tuple[str, str], Any] = {}
_anthropic_clients: Dict[Tuple[str, str], Any] = {}


def set_usage_meter(meter: Optional[List[Dict[str, Any]]]) -> None:
    """Set the usage accumulator for the current task (or clear it)."""
    _usage_meter.set(meter)


def _get_openai_client(api_key: str, base_url: str):
    key = (api_key, base_url)
    if key not in _openai_clients:
        from openai import AsyncOpenAI

        _openai_clients[key] = AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=60.0, max_retries=2
        )
    return _openai_clients[key]


def _get_anthropic_client(api_key: str):
    key = (api_key,)
    if key not in _anthropic_clients:
        from anthropic import AsyncAnthropic

        _anthropic_clients[key] = AsyncAnthropic(api_key=api_key, timeout=30.0, max_retries=2)
    return _anthropic_clients[key]


def _record_usage(model: str, prompt_tokens: int, completion_tokens: int) -> None:
    meter = _usage_meter.get()
    if meter is not None:
        meter.append({
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        })


async def llm_call(
    system_prompt: str,
    user_prompt: str,
    config: Optional[LLMConfig] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Unified LLM call wrapper with optional tracing and usage metering."""
    if config is None:
        config = LLMConfig()

    tracer = get_tracer()
    span = None
    if tracer is not None:
        span = tracer.start_span(
            name="llm_call",
            input={
                "model": config.model,
                "system": system_prompt[:2000],
                "user": user_prompt[:2000],
            },
        )

    provider = config.provider.lower()
    try:
        if provider == LLMProvider.OPENAI:
            text = await _call_openai(system_prompt, user_prompt, config, tools)
        elif provider == LLMProvider.ANTHROPIC:
            text = await _call_anthropic(system_prompt, user_prompt, config, tools)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    except Exception:
        if span is not None:
            span.end(level="ERROR")
        raise

    if span is not None:
        span.end(output=text[:2000])
    return text


async def _call_openai(
    system_prompt: str,
    user_prompt: str,
    config: LLMConfig,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Call OpenAI API."""
    from app.config import settings
    from app.services.config_service import get_active_config

    rt = get_active_config()
    api_key = rt.api_key if (rt and rt.api_key) else (settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY"))
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    base_url = rt.base_url if (rt and rt.base_url) else config.base_url
    client = _get_openai_client(api_key, base_url)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    kwargs = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }

    if tools:
        kwargs["tools"] = tools

    response = await client.chat.completions.create(**kwargs)
    usage = getattr(response, "usage", None)
    if usage is not None:
        _record_usage(config.model, usage.prompt_tokens or 0, usage.completion_tokens or 0)
    return response.choices[0].message.content or ""


async def _call_anthropic(
    system_prompt: str,
    user_prompt: str,
    config: LLMConfig,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Call Anthropic API."""
    from app.config import settings
    from app.services.config_service import get_active_config

    rt = get_active_config()
    api_key = rt.api_key if (rt and rt.api_key) else (settings.ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY"))
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    client = _get_anthropic_client(api_key)

    kwargs = {
        "model": config.model,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }

    if tools:
        kwargs["tools"] = tools

    response = await client.messages.create(**kwargs)
    usage = getattr(response, "usage", None)
    if usage is not None:
        _record_usage(
            config.model,
            usage.input_tokens or 0,
            usage.output_tokens or 0,
        )
    return response.content[0].text if response.content else ""


def resolve_model(agent_model: Optional[str] = None) -> str:
    """Resolve the model name: explicit agent model > runtime config > default."""
    if agent_model:
        return agent_model
    from app.services.config_service import get_active_config

    rt = get_active_config()
    if rt and rt.model:
        return rt.model
    return "gpt-4o"


def extract_json_from_response(response: str) -> Dict[str, Any]:
    """Extract JSON from LLM response."""
    import re

    json_str = response.strip()

    # Try markdown code block first
    match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", json_str)
    if match:
        json_str = match.group(1).strip()

    return json.loads(json_str)
