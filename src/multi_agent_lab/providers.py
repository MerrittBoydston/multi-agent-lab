from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any

from .models import AgentSpec, CompletionResult


class AgentProvider(ABC):
    @abstractmethod
    async def complete(self, agent: AgentSpec, prompt: str) -> CompletionResult:
        raise NotImplementedError


class MockProvider(AgentProvider):
    """Deterministic provider for learning, tests, and workflow inspection."""

    async def complete(self, agent: AgentSpec, prompt: str) -> CompletionResult:
        await asyncio.sleep(0)
        if agent.role == "producer":
            emphasis = "execution simplicity" if agent.agent_id.endswith("a") else "independent verification"
            content = json.dumps(
                {
                    "proposal": f"A proposal from {agent.agent_id} emphasizing {emphasis}.",
                    "assumptions": ["The objective can be evaluated from the supplied brief."],
                    "risks": ["Mock output is structural, not substantive."],
                    "confidence": 0.55,
                },
                indent=2,
            )
            return CompletionResult(content)
        if agent.role == "critic":
            content = json.dumps(
                {
                    "findings": ["The candidate needs more evidence and measurable acceptance criteria."],
                    "recommendation": "revise",
                    "confidence": 0.8,
                },
                indent=2,
            )
            return CompletionResult(content)
        content = json.dumps(
            {
                "decision": "merge",
                "final": "Combine the clearest proposal with explicit evidence and acceptance criteria.",
                "unresolved_questions": ["Which evidence sources should be authoritative?"],
                "confidence": 0.7,
            },
            indent=2,
        )
        return CompletionResult(content)


class AnthropicProvider(AgentProvider):
    async def complete(self, agent: AgentSpec, prompt: str) -> CompletionResult:
        from anthropic import AsyncAnthropic

        async with AsyncAnthropic().messages.stream(
            model=agent.model,
            max_tokens=3000,
            system=agent.system_prompt,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            message = await stream.get_final_message()
        content = "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )
        return CompletionResult(content, message.usage.input_tokens, message.usage.output_tokens)


class OpenAIProvider(AgentProvider):
    async def complete(self, agent: AgentSpec, prompt: str) -> CompletionResult:
        from openai import AsyncOpenAI

        response = await AsyncOpenAI().responses.create(
            model=agent.model,
            instructions=agent.system_prompt,
            input=prompt,
        )
        usage = response.usage
        return CompletionResult(
            response.output_text,
            getattr(usage, "input_tokens", None),
            getattr(usage, "output_tokens", None),
        )


def provider_for(name: str) -> AgentProvider:
    providers: dict[str, type[AgentProvider]] = {
        "mock": MockProvider,
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
    }
    try:
        return providers[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unknown provider {name!r}; choose {', '.join(providers)}") from exc


async def bounded_completion(
    provider: AgentProvider,
    agent: AgentSpec,
    prompt: str,
    *,
    timeout_seconds: float,
    max_output_chars: int,
) -> CompletionResult:
    result = await asyncio.wait_for(provider.complete(agent, prompt), timeout=timeout_seconds)
    if isinstance(result, str):
        result = CompletionResult(result)
    if len(result.content) > max_output_chars:
        raise ValueError(f"{agent.agent_id} exceeded the output limit")
    return result
