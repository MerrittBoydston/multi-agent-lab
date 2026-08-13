from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any

from .models import AgentSpec


class AgentProvider(ABC):
    @abstractmethod
    async def complete(self, agent: AgentSpec, prompt: str) -> str:
        raise NotImplementedError


class MockProvider(AgentProvider):
    """Deterministic provider for learning, tests, and workflow inspection."""

    async def complete(self, agent: AgentSpec, prompt: str) -> str:
        await asyncio.sleep(0)
        if agent.role == "producer":
            emphasis = "execution simplicity" if agent.agent_id.endswith("a") else "independent verification"
            return json.dumps(
                {
                    "proposal": f"A proposal from {agent.agent_id} emphasizing {emphasis}.",
                    "assumptions": ["The objective can be evaluated from the supplied brief."],
                    "risks": ["Mock output is structural, not substantive."],
                    "confidence": 0.55,
                },
                indent=2,
            )
        if agent.role == "critic":
            return json.dumps(
                {
                    "findings": ["The candidate needs more evidence and measurable acceptance criteria."],
                    "recommendation": "revise",
                    "confidence": 0.8,
                },
                indent=2,
            )
        return json.dumps(
            {
                "decision": "merge",
                "final": "Combine the clearest proposal with explicit evidence and acceptance criteria.",
                "unresolved_questions": ["Which evidence sources should be authoritative?"],
                "confidence": 0.7,
            },
            indent=2,
        )


class AnthropicProvider(AgentProvider):
    async def complete(self, agent: AgentSpec, prompt: str) -> str:
        from anthropic import AsyncAnthropic

        async with AsyncAnthropic().messages.stream(
            model=agent.model,
            max_tokens=3000,
            system=agent.system_prompt,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            return await stream.get_final_text()


class OpenAIProvider(AgentProvider):
    async def complete(self, agent: AgentSpec, prompt: str) -> str:
        from openai import AsyncOpenAI

        response = await AsyncOpenAI().responses.create(
            model=agent.model,
            instructions=agent.system_prompt,
            input=prompt,
        )
        return response.output_text


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
) -> str:
    result = await asyncio.wait_for(provider.complete(agent, prompt), timeout=timeout_seconds)
    if len(result) > max_output_chars:
        raise ValueError(f"{agent.agent_id} exceeded the output limit")
    return result
