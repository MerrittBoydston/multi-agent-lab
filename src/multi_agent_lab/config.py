from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")
REFERENCE_PATTERN = re.compile(r"{{\s*artifact\.([a-z][a-z0-9-]*)\.([a-z][a-z0-9-]*)\s*}}")
OBJECTIVE_PATTERN = re.compile(r"{{\s*objective\s*}}")
ROLES = {"producer", "critic", "reviewer"}
PROVIDERS = {"mock", "anthropic", "openai"}


@dataclass(frozen=True)
class ConfigAgent:
    agent_id: str
    role: str
    provider: str
    model: str
    system_prompt: str
    input_template: str


@dataclass(frozen=True)
class ConfigStage:
    stage_id: str
    agents: tuple[ConfigAgent, ...]


@dataclass(frozen=True)
class WorkflowConfig:
    version: int
    stages: tuple[ConfigStage, ...]

    def to_json(self) -> str:
        value = {
            "version": self.version,
            "stages": [
                {
                    "id": stage.stage_id,
                    "agents": [
                        {
                            "id": agent.agent_id,
                            "role": agent.role,
                            "provider": agent.provider,
                            "model": agent.model,
                            "system_prompt": agent.system_prompt,
                            "input": agent.input_template,
                        }
                        for agent in stage.agents
                    ],
                }
                for stage in self.stages
            ],
        }
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> WorkflowConfig:
        return config_from_mapping(json.loads(value))

    @property
    def slots(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (stage.stage_id, agent.agent_id)
            for stage in self.stages
            for agent in stage.agents
        )


def _nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value


def config_from_mapping(raw: Any) -> WorkflowConfig:
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a mapping")
    unknown = set(raw) - {"version", "stages"}
    if unknown:
        raise ValueError(f"unknown configuration field(s): {', '.join(sorted(unknown))}")
    if raw.get("version") != 1:
        raise ValueError("configuration version must be 1")
    raw_stages = raw.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("stages must be a non-empty array")

    stages = []
    known_slots: set[tuple[str, str]] = set()
    stage_ids: set[str] = set()
    agent_ids: set[str] = set()
    for stage_index, raw_stage in enumerate(raw_stages):
        location = f"stages[{stage_index}]"
        if not isinstance(raw_stage, dict) or set(raw_stage) != {"id", "agents"}:
            raise ValueError(f"{location} must contain only id and agents")
        stage_id = _nonempty_string(raw_stage["id"], f"{location}.id")
        if not ID_PATTERN.fullmatch(stage_id):
            raise ValueError(f"{location}.id must use lowercase letters, numbers, and hyphens")
        if stage_id in stage_ids:
            raise ValueError(f"duplicate stage id {stage_id!r}")
        stage_ids.add(stage_id)
        raw_agents = raw_stage["agents"]
        if not isinstance(raw_agents, list) or not raw_agents:
            raise ValueError(f"{location}.agents must be a non-empty array")
        agents = []
        stage_slots = set()
        for agent_index, raw_agent in enumerate(raw_agents):
            agent_location = f"{location}.agents[{agent_index}]"
            required = {"id", "role", "provider", "model", "system_prompt", "input"}
            if not isinstance(raw_agent, dict) or set(raw_agent) != required:
                raise ValueError(f"{agent_location} must contain exactly {', '.join(sorted(required))}")
            agent_id = _nonempty_string(raw_agent["id"], f"{agent_location}.id")
            if not ID_PATTERN.fullmatch(agent_id):
                raise ValueError(f"{agent_location}.id must use lowercase letters, numbers, and hyphens")
            if agent_id in agent_ids:
                raise ValueError(f"duplicate agent id {agent_id!r}")
            agent_ids.add(agent_id)
            role = _nonempty_string(raw_agent["role"], f"{agent_location}.role").lower()
            provider = _nonempty_string(raw_agent["provider"], f"{agent_location}.provider").lower()
            if role not in ROLES:
                raise ValueError(f"{agent_location}.role must be one of {', '.join(sorted(ROLES))}")
            if provider not in PROVIDERS:
                raise ValueError(f"{agent_location}.provider must be one of {', '.join(sorted(PROVIDERS))}")
            model = _nonempty_string(raw_agent["model"], f"{agent_location}.model")
            system_prompt = _nonempty_string(raw_agent["system_prompt"], f"{agent_location}.system_prompt")
            input_template = _nonempty_string(raw_agent["input"], f"{agent_location}.input")
            references = set(REFERENCE_PATTERN.findall(input_template))
            unresolved_tokens = re.findall(r"{{.*?}}", REFERENCE_PATTERN.sub("", OBJECTIVE_PATTERN.sub("", input_template)))
            if unresolved_tokens:
                raise ValueError(f"{agent_location}.input contains unsupported template {unresolved_tokens[0]!r}")
            missing = references - known_slots
            if missing:
                stage, author = sorted(missing)[0]
                raise ValueError(
                    f"{agent_location}.input references unavailable artifact {stage}/{author}"
                )
            agent = ConfigAgent(agent_id, role, provider, model, system_prompt, input_template)
            agents.append(agent)
            stage_slots.add((stage_id, agent_id))
        known_slots.update(stage_slots)
        stages.append(ConfigStage(stage_id, tuple(agents)))

    if len(stages[-1].agents) != 1:
        raise ValueError("the final stage must contain exactly one agent")
    return WorkflowConfig(1, tuple(stages))


def load_config(path: Path) -> WorkflowConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"Configuration file {str(path)!r} does not exist") from None
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {str(path)!r}: {exc}") from exc
    return config_from_mapping(raw)
