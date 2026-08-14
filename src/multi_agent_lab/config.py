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
PREVIOUS_PATTERN = re.compile(r"{{\s*previous\s*}}")
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
    depends_on: tuple[str, ...] = ()
    when: dict[str, Any] | None = None
    repeat: dict[str, Any] | None = None
    controls: dict[str, Any] | None = None


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
                    **({"depends_on": list(stage.depends_on)} if self.version == 2 else {}),
                    **({"when": stage.when} if stage.when else {}),
                    **({"repeat": stage.repeat} if stage.repeat else {}),
                    **({"controls": stage.controls} if stage.controls else {}),
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
    version = raw.get("version")
    if version not in {1, 2}:
        raise ValueError("configuration version must be 1 or 2")
    raw_stages = raw.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("stages must be a non-empty array")

    stages = []
    known_slots: set[tuple[str, str]] = set()
    stage_ids: set[str] = set()
    agent_ids: set[str] = set()
    for stage_index, raw_stage in enumerate(raw_stages):
        location = f"stages[{stage_index}]"
        allowed = {"id", "agents"} if version == 1 else {"id", "agents", "depends_on", "when", "repeat", "controls"}
        if not isinstance(raw_stage, dict) or not {"id", "agents"}.issubset(raw_stage) or set(raw_stage) - allowed:
            raise ValueError(f"{location} contains invalid stage fields")
        stage_id = _nonempty_string(raw_stage["id"], f"{location}.id")
        if not ID_PATTERN.fullmatch(stage_id):
            raise ValueError(f"{location}.id must use lowercase letters, numbers, and hyphens")
        if stage_id in stage_ids:
            raise ValueError(f"duplicate stage id {stage_id!r}")
        stage_ids.add(stage_id)
        depends_on = tuple(raw_stage.get("depends_on", ())) if version == 2 else (() if not stages else (stages[-1].stage_id,))
        if not all(isinstance(item, str) for item in depends_on) or len(set(depends_on)) != len(depends_on):
            raise ValueError(f"{location}.depends_on must contain unique stage ids")
        if stage_id in depends_on:
            raise ValueError(f"{location}.depends_on cannot reference itself")
        missing_dependencies = set(depends_on) - (stage_ids - {stage_id})
        if missing_dependencies:
            raise ValueError(f"{location}.depends_on references unavailable stage {sorted(missing_dependencies)[0]!r}")
        when = raw_stage.get("when") if version == 2 else None
        repeat = raw_stage.get("repeat") if version == 2 else None
        controls = raw_stage.get("controls") if version == 2 else None
        if controls is not None:
            allowed_controls = {"timeout_seconds", "max_retries", "max_output_chars"}
            if not isinstance(controls, dict) or set(controls) - allowed_controls:
                raise ValueError(f"{location}.controls contains unsupported fields")
            for key, value in controls.items():
                minimum = 0 if key == "max_retries" else 1
                if isinstance(value, bool) or not isinstance(value, (int, float)) or value < minimum:
                    raise ValueError(f"{location}.controls.{key} must be at least {minimum}")
        for name, condition in (("when", when), ("repeat.until", repeat.get("until") if isinstance(repeat, dict) else None)):
            if condition is not None:
                if not isinstance(condition, dict) or set(condition) != {"artifact", "field", "equals"}:
                    raise ValueError(f"{location}.{name} must contain artifact, field, and equals")
                parts = str(condition["artifact"]).split("/")
                allowed_condition_slots = known_slots
                if name == "repeat.until":
                    allowed_condition_slots = allowed_condition_slots | {
                        (stage_id, str(a.get("id"))) for a in raw_stage["agents"] if isinstance(a, dict)
                    }
                if len(parts) != 2 or tuple(parts) not in allowed_condition_slots:
                    raise ValueError(f"{location}.{name} references an unavailable artifact")
                _nonempty_string(condition["field"], f"{location}.{name}.field")
        if repeat is not None:
            if not isinstance(repeat, dict) or set(repeat) != {"max_iterations", "until"}:
                raise ValueError(f"{location}.repeat must contain max_iterations and until")
            maximum = repeat["max_iterations"]
            if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 20:
                raise ValueError(f"{location}.repeat.max_iterations must be from 1 to 20")
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
            stripped = REFERENCE_PATTERN.sub("", OBJECTIVE_PATTERN.sub("", PREVIOUS_PATTERN.sub("", input_template)))
            unresolved_tokens = re.findall(r"{{.*?}}", stripped)
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
        stages.append(ConfigStage(stage_id, tuple(agents), depends_on, when, repeat, controls))

    if len(stages[-1].agents) != 1:
        raise ValueError("the final stage must contain exactly one agent")
    if stages[-1].when or stages[-1].repeat:
        raise ValueError("the final stage cannot be conditional or repeated")
    return WorkflowConfig(version, tuple(stages))


def load_config(path: Path) -> WorkflowConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"Configuration file {str(path)!r} does not exist") from None
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {str(path)!r}: {exc}") from exc
    return config_from_mapping(raw)
