from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    role: str
    provider: str
    model: str
    system_prompt: str


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    run_id: str
    stage: str
    author: str
    path: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()

