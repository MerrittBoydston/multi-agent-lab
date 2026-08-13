from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .models import AgentSpec, Artifact
from .providers import bounded_completion, provider_for
from .store import RunStore
from .validation import OutputValidationError, validate_output


PRODUCER_PROMPT = """You are an independent solution producer. Do not assume other agents agree with you.
Return JSON with: proposal, assumptions[], risks[], confidence (0..1). Be concrete and concise."""
CRITIC_PROMPT = """You are an adversarial reviewer. Evaluate the candidate against the objective.
Return JSON with: findings[], recommendation (approve|revise|reject), confidence (0..1)."""
JUDGE_PROMPT = """You are an impartial synthesizer. Preserve important disagreements rather than forcing consensus.
Return JSON with: decision, final, unresolved_questions[], confidence (0..1)."""


@dataclass(frozen=True)
class WorkflowResult:
    run_id: str
    final_artifact: Artifact


class WorkflowInterrupted(Exception):
    def __init__(self, run_id: str):
        super().__init__(f"Run {run_id} was interrupted")
        self.run_id = run_id


class CollaborationWorkflow:
    def __init__(self, workspace: Path, progress: Callable[[str], None] | None = None):
        self.store = RunStore(workspace)
        self.progress = progress or (lambda message: None)
        self.timeout = float(os.getenv("AGENT_TIMEOUT_SECONDS", "90"))
        self.max_output = int(os.getenv("AGENT_MAX_OUTPUT_CHARS", "20000"))
        self.max_retries = int(os.getenv("AGENT_MAX_RETRIES", "3"))
        self.retry_delay = float(os.getenv("AGENT_RETRY_DELAY_SECONDS", "1"))
        self.reviewer_input_limit = int(os.getenv("REVIEWER_MAX_INPUT_CHARS", "16000"))
        if self.reviewer_input_limit < 1000:
            raise ValueError("REVIEWER_MAX_INPUT_CHARS must be at least 1000")

    def _agent(self, agent_id: str, role: str, prompt: str) -> AgentSpec:
        prefix = agent_id.upper().replace("-", "_")
        default_provider = os.getenv("AGENT_PROVIDER", "mock")
        provider = os.getenv(f"{prefix}_PROVIDER", default_provider)
        default_model = "mock-v1" if provider == "mock" else ""
        model = os.getenv(f"{prefix}_MODEL", default_model)
        if not model:
            raise ValueError(f"Set {prefix}_MODEL for provider {provider}")
        return AgentSpec(agent_id, role, provider, model, prompt)

    async def _call(self, run_id: str, stage: str, agent: AgentSpec, prompt: str) -> Artifact:
        self.progress(f"[{run_id}] {stage}: starting {agent.agent_id} ({agent.provider}/{agent.model})")
        self.store.event(run_id, stage, "agent_started", {"provider": agent.provider, "model": agent.model}, agent.agent_id)
        for attempt in range(self.max_retries + 1):
            try:
                result = await bounded_completion(
                    provider_for(agent.provider), agent, prompt,
                    timeout_seconds=self.timeout, max_output_chars=self.max_output,
                )
                break
            except Exception as exc:
                if attempt >= self.max_retries or not self._is_transient(exc):
                    raise
                delay = self.retry_delay * (2 ** attempt)
                self.store.event(
                    run_id,
                    stage,
                    "agent_retrying",
                    {
                        "attempt": attempt + 1,
                        "max_retries": self.max_retries,
                        "delay_seconds": delay,
                        "error": type(exc).__name__,
                        "message": str(exc),
                    },
                    agent.agent_id,
                )
                self.progress(
                    f"[{run_id}] {stage}: {agent.agent_id} retry {attempt + 1}/{self.max_retries} "
                    f"in {delay:g}s ({type(exc).__name__})"
                )
                await asyncio.sleep(delay)
        try:
            result = validate_output(agent.role, result)
        except OutputValidationError as exc:
            self.store.event(run_id, stage, "output_repairing", {"error": str(exc)}, agent.agent_id)
            self.progress(f"[{run_id}] {stage}: repairing malformed output from {agent.agent_id}")
            repair_prompt = (
                f"Your previous response failed validation: {exc}.\n"
                "Return only a corrected JSON object matching your required schema. "
                "Do not use Markdown fences. Preserve the substance of the response.\n\n"
                f"PREVIOUS RESPONSE:\n{result}"
            )
            repaired = await bounded_completion(
                provider_for(agent.provider), agent, repair_prompt,
                timeout_seconds=self.timeout, max_output_chars=self.max_output,
            )
            try:
                result = validate_output(agent.role, repaired)
            except OutputValidationError as repair_exc:
                self.store.event(
                    run_id, stage, "output_repair_failed", {"error": str(repair_exc)}, agent.agent_id
                )
                raise OutputValidationError(
                    f"{agent.agent_id} returned invalid output after one repair: {repair_exc}"
                ) from repair_exc
            self.store.event(run_id, stage, "output_repaired", {}, agent.agent_id)
        artifact = self.store.artifact(run_id, stage, agent.agent_id, result)
        self.store.event(run_id, stage, "agent_completed", {"artifact_id": artifact.artifact_id}, agent.agent_id)
        self.progress(f"[{run_id}] {stage}: completed {agent.agent_id}")
        return artifact

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        if isinstance(exc, (TimeoutError, ConnectionError, asyncio.TimeoutError)):
            return True
        if type(exc).__name__ in {
            "APIConnectionError",
            "APITimeoutError",
            "RateLimitError",
            "RemoteProtocolError",
        }:
            return True
        status_code = getattr(exc, "status_code", None)
        return status_code == 429 or isinstance(status_code, int) and status_code >= 500

    async def _complete_stage(self, calls) -> list[Artifact]:
        results = await asyncio.gather(*calls, return_exceptions=True)
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise failures[0]
        return results

    @staticmethod
    def _compact_text(text: str, limit: int) -> str:
        marker = "\n...[compacted]...\n"
        if len(text) <= limit:
            return text
        if limit <= len(marker):
            return text[:limit]
        remaining = limit - len(marker)
        head = (remaining + 1) // 2
        tail = remaining // 2
        return text[:head] + marker + (text[-tail:] if tail else "")

    def _build_judge_input(self, sections: list[tuple[str, str]]) -> tuple[str, bool]:
        def render(values: list[str]) -> str:
            return "\n\n".join(
                f"{label}:\n{value}" for (label, _), value in zip(sections, values)
            )

        values = [value for _, value in sections]
        original = render(values)
        if len(original) <= self.reviewer_input_limit:
            return original, False

        fixed_chars = len(render(["" for _ in sections]))
        available = self.reviewer_input_limit - fixed_chars
        allocations = [0] * len(values)
        active = set(range(len(values)))
        while active:
            share, remainder = divmod(available, len(active))
            satisfied = [i for i in active if len(values[i]) <= share]
            if not satisfied:
                for position, i in enumerate(sorted(active)):
                    allocations[i] = share + (1 if position < remainder else 0)
                break
            for i in satisfied:
                allocations[i] = len(values[i])
                available -= allocations[i]
                active.remove(i)

        compacted = render([
            self._compact_text(value, allocation)
            for value, allocation in zip(values, allocations)
        ])
        return compacted, True

    async def run(self, objective: str | None = None, *, resume_id: str | None = None) -> WorkflowResult:
        if resume_id is not None:
            if objective is not None:
                raise ValueError("Do not supply an objective when resuming a run")
            run_id = resume_id
            objective = self.store.resume_run(run_id)
            self.progress(f"[{run_id}] resuming run")
        else:
            if objective is None:
                raise ValueError("An objective is required for a new run")
            run_id = self.store.create_run(objective)
            self.progress(f"[{run_id}] created run")
        try:
            producer_a = self._agent("producer-a", "producer", PRODUCER_PROMPT)
            producer_b = self._agent("producer-b", "producer", PRODUCER_PROMPT)
            critic_a = self._agent("critic-a", "critic", CRITIC_PROMPT)
            critic_b = self._agent("critic-b", "critic", CRITIC_PROMPT)
            reviewer = self._agent("reviewer", "reviewer", JUDGE_PROMPT)

            completed = self.store.artifacts_for_run(run_id)
            if resume_id is not None:
                for artifact in completed.values():
                    self.store.event(
                        run_id,
                        artifact.stage,
                        "artifact_reused",
                        {"artifact_id": artifact.artifact_id},
                        artifact.author,
                    )
                    self.progress(f"[{run_id}] {artifact.stage}: reusing {artifact.author}")
            proposal_agents = [producer_a, producer_b]
            missing_proposals = [
                self._call(run_id, "proposal", agent, objective)
                for agent in proposal_agents
                if ("proposal", agent.agent_id) not in completed
            ]
            if missing_proposals:
                await self._complete_stage(missing_proposals)
                completed = self.store.artifacts_for_run(run_id)
            proposals = [completed[("proposal", agent.agent_id)] for agent in proposal_agents]
            proposal_text = [Path(item.path).read_text(encoding="utf-8") for item in proposals]

            critique_calls = [
                (critic_a, f"OBJECTIVE:\n{objective}\n\nCANDIDATE B:\n{proposal_text[1]}"),
                (critic_b, f"OBJECTIVE:\n{objective}\n\nCANDIDATE A:\n{proposal_text[0]}"),
            ]
            missing_critiques = [
                self._call(run_id, "critique", agent, prompt)
                for agent, prompt in critique_calls
                if ("critique", agent.agent_id) not in completed
            ]
            if missing_critiques:
                await self._complete_stage(missing_critiques)
                completed = self.store.artifacts_for_run(run_id)
            critiques = [completed[("critique", agent.agent_id)] for agent, _ in critique_calls]
            critique_text = [Path(item.path).read_text(encoding="utf-8") for item in critiques]

            judge_sections = [
                ("OBJECTIVE", objective),
                ("PROPOSAL A", proposal_text[0]),
                ("PROPOSAL B", proposal_text[1]),
                ("CRITIQUE OF B", critique_text[0]),
                ("CRITIQUE OF A", critique_text[1]),
            ]
            original_judge_chars = len("\n\n".join(
                f"{label}:\n{value}" for label, value in judge_sections
            ))
            judge_input, compacted = self._build_judge_input(judge_sections)
            if compacted:
                self.store.event(
                    run_id,
                    "final",
                    "reviewer_input_compacted",
                    {
                        "original_chars": original_judge_chars,
                        "compacted_chars": len(judge_input),
                        "max_chars": self.reviewer_input_limit,
                    },
                    reviewer.agent_id,
                )
                self.progress(
                    f"[{run_id}] final: compacted reviewer input "
                    f"from {original_judge_chars} to {len(judge_input)} characters"
                )
            final_artifact = completed.get(("final", reviewer.agent_id))
            if final_artifact is None:
                final_artifact = await self._call(run_id, "final", reviewer, judge_input)
            self.store.finish(run_id)
            self.progress(f"[{run_id}] workflow completed")
            return WorkflowResult(run_id, final_artifact)
        except asyncio.CancelledError:
            self.store.event(run_id, "workflow", "interrupted", {})
            self.store.finish(run_id, "failed")
            raise WorkflowInterrupted(run_id) from None
        except Exception as exc:
            self.store.event(run_id, "workflow", "failed", {"error": type(exc).__name__, "message": str(exc)})
            self.store.finish(run_id, "failed")
            raise
