from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import OBJECTIVE_PATTERN, PREVIOUS_PATTERN, REFERENCE_PATTERN, WorkflowConfig, config_from_mapping
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

    def resume_plan(
        self, run_id: str, requested_config: WorkflowConfig | None = None
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        run, _, _ = self.store.run_details(run_id)
        if run["status"] == "completed":
            raise ValueError(f"Run {run_id!r} is already completed")
        if run["status"] != "failed":
            raise ValueError(f"Run {run_id!r} is {run['status']!r}, not failed")
        config = self._config_for_run(run, requested_config)
        completed = self._with_iteration_aliases(self.store.artifacts_for_run(run_id))
        reused = [slot for slot in config.slots if slot in completed]
        missing = [slot for slot in config.slots if slot not in completed]
        return reused, missing

    def _default_config(self) -> WorkflowConfig:
        agents = {
            "producer-a": self._agent("producer-a", "producer", PRODUCER_PROMPT),
            "producer-b": self._agent("producer-b", "producer", PRODUCER_PROMPT),
            "critic-a": self._agent("critic-a", "critic", CRITIC_PROMPT),
            "critic-b": self._agent("critic-b", "critic", CRITIC_PROMPT),
            "reviewer": self._agent("reviewer", "reviewer", JUDGE_PROMPT),
        }

        def configured(agent_id: str, input_template: str) -> dict:
            agent = agents[agent_id]
            return {
                "id": agent.agent_id,
                "role": agent.role,
                "provider": agent.provider,
                "model": agent.model,
                "system_prompt": agent.system_prompt,
                "input": input_template,
            }

        return config_from_mapping({
            "version": 1,
            "stages": [
                {
                    "id": "proposal",
                    "agents": [
                        configured("producer-a", "{{ objective }}"),
                        configured("producer-b", "{{ objective }}"),
                    ],
                },
                {
                    "id": "critique",
                    "agents": [
                        configured(
                            "critic-a",
                            "OBJECTIVE:\n{{ objective }}\n\nCANDIDATE B:\n"
                            "{{ artifact.proposal.producer-b }}",
                        ),
                        configured(
                            "critic-b",
                            "OBJECTIVE:\n{{ objective }}\n\nCANDIDATE A:\n"
                            "{{ artifact.proposal.producer-a }}",
                        ),
                    ],
                },
                {
                    "id": "final",
                    "agents": [configured(
                        "reviewer",
                        "OBJECTIVE:\n{{ objective }}\n\n"
                        "PROPOSAL A:\n{{ artifact.proposal.producer-a }}\n\n"
                        "PROPOSAL B:\n{{ artifact.proposal.producer-b }}\n\n"
                        "CRITIQUE OF B:\n{{ artifact.critique.critic-a }}\n\n"
                        "CRITIQUE OF A:\n{{ artifact.critique.critic-b }}",
                    )],
                },
            ],
        })

    def _config_for_run(self, run, requested_config: WorkflowConfig | None) -> WorkflowConfig:
        stored_json = run["config_json"]
        stored = WorkflowConfig.from_json(stored_json) if stored_json else self._default_config()
        if requested_config is not None and requested_config.to_json() != stored.to_json():
            raise ValueError("Supplied configuration does not match the run's stored configuration snapshot")
        return stored

    @staticmethod
    def _validate_runtime_config(config: WorkflowConfig) -> None:
        requirements = {
            "anthropic": ("anthropic", "ANTHROPIC_API_KEY"),
            "openai": ("openai", "OPENAI_API_KEY"),
        }
        providers = {agent.provider for stage in config.stages for agent in stage.agents}
        for provider in sorted(providers):
            if provider == "mock":
                continue
            package, key_name = requirements[provider]
            if importlib.util.find_spec(package) is None:
                raise ValueError(
                    f"Configuration requires provider {provider!r}; install the {provider} package extra"
                )
            if not os.getenv(key_name):
                raise ValueError(f"Configuration requires environment variable {key_name}")

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
        started_at = time.perf_counter()
        retries = 0
        repairs = 0
        input_tokens = 0
        output_tokens = 0
        usage_reported = False
        self.progress(f"[{run_id}] {stage}: starting {agent.agent_id} ({agent.provider}/{agent.model})")
        self.store.event(run_id, stage, "agent_started", {"provider": agent.provider, "model": agent.model}, agent.agent_id)
        for attempt in range(self.max_retries + 1):
            try:
                completion = await bounded_completion(
                    provider_for(agent.provider), agent, prompt,
                    timeout_seconds=self.timeout, max_output_chars=self.max_output,
                )
                input_tokens += completion.input_tokens or 0
                output_tokens += completion.output_tokens or 0
                usage_reported = usage_reported or completion.input_tokens is not None or completion.output_tokens is not None
                break
            except Exception as exc:
                if attempt >= self.max_retries or not self._is_transient(exc):
                    raise
                delay = self.retry_delay * (2 ** attempt)
                retries += 1
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
            result = validate_output(agent.role, completion.content)
        except OutputValidationError as exc:
            repairs = 1
            self.store.event(run_id, stage, "output_repairing", {"error": str(exc)}, agent.agent_id)
            self.progress(f"[{run_id}] {stage}: repairing malformed output from {agent.agent_id}")
            repair_prompt = (
                f"Your previous response failed validation: {exc}.\n"
                "Return only a corrected JSON object matching your required schema. "
                "Do not use Markdown fences. Preserve the substance of the response.\n\n"
                f"PREVIOUS RESPONSE:\n{completion.content}"
            )
            repaired = await bounded_completion(
                provider_for(agent.provider), agent, repair_prompt,
                timeout_seconds=self.timeout, max_output_chars=self.max_output,
            )
            input_tokens += repaired.input_tokens or 0
            output_tokens += repaired.output_tokens or 0
            usage_reported = usage_reported or repaired.input_tokens is not None or repaired.output_tokens is not None
            try:
                result = validate_output(agent.role, repaired.content)
            except OutputValidationError as repair_exc:
                self.store.event(
                    run_id, stage, "output_repair_failed", {"error": str(repair_exc)}, agent.agent_id
                )
                raise OutputValidationError(
                    f"{agent.agent_id} returned invalid output after one repair: {repair_exc}"
                ) from repair_exc
            self.store.event(run_id, stage, "output_repaired", {}, agent.agent_id)
        artifact = self.store.artifact(run_id, stage, agent.agent_id, result)
        duration = time.perf_counter() - started_at
        metrics = {
            "provider": agent.provider,
            "model": agent.model,
            "duration_seconds": round(duration, 6),
            "retries": retries,
            "repairs": repairs,
            "input_tokens": input_tokens if usage_reported else None,
            "output_tokens": output_tokens if usage_reported else None,
        }
        estimated_cost = self._estimated_cost(agent, input_tokens, output_tokens) if usage_reported else None
        if estimated_cost is not None:
            metrics["estimated_cost_usd"] = round(estimated_cost, 8)
        self.store.event(
            run_id, stage, "agent_completed",
            {"artifact_id": artifact.artifact_id, "metrics": metrics}, agent.agent_id,
        )
        self.progress(f"[{run_id}] {stage}: completed {agent.agent_id}")
        return artifact

    @staticmethod
    def _estimated_cost(agent: AgentSpec, input_tokens: int, output_tokens: int) -> float | None:
        agent_prefix = agent.agent_id.upper().replace("-", "_")
        provider_prefix = re.sub(r"[^A-Z0-9]+", "_", agent.provider.upper()).strip("_")
        model_prefix = re.sub(r"[^A-Z0-9]+", "_", agent.model.upper()).strip("_")

        def rate(kind: str) -> str | None:
            names = [
                f"{agent_prefix}_{kind}_COST_PER_MILLION",
                f"MODEL_{model_prefix}_{kind}_COST_PER_MILLION",
                f"PROVIDER_{provider_prefix}_{kind}_COST_PER_MILLION",
                f"AGENT_{kind}_COST_PER_MILLION",
            ]
            return next((os.environ[name] for name in names if name in os.environ), None)

        input_rate = rate("INPUT")
        output_rate = rate("OUTPUT")
        if input_rate is None or output_rate is None:
            return None
        return input_tokens * float(input_rate) / 1_000_000 + output_tokens * float(output_rate) / 1_000_000

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

    def _render_input(
        self,
        template: str,
        objective: str,
        completed: dict[tuple[str, str], Artifact],
        *,
        compact: bool,
        previous: str = "",
    ) -> tuple[str, bool, int]:
        token_pattern = re.compile(
            rf"{OBJECTIVE_PATTERN.pattern}|{PREVIOUS_PATTERN.pattern}|{REFERENCE_PATTERN.pattern}"
        )
        matches = list(token_pattern.finditer(template))
        values = []
        for match in matches:
            if OBJECTIVE_PATTERN.fullmatch(match.group(0)):
                values.append(objective)
            elif PREVIOUS_PATTERN.fullmatch(match.group(0)):
                values.append(previous)
            else:
                reference = REFERENCE_PATTERN.fullmatch(match.group(0))
                slot = (reference.group(1), reference.group(2))
                artifact = completed.get(slot)
                if artifact is None:
                    raise ValueError(f"Missing required artifact {slot[0]}/{slot[1]}")
                values.append(Path(artifact.path).read_text(encoding="utf-8"))

        def render(replacements: list[str]) -> str:
            pieces = []
            position = 0
            for match, replacement in zip(matches, replacements):
                pieces.extend((template[position:match.start()], replacement))
                position = match.end()
            pieces.append(template[position:])
            return "".join(pieces)

        original = render(values)
        if not compact or len(original) <= self.reviewer_input_limit:
            return original, False, len(original)
        fixed_chars = len(render(["" for _ in values]))
        available = max(0, self.reviewer_input_limit - fixed_chars)
        allocations = [0] * len(values)
        active = set(range(len(values)))
        while active:
            share, remainder = divmod(available, len(active))
            satisfied = [index for index in active if len(values[index]) <= share]
            if not satisfied:
                for position, index in enumerate(sorted(active)):
                    allocations[index] = share + (1 if position < remainder else 0)
                break
            for index in satisfied:
                allocations[index] = len(values[index])
                available -= allocations[index]
                active.remove(index)
        rendered = render([
            self._compact_text(value, allocation)
            for value, allocation in zip(values, allocations)
        ])
        return rendered[:self.reviewer_input_limit], True, len(original)

    @staticmethod
    def _condition_met(condition: dict, completed: dict[tuple[str, str], Artifact]) -> bool:
        stage, author = condition["artifact"].split("/", 1)
        artifact = completed.get((stage, author))
        if artifact is None:
            return False
        value = json.loads(Path(artifact.path).read_text(encoding="utf-8"))
        for part in condition["field"].split("."):
            if not isinstance(value, dict) or part not in value:
                return False
            value = value[part]
        return value == condition["equals"]

    @staticmethod
    def _with_iteration_aliases(
        completed: dict[tuple[str, str], Artifact],
    ) -> dict[tuple[str, str], Artifact]:
        result = dict(completed)
        pattern = re.compile(r"^(.+)--iteration-(\d+)$")
        latest = {}
        for (stage, author), artifact in completed.items():
            match = pattern.fullmatch(stage)
            if match:
                key = (match.group(1), author)
                iteration = int(match.group(2))
                if iteration > latest.get(key, (0, None))[0]:
                    latest[key] = (iteration, artifact)
        result.update({key: value[1] for key, value in latest.items()})
        return result

    @staticmethod
    def describe(config: WorkflowConfig) -> str:
        lines = [f"Workflow graph (version {config.version}):"]
        for stage in config.stages:
            deps = ", ".join(stage.depends_on) or "root"
            detail = f"  {stage.stage_id} <- {deps}"
            if stage.when:
                detail += f" [when {stage.when['artifact']}.{stage.when['field']} == {stage.when['equals']!r}]"
            if stage.repeat:
                detail += f" [repeat <= {stage.repeat['max_iterations']} until condition]"
            lines.append(detail)
            for agent in stage.agents:
                lines.append(f"    {agent.agent_id}: {agent.role} ({agent.provider}/{agent.model})")
        return "\n".join(lines)

    async def run(
        self,
        objective: str | None = None,
        *,
        resume_id: str | None = None,
        config: WorkflowConfig | None = None,
    ) -> WorkflowResult:
        if resume_id is not None:
            if objective is not None:
                raise ValueError("Do not supply an objective when resuming a run")
            run_id = resume_id
            run, _, _ = self.store.run_details(run_id)
            effective_config = self._config_for_run(run, config)
            self._validate_runtime_config(effective_config)
            objective = self.store.resume_run(run_id)
            self.progress(f"[{run_id}] resuming run")
        else:
            if objective is None:
                raise ValueError("An objective is required for a new run")
            effective_config = config or self._default_config()
            self._validate_runtime_config(effective_config)
            run_id = self.store.create_run(objective, effective_config.to_json())
            self.progress(f"[{run_id}] created run")
        try:
            completed = self.store.artifacts_for_run(run_id)
            if resume_id is not None:
                for slot in effective_config.slots:
                    artifact = completed.get(slot)
                    if artifact is None:
                        continue
                    self.store.event(
                        run_id,
                        artifact.stage,
                        "artifact_reused",
                        {"artifact_id": artifact.artifact_id},
                        artifact.author,
                    )
                    self.progress(f"[{run_id}] {artifact.stage}: reusing {artifact.author}")
            completed = self._with_iteration_aliases(completed)
            for stage in effective_config.stages:
                if stage.when:
                    execute = self._condition_met(stage.when, completed)
                    self.store.event(
                        run_id, stage.stage_id, "stage_condition_evaluated",
                        {"condition": stage.when, "execute": execute},
                    )
                    if not execute:
                        self.store.event(run_id, stage.stage_id, "stage_skipped", {"condition": stage.when})
                        self.progress(f"[{run_id}] {stage.stage_id}: skipped (condition false)")
                        continue
                maximum = stage.repeat["max_iterations"] if stage.repeat else 1
                for iteration in range(1, maximum + 1):
                    persisted_stage = (
                        f"{stage.stage_id}--iteration-{iteration}" if stage.repeat else stage.stage_id
                    )
                    calls = []
                    for configured_agent in stage.agents:
                        slot = (persisted_stage, configured_agent.agent_id)
                        if slot in completed:
                            continue
                        previous_artifact = completed.get((stage.stage_id, configured_agent.agent_id))
                        previous = (
                            Path(previous_artifact.path).read_text(encoding="utf-8")
                            if previous_artifact else ""
                        )
                        agent = AgentSpec(
                            configured_agent.agent_id,
                            configured_agent.role,
                            configured_agent.provider,
                            configured_agent.model,
                            configured_agent.system_prompt,
                        )
                        prompt, compacted, original_chars = self._render_input(
                            configured_agent.input_template,
                            objective,
                            completed,
                            compact=agent.role == "reviewer",
                            previous=previous,
                        )
                        if compacted:
                            self.store.event(
                                run_id, persisted_stage, "reviewer_input_compacted",
                                {"original_chars": original_chars, "compacted_chars": len(prompt),
                                 "max_chars": self.reviewer_input_limit}, agent.agent_id,
                            )
                        calls.append(self._call(run_id, persisted_stage, agent, prompt))
                    if calls:
                        await self._complete_stage(calls)
                        completed = self._with_iteration_aliases(self.store.artifacts_for_run(run_id))
                    if stage.repeat:
                        stop = self._condition_met(stage.repeat["until"], completed)
                        self.store.event(
                            run_id, stage.stage_id, "loop_iteration_completed",
                            {"iteration": iteration, "stop": stop, "max_iterations": maximum},
                        )
                        if stop:
                            break
            final_stage = effective_config.stages[-1]
            final_agent = final_stage.agents[0]
            final_artifact = completed[(final_stage.stage_id, final_agent.agent_id)]
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
