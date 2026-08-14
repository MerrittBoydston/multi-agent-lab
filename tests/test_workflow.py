import asyncio
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from multi_agent_lab.config import config_from_mapping, load_config
from multi_agent_lab.models import AgentSpec, CompletionResult
from multi_agent_lab.providers import AnthropicProvider, MockProvider, OpenAIProvider
from multi_agent_lab.workflow import CollaborationWorkflow, WorkflowInterrupted
from multi_agent_lab.validation import (
    OutputValidationError,
    schema_for_anthropic,
    schema_for_role,
    validate_output,
)


class WorkflowTest(unittest.TestCase):
    def test_conditional_bounded_loop_preserves_iterations_and_decisions(self):
        config = load_config(Path(__file__).parents[1] / "examples" / "conditional-workflow.yml")
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder)
            result = asyncio.run(CollaborationWorkflow(workspace).run("Improve it", config=config))
            with closing(sqlite3.connect(workspace / "runs.db")) as db:
                stages = [row[0] for row in db.execute(
                    "SELECT stage FROM artifacts WHERE run_id = ? ORDER BY created_at", (result.run_id,)
                )]
                events = [row[0] for row in db.execute(
                    "SELECT event_type FROM events WHERE run_id = ?", (result.run_id,)
                )]
            self.assertIn("revision--iteration-1", stages)
            self.assertIn("revision--iteration-2", stages)
            self.assertIn("revision--iteration-3", stages)
            self.assertEqual(events.count("loop_iteration_completed"), 3)

    @staticmethod
    def custom_config(model="mock-v1"):
        return config_from_mapping({
            "version": 1,
            "stages": [
                {
                    "id": "draft",
                    "agents": [{
                        "id": "writer",
                        "role": "producer",
                        "provider": "mock",
                        "model": model,
                        "system_prompt": "Write a draft",
                        "input": "Brief: {{ objective }}",
                    }],
                },
                {
                    "id": "decision",
                    "agents": [{
                        "id": "judge",
                        "role": "reviewer",
                        "provider": "mock",
                        "model": model,
                        "system_prompt": "Make a decision",
                        "input": "BRIEF:\n{{ objective }}\n\nDRAFT:\n{{ artifact.draft.writer }}",
                    }],
                },
            ],
        })

    def test_custom_config_executes_and_persists_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder)
            config = self.custom_config()
            result = asyncio.run(
                CollaborationWorkflow(workspace).run("Configurable objective", config=config)
            )
            with closing(sqlite3.connect(workspace / "runs.db")) as db:
                config_json = db.execute(
                    "SELECT config_json FROM runs WHERE run_id = ?", (result.run_id,)
                ).fetchone()[0]
                slots = db.execute(
                    "SELECT stage, author FROM artifacts WHERE run_id = ? ORDER BY created_at",
                    (result.run_id,),
                ).fetchall()
            self.assertEqual(config_json, config.to_json())
            self.assertEqual(slots, [("draft", "writer"), ("decision", "judge")])

    def test_resume_uses_snapshot_and_rejects_incompatible_config(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder)
            workflow = CollaborationWorkflow(workspace)
            config = self.custom_config()
            original_complete = MockProvider.complete

            async def fail_judge(provider, agent, prompt):
                if agent.agent_id == "judge":
                    raise RuntimeError("stop after draft")
                return await original_complete(provider, agent, prompt)

            with patch.object(MockProvider, "complete", fail_judge):
                with self.assertRaisesRegex(RuntimeError, "stop after draft"):
                    asyncio.run(workflow.run("Snapshot me", config=config))
            run_id = workflow.store.failed_run_ids()[0]

            with self.assertRaisesRegex(ValueError, "does not match"):
                asyncio.run(workflow.run(resume_id=run_id, config=self.custom_config("other-model")))
            result = asyncio.run(workflow.run(resume_id=run_id))
            self.assertEqual(result.run_id, run_id)

    def test_runtime_requirements_are_checked_before_run_creation(self):
        config = self.custom_config()
        raw = json.loads(config.to_json())
        raw["stages"][0]["agents"][0]["provider"] = "anthropic"
        config = config_from_mapping(raw)
        with tempfile.TemporaryDirectory() as folder, patch.dict("os.environ", {}, clear=True), patch(
            "multi_agent_lab.workflow.importlib.util.find_spec", return_value=object()
        ):
            workflow = CollaborationWorkflow(Path(folder))
            with self.assertRaisesRegex(ValueError, "ANTHROPIC_API_KEY"):
                asyncio.run(workflow.run("Missing credentials", config=config))
            self.assertEqual(workflow.store.list_runs(), [])

    def test_existing_database_is_migrated_for_config_snapshots(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder)
            workspace.mkdir(exist_ok=True)
            with closing(sqlite3.connect(workspace / "runs.db")) as db:
                db.execute(
                    "CREATE TABLE runs (run_id TEXT PRIMARY KEY, objective TEXT NOT NULL, "
                    "status TEXT NOT NULL, created_at TEXT NOT NULL, completed_at TEXT)"
                )
                db.commit()
            CollaborationWorkflow(workspace)
            with closing(sqlite3.connect(workspace / "runs.db")) as db:
                columns = {row[1] for row in db.execute("PRAGMA table_info(runs)")}
            self.assertIn("config_json", columns)

    def test_mock_workflow_creates_auditable_artifacts(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict("os.environ", {"AGENT_PROVIDER": "mock"}):
            workspace = Path(folder)
            result = asyncio.run(CollaborationWorkflow(workspace).run("Design a local knowledge assistant"))

            final = json.loads(Path(result.final_artifact.path).read_text(encoding="utf-8"))
            self.assertEqual(final["decision"], "merge")

            with closing(sqlite3.connect(workspace / "runs.db")) as db:
                status = db.execute("SELECT status FROM runs WHERE run_id = ?", (result.run_id,)).fetchone()[0]
                artifacts = db.execute("SELECT COUNT(*) FROM artifacts WHERE run_id = ?", (result.run_id,)).fetchone()[0]
            self.assertEqual(status, "completed")
            self.assertEqual(artifacts, 5)

    def test_failed_run_resumes_without_repeating_completed_agents(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict("os.environ", {"AGENT_PROVIDER": "mock"}):
            workspace = Path(folder)
            workflow = CollaborationWorkflow(workspace)
            original_complete = MockProvider.complete
            failed_once = False

            async def fail_one_critic(provider, agent, prompt):
                nonlocal failed_once
                if agent.agent_id == "critic-b" and not failed_once:
                    failed_once = True
                    raise RuntimeError("injected failure")
                return await original_complete(provider, agent, prompt)

            with patch.object(MockProvider, "complete", fail_one_critic):
                with self.assertRaisesRegex(RuntimeError, "injected failure"):
                    asyncio.run(workflow.run("Design a resumable workflow"))

            with closing(sqlite3.connect(workspace / "runs.db")) as db:
                run_id = db.execute("SELECT run_id FROM runs").fetchone()[0]
                before = db.execute(
                    "SELECT stage, author, artifact_id FROM artifacts WHERE run_id = ?", (run_id,)
                ).fetchall()
            self.assertEqual({(row[0], row[1]) for row in before}, {
                ("proposal", "producer-a"), ("proposal", "producer-b"), ("critique", "critic-a")
            })

            result = asyncio.run(workflow.run(resume_id=run_id))
            self.assertEqual(result.run_id, run_id)
            with closing(sqlite3.connect(workspace / "runs.db")) as db:
                after = db.execute(
                    "SELECT stage, author, artifact_id FROM artifacts WHERE run_id = ?", (run_id,)
                ).fetchall()
                status = db.execute("SELECT status FROM runs WHERE run_id = ?", (run_id,)).fetchone()[0]
            self.assertEqual(status, "completed")
            self.assertEqual(len(after), 5)
            self.assertTrue(set(before).issubset(set(after)))
            with closing(sqlite3.connect(workspace / "runs.db")) as db:
                event_types = [row[0] for row in db.execute(
                    "SELECT event_type FROM events WHERE run_id = ?", (run_id,)
                )]
            self.assertIn("resumed", event_types)
            self.assertEqual(event_types.count("artifact_reused"), 3)

    def test_resume_rejects_unknown_and_completed_runs(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict("os.environ", {"AGENT_PROVIDER": "mock"}):
            workflow = CollaborationWorkflow(Path(folder))
            with self.assertRaisesRegex(ValueError, "does not exist"):
                asyncio.run(workflow.run(resume_id="missing"))
            result = asyncio.run(workflow.run("Complete me"))
            with self.assertRaisesRegex(ValueError, "already completed"):
                asyncio.run(workflow.run(resume_id=result.run_id))

    def test_interruption_marks_run_failed_and_preserves_resume_id(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict("os.environ", {"AGENT_PROVIDER": "mock"}):
            workspace = Path(folder)
            workflow = CollaborationWorkflow(workspace)

            async def interrupt(*args, **kwargs):
                raise asyncio.CancelledError

            with patch.object(workflow, "_call", interrupt):
                with self.assertRaises(WorkflowInterrupted) as raised:
                    asyncio.run(workflow.run("Interrupt me"))

            with closing(sqlite3.connect(workspace / "runs.db")) as db:
                status = db.execute(
                    "SELECT status FROM runs WHERE run_id = ?", (raised.exception.run_id,)
                ).fetchone()[0]
                events = [row[0] for row in db.execute(
                    "SELECT event_type FROM events WHERE run_id = ?", (raised.exception.run_id,)
                )]
            self.assertEqual(status, "failed")
            self.assertIn("interrupted", events)

    def test_artifacts_are_unique_per_stage_and_author(self):
        with tempfile.TemporaryDirectory() as folder:
            workflow = CollaborationWorkflow(Path(folder))
            run_id = workflow.store.create_run("Unique artifacts")
            workflow.store.artifact(run_id, "proposal", "producer-a", "first")
            with self.assertRaises(sqlite3.IntegrityError):
                workflow.store.artifact(run_id, "proposal", "producer-a", "second")

    def test_transient_provider_errors_are_retried_and_audited(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict("os.environ", {
            "AGENT_PROVIDER": "mock", "AGENT_MAX_RETRIES": "2", "AGENT_RETRY_DELAY_SECONDS": "0"
        }):
            workspace = Path(folder)
            workflow = CollaborationWorkflow(workspace)
            original_complete = MockProvider.complete
            attempts = 0

            class APIConnectionError(Exception):
                pass

            async def fail_then_succeed(provider, agent, prompt):
                nonlocal attempts
                if agent.agent_id == "reviewer" and attempts < 2:
                    attempts += 1
                    raise APIConnectionError("temporary disconnect")
                return await original_complete(provider, agent, prompt)

            with patch.object(MockProvider, "complete", fail_then_succeed):
                result = asyncio.run(workflow.run("Retry transient failures"))

            with closing(sqlite3.connect(workspace / "runs.db")) as db:
                retries = db.execute(
                    "SELECT COUNT(*) FROM events WHERE run_id = ? AND event_type = 'agent_retrying'",
                    (result.run_id,),
                ).fetchone()[0]
            self.assertEqual(attempts, 2)
            self.assertEqual(retries, 2)

    def test_permanent_provider_errors_are_not_retried(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict("os.environ", {
            "AGENT_PROVIDER": "mock", "AGENT_MAX_RETRIES": "3", "AGENT_RETRY_DELAY_SECONDS": "0"
        }):
            workflow = CollaborationWorkflow(Path(folder))
            attempts = 0

            async def fail_permanently(provider, agent, prompt):
                nonlocal attempts
                attempts += 1
                raise ValueError("bad request")

            with patch.object(MockProvider, "complete", fail_permanently):
                with self.assertRaisesRegex(ValueError, "bad request"):
                    asyncio.run(workflow.run("Do not retry permanent failures"))
            self.assertEqual(attempts, 2)

    def test_anthropic_provider_streams_long_responses(self):
        class FakeStream:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def get_final_message(self):
                return SimpleNamespace(
                    content=[SimpleNamespace(type="text", text="streamed response")],
                    usage=SimpleNamespace(input_tokens=12, output_tokens=3),
                )

        class FakeMessages:
            def stream(self, **kwargs):
                self.kwargs = kwargs
                return FakeStream()

        class FakeClient:
            def __init__(self):
                self.messages = FakeMessages()

        agent = AgentSpec("reviewer", "reviewer", "anthropic", "test-model", "system")
        client = FakeClient()
        with patch("anthropic.AsyncAnthropic", return_value=client):
            result = asyncio.run(AnthropicProvider().complete(agent, "prompt"))
        self.assertEqual(result.content, "streamed response")
        self.assertEqual(result.input_tokens, 12)
        self.assertEqual(result.output_tokens, 3)
        output_format = client.messages.kwargs["output_config"]["format"]
        self.assertEqual(output_format["type"], "json_schema")
        self.assertEqual(output_format["schema"], schema_for_anthropic("reviewer"))
        confidence = output_format["schema"]["properties"]["confidence"]
        self.assertNotIn("minimum", confidence)
        self.assertNotIn("maximum", confidence)

    def test_openai_provider_uses_strict_structured_output(self):
        class FakeResponses:
            async def create(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(
                    output_text='{"findings":[],"recommendation":"approve","confidence":1}',
                    usage=SimpleNamespace(input_tokens=10, output_tokens=4),
                )

        class FakeClient:
            def __init__(self):
                self.responses = FakeResponses()

        client = FakeClient()
        agent = AgentSpec("critic-a", "critic", "openai", "test-model", "system")
        with patch("openai.AsyncOpenAI", return_value=client):
            result = asyncio.run(OpenAIProvider().complete(agent, "prompt"))
        self.assertEqual(result.input_tokens, 10)
        output_format = client.responses.kwargs["text"]["format"]
        self.assertEqual(output_format["type"], "json_schema")
        self.assertTrue(output_format["strict"])
        self.assertEqual(output_format["schema"], schema_for_role("critic"))
        confidence = output_format["schema"]["properties"]["confidence"]
        self.assertEqual(confidence["minimum"], 0)
        self.assertEqual(confidence["maximum"], 1)

    def test_role_schemas_are_closed_and_fully_required(self):
        for role in ("producer", "critic", "reviewer"):
            schema = schema_for_role(role)
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(set(schema["required"]), set(schema["properties"]))

    def test_anthropic_schema_keeps_local_range_constraint_unchanged(self):
        anthropic_schema = schema_for_anthropic("producer")
        local_schema = schema_for_role("producer")
        self.assertNotIn("minimum", anthropic_schema["properties"]["confidence"])
        self.assertEqual(local_schema["properties"]["confidence"]["minimum"], 0)

    def test_large_reviewer_input_is_compacted_and_audited(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict("os.environ", {
            "AGENT_PROVIDER": "mock", "REVIEWER_MAX_INPUT_CHARS": "1000"
        }):
            workspace = Path(folder)
            workflow = CollaborationWorkflow(workspace)
            original_complete = MockProvider.complete
            reviewer_prompt = None

            async def produce_large_artifacts(provider, agent, prompt):
                nonlocal reviewer_prompt
                large_text = f"START-{agent.agent_id}-" + (agent.agent_id * 300) + f"-END-{agent.agent_id}"
                if agent.role == "producer":
                    return json.dumps({
                        "proposal": large_text,
                        "assumptions": [],
                        "risks": [],
                        "confidence": 0.5,
                    })
                if agent.role == "critic":
                    return json.dumps({
                        "findings": [large_text],
                        "recommendation": "revise",
                        "confidence": 0.5,
                    })
                reviewer_prompt = prompt
                return await original_complete(provider, agent, prompt)

            with patch.object(MockProvider, "complete", produce_large_artifacts):
                result = asyncio.run(workflow.run("Preserve every oversized section"))

            self.assertIsNotNone(reviewer_prompt)
            self.assertLessEqual(len(reviewer_prompt), 1000)
            for label in ("OBJECTIVE", "PROPOSAL A", "PROPOSAL B", "CRITIQUE OF B", "CRITIQUE OF A"):
                self.assertIn(label, reviewer_prompt)
            self.assertIn("...[compacted]...", reviewer_prompt)
            with closing(sqlite3.connect(workspace / "runs.db")) as db:
                payload = db.execute(
                    "SELECT payload_json FROM events WHERE run_id = ? "
                    "AND event_type = 'reviewer_input_compacted'",
                    (result.run_id,),
                ).fetchone()
            self.assertIsNotNone(payload)
            details = json.loads(payload[0])
            self.assertGreater(details["original_chars"], details["compacted_chars"])
            self.assertLessEqual(details["compacted_chars"], details["max_chars"])

    def test_reviewer_input_under_budget_is_unchanged(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict("os.environ", {
            "REVIEWER_MAX_INPUT_CHARS": "1000"
        }):
            workflow = CollaborationWorkflow(Path(folder))
            sections = [("ONE", "alpha"), ("TWO", "omega")]
            result, compacted = workflow._build_judge_input(sections)
            self.assertFalse(compacted)
            self.assertEqual(result, "ONE:\nalpha\n\nTWO:\nomega")

    def test_malformed_output_is_repaired_once_and_audited(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict("os.environ", {"AGENT_PROVIDER": "mock"}):
            workspace = Path(folder)
            workflow = CollaborationWorkflow(workspace)
            original_complete = MockProvider.complete
            malformed_once = True

            async def malformed_then_repaired(provider, agent, prompt):
                nonlocal malformed_once
                if agent.agent_id == "producer-a" and malformed_once:
                    malformed_once = False
                    return "not json"
                return await original_complete(provider, agent, prompt)

            with patch.object(MockProvider, "complete", malformed_then_repaired):
                result = asyncio.run(workflow.run("Repair malformed output"))

            with closing(sqlite3.connect(workspace / "runs.db")) as db:
                events = [row[0] for row in db.execute(
                    "SELECT event_type FROM events WHERE run_id = ? AND agent_id = 'producer-a'",
                    (result.run_id,),
                )]
            self.assertIn("output_repairing", events)
            self.assertIn("output_repaired", events)

    def test_invalid_repair_fails_without_writing_artifact(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict("os.environ", {"AGENT_PROVIDER": "mock"}):
            workspace = Path(folder)
            workflow = CollaborationWorkflow(workspace)

            async def always_invalid(provider, agent, prompt):
                return "still not json"

            with patch.object(MockProvider, "complete", always_invalid):
                with self.assertRaisesRegex(OutputValidationError, "after one repair"):
                    asyncio.run(workflow.run("Reject invalid repair"))

            with closing(sqlite3.connect(workspace / "runs.db")) as db:
                artifact_count = db.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
                repair_failures = db.execute(
                    "SELECT COUNT(*) FROM events WHERE event_type = 'output_repair_failed'"
                ).fetchone()[0]
            self.assertEqual(artifact_count, 0)
            self.assertEqual(repair_failures, 2)

    def test_validation_accepts_fenced_json_and_normalizes_it(self):
        output = validate_output(
            "critic",
            '```json\n{"findings": [], "recommendation": "approve", "confidence": 1}\n```',
        )
        self.assertEqual(json.loads(output)["recommendation"], "approve")

    def test_validation_rejects_wrong_schema(self):
        with self.assertRaisesRegex(OutputValidationError, "confidence"):
            validate_output(
                "reviewer",
                '{"decision":"merge","final":"x","unresolved_questions":[],"confidence":2}',
            )

    def test_completed_events_include_usage_latency_and_cost(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict("os.environ", {
            "AGENT_PROVIDER": "mock",
            "AGENT_INPUT_COST_PER_MILLION": "2",
            "AGENT_OUTPUT_COST_PER_MILLION": "8",
        }):
            workspace = Path(folder)
            workflow = CollaborationWorkflow(workspace)
            original_complete = MockProvider.complete

            async def with_usage(provider, agent, prompt):
                result = await original_complete(provider, agent, prompt)
                return CompletionResult(result.content, input_tokens=100, output_tokens=25)

            with patch.object(MockProvider, "complete", with_usage):
                result = asyncio.run(workflow.run("Measure this run"))

            with closing(sqlite3.connect(workspace / "runs.db")) as db:
                payloads = [json.loads(row[0]) for row in db.execute(
                    "SELECT payload_json FROM events WHERE run_id = ? AND event_type = 'agent_completed'",
                    (result.run_id,),
                )]
            self.assertEqual(len(payloads), 5)
            for payload in payloads:
                metrics = payload["metrics"]
                self.assertEqual(metrics["provider"], "mock")
                self.assertEqual(metrics["model"], "mock-v1")
                self.assertGreaterEqual(metrics["duration_seconds"], 0)
                self.assertEqual(metrics["input_tokens"], 100)
                self.assertEqual(metrics["output_tokens"], 25)
                self.assertEqual(metrics["retries"], 0)
                self.assertEqual(metrics["repairs"], 0)
                self.assertAlmostEqual(metrics["estimated_cost_usd"], 0.0004)

    def test_cost_rate_precedence(self):
        agent = AgentSpec("reviewer", "reviewer", "anthropic", "claude-sonnet-4-6", "system")
        with patch.dict("os.environ", {
            "AGENT_INPUT_COST_PER_MILLION": "1",
            "AGENT_OUTPUT_COST_PER_MILLION": "1",
            "PROVIDER_ANTHROPIC_INPUT_COST_PER_MILLION": "2",
            "PROVIDER_ANTHROPIC_OUTPUT_COST_PER_MILLION": "2",
            "MODEL_CLAUDE_SONNET_4_6_INPUT_COST_PER_MILLION": "3",
            "MODEL_CLAUDE_SONNET_4_6_OUTPUT_COST_PER_MILLION": "3",
            "REVIEWER_INPUT_COST_PER_MILLION": "4",
            "REVIEWER_OUTPUT_COST_PER_MILLION": "4",
        }, clear=True):
            self.assertAlmostEqual(
                CollaborationWorkflow._estimated_cost(agent, 1_000_000, 1_000_000),
                8.0,
            )

    def test_provider_cost_rate_falls_back_before_global(self):
        agent = AgentSpec("reviewer", "reviewer", "anthropic", "claude-sonnet-4-6", "system")
        with patch.dict("os.environ", {
            "AGENT_INPUT_COST_PER_MILLION": "1",
            "AGENT_OUTPUT_COST_PER_MILLION": "1",
            "PROVIDER_ANTHROPIC_INPUT_COST_PER_MILLION": "2",
            "PROVIDER_ANTHROPIC_OUTPUT_COST_PER_MILLION": "3",
        }, clear=True):
            self.assertAlmostEqual(
                CollaborationWorkflow._estimated_cost(agent, 1_000_000, 1_000_000),
                5.0,
            )


if __name__ == "__main__":
    unittest.main()
