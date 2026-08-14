import os
import subprocess
import sys
import tempfile
import unittest
import sqlite3
import json
import zipfile
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

from multi_agent_lab.cli import _timing_metrics


class CliTest(unittest.TestCase):
    def run_cli(self, workspace: Path, *args: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.update({"PYTHONPATH": "src", "AGENT_PROVIDER": "mock"})
        return subprocess.run(
            [sys.executable, "-m", "multi_agent_lab.cli", "--workspace", str(workspace), *args],
            cwd=Path(__file__).parents[1],
            env=environment,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )

    def test_run_list_and_show(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder)
            created = self.run_cli(workspace, "CLI integration test")
            self.assertEqual(created.returncode, 0, created.stderr)
            self.assertIn("proposal: starting producer-a", created.stderr)
            self.assertIn("workflow completed", created.stderr)
            run_id = next(
                line.removeprefix("Run: ")
                for line in created.stdout.splitlines()
                if line.startswith("Run: ")
            )

            listed = self.run_cli(workspace, "--list-runs")
            self.assertEqual(listed.returncode, 0, listed.stderr)
            self.assertIn(run_id, listed.stdout)
            self.assertIn("completed", listed.stdout)

            shown = self.run_cli(workspace, "--show-run", run_id)
            self.assertEqual(shown.returncode, 0, shown.stderr)
            self.assertIn(f"Run: {run_id}", shown.stdout)
            self.assertIn("final/reviewer", shown.stdout)
            self.assertIn("agent_completed", shown.stdout)
            self.assertIn("Metrics:", shown.stdout)
            self.assertIn("Lifecycle time:", shown.stdout)
            self.assertIn("Active wall time:", shown.stdout)
            self.assertIn("Downtime: 0.000s", shown.stdout)
            self.assertIn("Agent time:", shown.stdout)
            self.assertIn("Retries: 0", shown.stdout)
            self.assertIn("Providers/models:", shown.stdout)
            self.assertIn("mock/mock-v1: 5 agents", shown.stdout)

    def test_runs_with_yaml_config_and_reports_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder)
            config = Path(__file__).parents[1] / "examples" / "workflow.yml"
            created = self.run_cli(workspace, "--config", str(config), "Configured run")
            self.assertEqual(created.returncode, 0, created.stderr)
            run_id = next(
                line.removeprefix("Run: ")
                for line in created.stdout.splitlines()
                if line.startswith("Run: ")
            )
            shown = self.run_cli(workspace, "--show-run", run_id)
            self.assertEqual(shown.returncode, 0, shown.stderr)
            self.assertIn("Configuration: stored snapshot", shown.stdout)

    def test_dry_run_validates_and_displays_graph_without_workspace_run(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder)
            config = Path(__file__).parents[1] / "examples" / "conditional-workflow.yml"
            result = self.run_cli(workspace, "--config", str(config), "--dry-run")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Workflow graph (version 2)", result.stdout)
            self.assertIn("revision <- critique", result.stdout)
            self.assertEqual(self.run_cli(workspace, "--list-runs").stdout, "No runs found.\n")

    def test_errors_are_concise(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_cli(Path(folder), "--resume", "missing")
            self.assertEqual(result.returncode, 1)
            self.assertIn("Error: Run 'missing' does not exist", result.stderr)
            self.assertIn("Resume with:", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_show_unknown_run_is_concise(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_cli(Path(folder), "--show-run", "missing")
            self.assertEqual(result.returncode, 1)
            self.assertIn("Error: Run 'missing' does not exist", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_export_import_report_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            destination = root / "destination"
            bundle = root / "run.zip"
            created = self.run_cli(source, "Portable run")
            self.assertEqual(created.returncode, 0, created.stderr)
            run_id = next(line.removeprefix("Run: ") for line in created.stdout.splitlines()
                          if line.startswith("Run: "))

            exported = self.run_cli(source, "--export-run", run_id, "--output", str(bundle))
            self.assertEqual(exported.returncode, 0, exported.stderr)
            self.assertTrue(bundle.is_file())
            imported = self.run_cli(destination, "--import-run", str(bundle))
            self.assertEqual(imported.returncode, 0, imported.stderr)
            self.assertIn(f"Imported run: {run_id}", imported.stdout)

            shown = self.run_cli(destination, "--show-run", run_id)
            self.assertEqual(shown.returncode, 0, shown.stderr)
            self.assertIn("Objective: Portable run", shown.stdout)
            self.assertIn("final/reviewer", shown.stdout)
            report = self.run_cli(destination, "--report", run_id)
            self.assertEqual(report.returncode, 0, report.stderr)
            self.assertIn(f"# Run {run_id}", report.stdout)
            self.assertIn("## Metrics", report.stdout)
            self.assertIn("## Audit summary", report.stdout)

            duplicate = self.run_cli(destination, "--import-run", str(bundle))
            self.assertEqual(duplicate.returncode, 1)
            self.assertIn("already exists", duplicate.stderr)

    def test_import_rejects_tampered_artifact_without_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            destination = root / "destination"
            bundle = root / "run.zip"
            tampered = root / "tampered.zip"
            created = self.run_cli(source, "Tamper test")
            run_id = next(line.removeprefix("Run: ") for line in created.stdout.splitlines()
                          if line.startswith("Run: "))
            self.assertEqual(self.run_cli(source, "--export-run", run_id, "--output", str(bundle)).returncode, 0)
            with zipfile.ZipFile(bundle) as original, zipfile.ZipFile(tampered, "w") as changed:
                for info in original.infolist():
                    content = original.read(info.filename)
                    if info.filename.startswith("artifacts/"):
                        content += b"modified"
                    changed.writestr(info.filename, content)
            result = self.run_cli(destination, "--import-run", str(tampered))
            self.assertEqual(result.returncode, 1)
            self.assertIn("checksum does not match", result.stderr)
            self.assertEqual(self.run_cli(destination, "--list-runs").stdout, "No runs found.\n")

    def test_import_rejects_unsafe_or_undeclared_archive_member(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bundle = root / "unsafe.zip"
            manifest = {"schema_version": 1, "run": {}, "events": [], "artifacts": []}
            manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
            import hashlib
            with zipfile.ZipFile(bundle, "w") as archive:
                archive.writestr("manifest.json", manifest_bytes)
                archive.writestr("manifest.sha256", hashlib.sha256(manifest_bytes).hexdigest())
                archive.writestr("../escape", b"bad")
            result = self.run_cli(root / "workspace", "--import-run", str(bundle))
            self.assertEqual(result.returncode, 1)
            self.assertIn("unsafe archive path", result.stderr)

    def create_failed_run(self, workspace: Path, objective: str, completed_slots=()) -> str:
        from multi_agent_lab.workflow import CollaborationWorkflow

        workflow = CollaborationWorkflow(workspace)
        run_id = workflow.store.create_run(objective)
        for stage, author in completed_slots:
            workflow.store.artifact(run_id, stage, author, "{}")
        workflow.store.finish(run_id, "failed")
        return run_id

    def test_resume_latest_selects_newest_failed_run_and_prints_plan(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder)
            older = self.create_failed_run(workspace, "Older")
            newer = self.create_failed_run(
                workspace, "Newer", (("proposal", "producer-a"),)
            )

            result = self.run_cli(workspace, "--resume-latest")

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"Resume plan for {newer}:", result.stderr)
            self.assertIn("Reuse: proposal/producer-a", result.stderr)
            self.assertIn("Execute: proposal/producer-b", result.stderr)
            self.assertIn(f"Run: {newer}", result.stdout)
            with closing(sqlite3.connect(workspace / "runs.db")) as db:
                statuses = dict(db.execute("SELECT run_id, status FROM runs"))
            self.assertEqual(statuses[older], "failed")
            self.assertEqual(statuses[newer], "completed")

    def test_retry_failed_resumes_all_oldest_first(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder)
            first = self.create_failed_run(workspace, "First")
            second = self.create_failed_run(workspace, "Second")

            result = self.run_cli(workspace, "--retry-failed")

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertLess(result.stderr.index(first), result.stderr.index(second))
            self.assertIn(f"Run: {first}", result.stdout)
            self.assertIn(f"Run: {second}", result.stdout)
            with closing(sqlite3.connect(workspace / "runs.db")) as db:
                self.assertEqual(
                    db.execute("SELECT COUNT(*) FROM runs WHERE status = 'failed'").fetchone()[0], 0
                )

    def test_automatic_resume_handles_empty_failed_set(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder)
            latest = self.run_cli(workspace, "--resume-latest")
            retry_all = self.run_cli(workspace, "--retry-failed")
            self.assertEqual(latest.returncode, 1)
            self.assertIn("Error: No failed runs found", latest.stderr)
            self.assertEqual(retry_all.returncode, 0)
            self.assertEqual(retry_all.stdout, "No failed runs found.\n")

    def test_cli_budget_exhaustion_and_cancellation_are_resumable(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder)
            exhausted = self.run_cli(workspace, "--max-tokens", "1", "Budget me")
            self.assertEqual(exhausted.returncode, 1)
            listed = self.run_cli(workspace, "--list-runs")
            self.assertIn("budget_exhausted", listed.stdout)
            run_id = listed.stdout.splitlines()[1].split()[0]
            cancelled = self.run_cli(workspace, "--cancel", run_id)
            self.assertEqual(cancelled.returncode, 0, cancelled.stderr)
            self.assertIn("Cancelled run", cancelled.stdout)

    def test_timing_metrics_exclude_failed_to_resumed_downtime(self):
        start = datetime(2026, 1, 1, tzinfo=UTC)
        run = {
            "created_at": start.isoformat(),
            "completed_at": (start + timedelta(seconds=120)).isoformat(),
        }
        events = [
            {"event_type": "failed", "created_at": (start + timedelta(seconds=10)).isoformat()},
            {"event_type": "resumed", "created_at": (start + timedelta(seconds=80)).isoformat()},
        ]
        self.assertEqual(_timing_metrics(run, events), (120.0, 50.0, 70.0))

    def test_timing_metrics_handle_multiple_interruptions(self):
        start = datetime(2026, 1, 1, tzinfo=UTC)
        run = {
            "created_at": start.isoformat(),
            "completed_at": (start + timedelta(seconds=100)).isoformat(),
        }
        events = [
            {"event_type": "interrupted", "created_at": (start + timedelta(seconds=10)).isoformat()},
            {"event_type": "resumed", "created_at": (start + timedelta(seconds=30)).isoformat()},
            {"event_type": "failed", "created_at": (start + timedelta(seconds=50)).isoformat()},
            {"event_type": "resumed", "created_at": (start + timedelta(seconds=90)).isoformat()},
        ]
        self.assertEqual(_timing_metrics(run, events), (100.0, 40.0, 60.0))


if __name__ == "__main__":
    unittest.main()
