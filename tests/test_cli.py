import os
import subprocess
import sys
import tempfile
import unittest
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
