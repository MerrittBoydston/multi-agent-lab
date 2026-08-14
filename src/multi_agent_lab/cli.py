from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from .workflow import CollaborationWorkflow, WorkflowInterrupted


def _timing_metrics(run, events) -> tuple[float, float, float] | None:
    if not run["completed_at"]:
        return None
    created = datetime.fromisoformat(run["created_at"])
    completed = datetime.fromisoformat(run["completed_at"])
    lifecycle = max(0.0, (completed - created).total_seconds())
    stopped_at = None
    downtime = 0.0
    for event in events:
        event_type = event["event_type"]
        event_time = datetime.fromisoformat(event["created_at"])
        if event_type in {"failed", "interrupted"} and stopped_at is None:
            stopped_at = event_time
        elif event_type == "resumed" and stopped_at is not None:
            downtime += max(0.0, (event_time - stopped_at).total_seconds())
            stopped_at = None
    downtime = min(downtime, lifecycle)
    return lifecycle, lifecycle - downtime, downtime


def _print_runs(workflow: CollaborationWorkflow) -> None:
    rows = workflow.store.list_runs()
    if not rows:
        print("No runs found.")
        return
    print(f"{'RUN ID':12}  {'STATUS':9}  OBJECTIVE")
    for row in rows:
        print(f"{row['run_id']:12}  {row['status']:9}  {row['objective']}")


def _show_run(workflow: CollaborationWorkflow, run_id: str) -> None:
    run, artifacts, events = workflow.store.run_details(run_id)
    completed_metrics = []
    for event in events:
        if event["event_type"] == "agent_completed":
            metrics = json.loads(event["payload_json"]).get("metrics")
            if metrics:
                completed_metrics.append(metrics)
    print(f"Run: {run['run_id']}")
    print(f"Status: {run['status']}")
    print(f"Created: {run['created_at']}")
    print(f"Completed: {run['completed_at'] or '-'}")
    print(f"Objective: {run['objective']}")
    if completed_metrics:
        duration = sum(item["duration_seconds"] for item in completed_metrics)
        retries = sum(item["retries"] for item in completed_metrics)
        repairs = sum(item["repairs"] for item in completed_metrics)
        token_metrics = [item for item in completed_metrics if item["input_tokens"] is not None]
        print("\nMetrics:")
        timing = _timing_metrics(run, events)
        if timing:
            lifecycle, active, downtime = timing
            print(f"  Lifecycle time: {lifecycle:.3f}s")
            print(f"  Active wall time: {active:.3f}s")
            print(f"  Downtime: {downtime:.3f}s")
        print(f"  Agent time: {duration:.3f}s")
        print(f"  Retries: {retries}")
        print(f"  Repairs: {repairs}")
        if token_metrics:
            print(f"  Input tokens: {sum(item['input_tokens'] for item in token_metrics)}")
            print(f"  Output tokens: {sum(item['output_tokens'] for item in token_metrics)}")
            print(f"  Token coverage: {len(token_metrics)}/{len(completed_metrics)} agents")
        costs = [item.get("estimated_cost_usd") for item in completed_metrics]
        if costs and all(cost is not None for cost in costs):
            print(f"  Estimated cost: ${sum(costs):.6f}")
        provider_totals = defaultdict(lambda: {"agents": 0, "input": 0, "output": 0, "cost": 0.0, "cost_complete": True})
        for item in completed_metrics:
            provider = item.get("provider", "unknown")
            model = item.get("model", "unknown")
            total = provider_totals[(provider, model)]
            total["agents"] += 1
            total["input"] += item.get("input_tokens") or 0
            total["output"] += item.get("output_tokens") or 0
            if item.get("estimated_cost_usd") is None:
                total["cost_complete"] = False
            else:
                total["cost"] += item["estimated_cost_usd"]
        print("  Providers/models:")
        for (provider, model), total in sorted(provider_totals.items()):
            usage = f"{total['input']} in / {total['output']} out"
            cost = f", ${total['cost']:.6f}" if total["cost_complete"] else ""
            print(f"    {provider}/{model}: {total['agents']} agents, {usage}{cost}")
    print("\nArtifacts:")
    if not artifacts:
        print("  None")
    for artifact in artifacts:
        print(f"  {artifact['stage']}/{artifact['author']}  {artifact['artifact_id']}  {artifact['path']}")
    print("\nEvents:")
    if not events:
        print("  None")
    for event in events:
        actor = f"/{event['agent_id']}" if event["agent_id"] else ""
        payload = json.loads(event["payload_json"])
        detail = f" {json.dumps(payload, sort_keys=True)}" if payload else ""
        print(f"  {event['created_at']}  {event['stage']}{actor}  {event['event_type']}{detail}")


def _print_resume_plan(workflow: CollaborationWorkflow, run_id: str) -> None:
    reused, missing = workflow.resume_plan(run_id)
    format_slots = lambda slots: ", ".join(f"{stage}/{agent}" for stage, agent in slots) or "none"
    print(f"Resume plan for {run_id}:", file=sys.stderr)
    print(f"  Reuse: {format_slots(reused)}", file=sys.stderr)
    print(f"  Execute: {format_slots(missing)}", file=sys.stderr)


async def _resume_many(workflow: CollaborationWorkflow, run_ids: list[str]) -> list[WorkflowResult]:
    results = []
    failures = []
    for run_id in run_ids:
        _print_resume_plan(workflow, run_id)
        try:
            results.append(await workflow.run(resume_id=run_id))
        except WorkflowInterrupted:
            raise
        except Exception as exc:
            failures.append((run_id, exc))
            print(f"Error resuming {run_id}: {exc}", file=sys.stderr)
    if failures:
        failed_ids = ", ".join(run_id for run_id, _ in failures)
        raise RuntimeError(f"Failed to resume {len(failures)} run(s): {failed_ids}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a controlled multi-agent collaboration")
    parser.add_argument("objective", nargs="?", help="The topic or deliverable to work on")
    parser.add_argument("--workspace", type=Path, default=Path(".multi-agent-lab"))
    parser.add_argument("--resume", metavar="RUN_ID", help="Resume a failed run, executing only missing stages")
    parser.add_argument("--resume-latest", action="store_true", help="Resume the newest failed run")
    parser.add_argument("--retry-failed", action="store_true", help="Resume all failed runs, oldest first")
    parser.add_argument("--list-runs", action="store_true", help="List runs in the workspace")
    parser.add_argument("--show-run", metavar="RUN_ID", help="Show a run's artifacts and audit events")
    args = parser.parse_args()

    selected = sum((
        bool(args.objective), bool(args.resume), args.resume_latest, args.retry_failed,
        args.list_runs, bool(args.show_run),
    ))
    if selected != 1:
        parser.error(
            "provide exactly one of an objective, --resume RUN_ID, --resume-latest, "
            "--retry-failed, --list-runs, or --show-run RUN_ID"
        )

    try:
        workflow = CollaborationWorkflow(args.workspace, progress=lambda message: print(message, file=sys.stderr, flush=True))
        if args.list_runs:
            _print_runs(workflow)
            return
        if args.show_run:
            _show_run(workflow, args.show_run)
            return
        if args.resume_latest:
            failed = workflow.store.failed_run_ids(newest_first=True)
            if not failed:
                raise ValueError("No failed runs found")
            args.resume = failed[0]
        if args.retry_failed:
            failed = workflow.store.failed_run_ids()
            if not failed:
                print("No failed runs found.")
                return
            results = asyncio.run(_resume_many(workflow, failed))
            for result in results:
                print(f"Run: {result.run_id}")
                print(f"Final artifact: {result.final_artifact.path}")
            return
        if args.resume:
            _print_resume_plan(workflow, args.resume)
        result = asyncio.run(workflow.run(args.objective, resume_id=args.resume))
    except WorkflowInterrupted as exc:
        print(f"\nRun {exc.run_id} was interrupted; completed artifacts were preserved.")
        print(
            "Resume with: "
            f"PYTHONPATH=src python3 -m multi_agent_lab.cli "
            f"--workspace {args.workspace} --resume {exc.run_id}"
        )
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if args.resume:
            print(
                "Resume with: "
                f"PYTHONPATH=src python3 -m multi_agent_lab.cli "
                f"--workspace {args.workspace} --resume {args.resume}",
                file=sys.stderr,
            )
        raise SystemExit(1) from None
    print(f"Run: {result.run_id}")
    print(f"Final artifact: {result.final_artifact.path}")


if __name__ == "__main__":
    main()
