from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from .config import WorkflowConfig, load_config
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
    print(f"Configuration: {'stored snapshot' if run['config_json'] else 'legacy default'}")
    controls = workflow.store.controls(run_id)
    if any(controls[name] is not None for name in ("max_tokens", "max_cost", "deadline")):
        print("Controls:")
        print(f"  Token budget: {controls['used_tokens']}/{controls['max_tokens'] or '-'}")
        print(f"  Cost budget: ${controls['used_cost']:.6f}/{controls['max_cost'] if controls['max_cost'] is not None else '-'}")
        print(f"  Deadline: {controls['deadline'] or '-'}")
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


def _print_resume_plan(
    workflow: CollaborationWorkflow,
    run_id: str,
    config: WorkflowConfig | None = None,
) -> None:
    reused, missing = workflow.resume_plan(run_id, config)
    format_slots = lambda slots: ", ".join(f"{stage}/{agent}" for stage, agent in slots) or "none"
    print(f"Resume plan for {run_id}:", file=sys.stderr)
    print(f"  Reuse: {format_slots(reused)}", file=sys.stderr)
    print(f"  Execute: {format_slots(missing)}", file=sys.stderr)
    controls = workflow.store.controls(run_id)
    if controls["max_tokens"] is not None:
        print(f"  Tokens remaining: {controls['max_tokens'] - controls['used_tokens']}", file=sys.stderr)
    if controls["max_cost"] is not None:
        print(f"  Cost remaining: ${controls['max_cost'] - controls['used_cost']:.6f}", file=sys.stderr)


async def _resume_many(
    workflow: CollaborationWorkflow,
    run_ids: list[str],
    config: WorkflowConfig | None = None,
    controls: dict | None = None,
) -> list[WorkflowResult]:
    results = []
    failures = []
    for run_id in run_ids:
        _print_resume_plan(workflow, run_id, config)
        try:
            results.append(await workflow.run(resume_id=run_id, config=config, controls=controls))
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
    parser.add_argument("--config", type=Path, help="YAML workflow configuration for a new run or resume verification")
    parser.add_argument("--dry-run", action="store_true", help="Validate and display a configuration without calling models")
    parser.add_argument("--resume", metavar="RUN_ID", help="Resume a failed run, executing only missing stages")
    parser.add_argument("--resume-latest", action="store_true", help="Resume the newest failed run")
    parser.add_argument("--retry-failed", action="store_true", help="Resume all failed runs, oldest first")
    parser.add_argument("--list-runs", action="store_true", help="List runs in the workspace")
    parser.add_argument("--show-run", metavar="RUN_ID", help="Show a run's artifacts and audit events")
    parser.add_argument("--cancel", metavar="RUN_ID", help="Request cancellation of a resumable run")
    parser.add_argument("--max-tokens", type=int, help="Maximum reserved and consumed tokens")
    parser.add_argument("--max-cost", type=float, help="Maximum estimated cost in USD")
    parser.add_argument("--deadline", help="Absolute ISO-8601 UTC deadline")
    args = parser.parse_args()

    selected = sum((
        bool(args.objective), bool(args.resume), args.resume_latest, args.retry_failed,
        args.list_runs, bool(args.show_run), args.dry_run, bool(args.cancel),
    ))
    if selected != 1:
        parser.error(
            "provide exactly one of an objective, --resume RUN_ID, --resume-latest, "
            "--retry-failed, --list-runs, --show-run RUN_ID, --cancel RUN_ID, or --dry-run"
        )
    if args.max_tokens is not None and args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.max_cost is not None and args.max_cost <= 0:
        parser.error("--max-cost must be positive")
    if args.deadline:
        try:
            deadline = datetime.fromisoformat(args.deadline.replace("Z", "+00:00"))
            if deadline.tzinfo is None:
                raise ValueError
            args.deadline = deadline.astimezone(UTC).isoformat()
        except ValueError:
            parser.error("--deadline must be an ISO-8601 timestamp with a timezone")
    if args.dry_run and not args.config:
        parser.error("--dry-run requires --config PATH")
    if args.config and (args.list_runs or args.show_run):
        parser.error("--config can only be used when starting or resuming runs")

    try:
        workflow = CollaborationWorkflow(args.workspace, progress=lambda message: print(message, file=sys.stderr, flush=True))
        config = load_config(args.config) if args.config else None
        controls = {"max_tokens": args.max_tokens, "max_cost": args.max_cost, "deadline": args.deadline}
        if args.cancel:
            workflow.store.cancel(args.cancel)
            print(f"Cancelled run: {args.cancel}")
            return
        if args.dry_run:
            print(workflow.describe(config))
            return
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
            results = asyncio.run(_resume_many(workflow, failed, config, controls))
            for result in results:
                print(f"Run: {result.run_id}")
                print(f"Final artifact: {result.final_artifact.path}")
            return
        if args.resume:
            _print_resume_plan(workflow, args.resume, config)
        result = asyncio.run(workflow.run(args.objective, resume_id=args.resume, config=config, controls=controls))
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
