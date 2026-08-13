# Multi-Agent Lab

A small local reference implementation of a controlled multi-agent environment:

1. Two independent producers create proposals in parallel.
2. Two critics cross-review the proposals in parallel.
3. A separate reviewer synthesizes the final result.
4. SQLite records the run and event history.
5. Every output is saved as a versioned artifact.

The workflow is deterministic; the agents inside each step can come from different providers.

## Step 1 — run it safely with mock agents

No API key or package installation is needed:

```bash
cd multi-agent-lab
PYTHONPATH=src python3 -m multi_agent_lab.cli \
  "Design a client onboarding system for a small AI consultancy"
```

The command prints the final artifact path. Inspect `.multi-agent-lab/artifacts/` to see every proposal, critique, and final result. The audit database is `.multi-agent-lab/runs.db`.

## Resume a failed run

Completed agent artifacts are saved as soon as they are produced, including when a later agent or stage fails. Resume the same run by passing its ID:

```bash
PYTHONPATH=src python3 -m multi_agent_lab.cli \
  --workspace .multi-agent-lab \
  --resume RUN_ID
```

Resuming keeps the original objective and run ID. It reuses every existing proposal or critique artifact and calls only agents whose artifacts are missing, then runs the final reviewer if needed. Completed runs cannot be resumed.

Pressing Ctrl-C marks the run as failed, preserves completed artifacts, and prints the exact command needed to resume it. Resume and artifact-reuse actions are also recorded in the event audit trail.

List all runs and their statuses without querying SQLite directly:

```bash
PYTHONPATH=src python3 -m multi_agent_lab.cli \
  --workspace .multi-agent-lab \
  --list-runs
```

Inspect one run's metadata, artifacts, and complete audit event history:

```bash
PYTHONPATH=src python3 -m multi_agent_lab.cli \
  --workspace .multi-agent-lab \
  --show-run RUN_ID
```

During execution, stage starts, completions, reused artifacts, compaction, and retries are printed to stderr. Errors are reported concisely without a Python traceback and failed resumes print a reusable resume command.

The database enforces at most one artifact for each run, stage, and agent.

Transient provider failures (connection errors, timeouts, rate limits, and server errors) are retried three times with exponential backoff. Configure this behavior with `AGENT_MAX_RETRIES` and `AGENT_RETRY_DELAY_SECONDS`; each retry is recorded as an `agent_retrying` audit event.

Anthropic responses use streaming so long generations continue sending data instead of leaving an idle HTTP connection open.

Reviewer inputs are capped at 16,000 characters by default. If proposals and critiques exceed that budget, the workflow preserves the beginning and end of every section and records a `reviewer_input_compacted` audit event with the original and compacted sizes. Set `REVIEWER_MAX_INPUT_CHARS` to change the budget (minimum 1,000).

Every agent response is validated as JSON against its role contract before it becomes an artifact. Malformed output receives one bounded repair attempt. Repair attempts and outcomes are recorded as `output_repairing`, `output_repaired`, or `output_repair_failed` events; a second invalid response fails the run without saving it.

## Step 2 — connect one real provider

Create and activate a virtual environment, then install one adapter:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[anthropic]'
export ANTHROPIC_API_KEY='your-key'
export AGENT_PROVIDER=anthropic
export PRODUCER_A_MODEL=claude-sonnet-4-6
export PRODUCER_B_MODEL=claude-sonnet-4-6
export CRITIC_A_MODEL=claude-sonnet-4-6
export CRITIC_B_MODEL=claude-sonnet-4-6
export REVIEWER_MODEL=claude-sonnet-4-6
multi-agent-lab "Your objective here"
```

Do not commit API keys. Environment variables are intentionally read at runtime.

## Step 3 — make the producers genuinely independent

Install both adapters and configure different providers:

```bash
pip install -e '.[anthropic,openai]'
export ANTHROPIC_API_KEY='your-anthropic-key'
export OPENAI_API_KEY='your-openai-key'

export PRODUCER_A_PROVIDER=anthropic
export PRODUCER_A_MODEL=claude-sonnet-4-6
export PRODUCER_B_PROVIDER=openai
export PRODUCER_B_MODEL=gpt-5-mini

export CRITIC_A_PROVIDER=openai
export CRITIC_A_MODEL=gpt-5-mini
export CRITIC_B_PROVIDER=anthropic
export CRITIC_B_MODEL=claude-sonnet-4-6
export REVIEWER_PROVIDER=anthropic
export REVIEWER_MODEL=claude-sonnet-4-6

multi-agent-lab "Your objective here"
```

Model names change. Use models currently enabled for your provider accounts.

## What each file teaches

- `workflow.py`: deterministic graph, parallel stages, and stopping behavior
- `providers.py`: the boundary that makes agents replaceable
- `store.py`: durable task, event, and artifact history
- `models.py`: shared contracts
- `tests/test_workflow.py`: an offline acceptance test

## Deliberate limits in version one

- Agents cannot spawn more agents.
- Every call has a timeout and output-size ceiling.
- Agents receive only the artifact needed for their stage.
- No agent has tools that can write externally.
- The final reviewer preserves unresolved questions.

## Upgrade path

Once the local workflow is useful, add in this order:

1. A human approval state before external actions.
2. OpenTelemetry-compatible traces and token/cost accounting.
3. A remote-agent adapter using an A2A-style task and artifact contract.
4. A durable workflow runtime only when runs need to survive process restarts.
5. MCP servers for narrowly permissioned tools and shared data.
