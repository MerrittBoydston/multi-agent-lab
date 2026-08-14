# Multi-Agent Lab

The GitHub Actions workflow runs the offline test suite on Python 3.11 and 3.13 for every push to `main` and every pull request. Provider SDKs are installed for adapter contract tests, but CI does not use API keys or make model calls.

A small local reference implementation of a controlled multi-agent environment:

1. Two independent producers create proposals in parallel.
2. Two critics cross-review the proposals in parallel.
3. A separate reviewer synthesizes the final result.
4. SQLite records the run and event history.
5. Every output is saved as a versioned artifact.

The workflow is deterministic; the agents inside each step can come from different providers.

Multi-Agent Lab is released under the [MIT License](LICENSE). Contributions are
welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). Report security issues privately
as described in [SECURITY.md](SECURITY.md).

## Step 1 — run it safely with mock agents

No API key is needed. Install the package in a virtual environment:

```bash
cd multi-agent-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
multi-agent-lab \
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

Resume the newest failed run without looking up its ID:

```bash
PYTHONPATH=src python3 -m multi_agent_lab.cli \
  --workspace .multi-agent-lab \
  --resume-latest
```

Or retry every currently failed run sequentially, oldest first:

```bash
PYTHONPATH=src python3 -m multi_agent_lab.cli \
  --workspace .multi-agent-lab \
  --retry-failed
```

Before each recovery starts, the CLI previews which `stage/agent` artifacts will be reused and which missing stages will execute. `--retry-failed` takes a snapshot of failed runs at startup, continues after individual failures, and exits with an error summary if any remain unsuccessful. If there are no failed runs, `--retry-failed` exits successfully without doing work; `--resume-latest` reports an error.

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

## Configure a workflow with YAML

Pass `--config` to replace the built-in producer/critic/reviewer topology with a
validated sequence of stages. Agents in the same stage run in parallel; stages
run in file order. The final stage must contain exactly one agent.

```bash
multi-agent-lab \
  --workspace .multi-agent-lab \
  --config examples/workflow.yml \
  "Evaluate a practical workflow improvement"
```

Each agent declares an `id`, output-validation `role`, `provider`, `model`,
`system_prompt`, and `input`. Input templates support only these inert tokens:

- `{{ objective }}` — the run objective.
- `{{ artifact.STAGE.AGENT }}` — an artifact produced by an earlier stage.

Unknown tokens, forward references, duplicate stage or agent IDs, unsupported
roles or providers, missing models, and ambiguous final stages are rejected
before a run is created. See [`examples/workflow.yml`](examples/workflow.yml) for
a complete configuration that runs entirely with mock providers.

The fully resolved configuration—including providers, models, prompts, and
templates—is stored as a canonical snapshot on every new run. Resuming without
`--config` always uses that snapshot:

```bash
multi-agent-lab --workspace .multi-agent-lab --resume RUN_ID
```

Supplying `--config` while resuming is optional verification. The command is
rejected before changing run state unless the supplied file exactly matches the
stored snapshot. This also applies to `--resume-latest` and `--retry-failed`.

Runs created before v0.2.0 have no snapshot and continue using the original
built-in topology, resolved from the current provider environment variables.
No manual database migration is required; the snapshot column is added safely
when an existing workspace is opened.

Transient provider failures (connection errors, timeouts, rate limits, and server errors) are retried three times with exponential backoff. Configure this behavior with `AGENT_MAX_RETRIES` and `AGENT_RETRY_DELAY_SECONDS`; each retry is recorded as an `agent_retrying` audit event.

Anthropic responses use streaming so long generations continue sending data instead of leaving an idle HTTP connection open.

Reviewer inputs are capped at 16,000 characters by default. If proposals and critiques exceed that budget, the workflow preserves the beginning and end of every section and records a `reviewer_input_compacted` audit event with the original and compacted sizes. Set `REVIEWER_MAX_INPUT_CHARS` to change the budget (minimum 1,000).

Every agent response is validated as JSON against its role contract before it becomes an artifact. Malformed output receives one bounded repair attempt. Repair attempts and outcomes are recorded as `output_repairing`, `output_repaired`, or `output_repair_failed` events; a second invalid response fails the run without saving it.

Anthropic and OpenAI requests also use each provider's native JSON Schema output constraint. The schemas are shared with local validation so the provider contract and persisted-artifact contract stay aligned. Local validation and repair remain as defense in depth for refusals, unsupported provider models, or unexpected SDK behavior.

Provider schema subsets are handled explicitly: OpenAI receives numeric confidence bounds, while Anthropic receives a compatible numeric field and the same 0–1 bound is enforced by local validation.

Completed-agent events include provider, model, latency, retry and repair counts, and provider-reported token usage. `--show-run` separates total lifecycle time, active wall time, downtime awaiting resume, and summed agent time, then aggregates token usage, costs, and provider breakdowns. Timing is derived from existing failure, interruption, and resume audit events, so older resumed runs are supported.

Optional cost estimates use configurable per-million-token rates. Precedence is per-agent (`REVIEWER_INPUT_COST_PER_MILLION`), per-model (`MODEL_CLAUDE_SONNET_4_6_INPUT_COST_PER_MILLION`), per-provider (`PROVIDER_ANTHROPIC_INPUT_COST_PER_MILLION`), then the global fallback (`AGENT_INPUT_COST_PER_MILLION`). Define matching output rates as well. Prices are intentionally not hard-coded because they change over time.

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
2. OpenTelemetry-compatible trace export.
3. A remote-agent adapter using an A2A-style task and artifact contract.
4. A durable workflow runtime only when runs need to survive process restarts.
5. MCP servers for narrowly permissioned tools and shared data.
