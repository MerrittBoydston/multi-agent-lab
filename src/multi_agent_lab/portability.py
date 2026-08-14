from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

from .store import RunStore

BUNDLE_SCHEMA_VERSION = 1
MAX_MANIFEST_BYTES = 10 * 1024 * 1024
MAX_ARTIFACT_BYTES = 100 * 1024 * 1024
MAX_BUNDLE_BYTES = 500 * 1024 * 1024
MAX_BUNDLE_MEMBERS = 10_000
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def export_run(store: RunStore, run_id: str, output: Path | None = None) -> Path:
    run, artifacts, events = store.export_snapshot(run_id)
    output = output or Path(f"{run_id}.multi-agent-lab.zip")
    if output.exists():
        raise ValueError(f"Output path already exists: {output}")
    payloads: dict[str, bytes] = {}
    manifest_artifacts = []
    for artifact in artifacts:
        path = Path(artifact["path"])
        if not path.is_file():
            raise ValueError(f"Artifact file is missing: {path}")
        content = path.read_bytes()
        member = f"artifacts/{artifact['artifact_id']}.md"
        payloads[member] = content
        manifest_artifacts.append({
            key: artifact[key] for key in ("artifact_id", "run_id", "stage", "author", "created_at")
        } | {"member": member, "sha256": _digest(content), "size": len(content)})
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "run": run,
        "events": events,
        "artifacts": manifest_artifacts,
    }
    manifest_bytes = _canonical_json(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", manifest_bytes)
        archive.writestr("manifest.sha256", _digest(manifest_bytes) + "\n")
        for member, content in payloads.items():
            archive.writestr(member, content)
    return output


def _read_member(archive: zipfile.ZipFile, name: str, maximum: int) -> bytes:
    try:
        info = archive.getinfo(name)
    except KeyError:
        raise ValueError(f"Bundle is missing {name}") from None
    if info.file_size > maximum:
        raise ValueError(f"Bundle member is too large: {name}")
    return archive.read(info)


def import_run(store: RunStore, bundle: Path) -> str:
    try:
        with zipfile.ZipFile(bundle) as archive:
            names = archive.namelist()
            if len(names) > MAX_BUNDLE_MEMBERS or sum(info.file_size for info in archive.infolist()) > MAX_BUNDLE_BYTES:
                raise ValueError("Bundle exceeds the safe size limit")
            if len(names) != len(set(names)):
                raise ValueError("Bundle contains duplicate archive members")
            if any(name.startswith(("/", "\\")) or ".." in Path(name).parts for name in names):
                raise ValueError("Bundle contains an unsafe archive path")
            manifest_bytes = _read_member(archive, "manifest.json", MAX_MANIFEST_BYTES)
            recorded_hash = _read_member(archive, "manifest.sha256", 128).decode("ascii").strip()
            if recorded_hash != _digest(manifest_bytes):
                raise ValueError("Bundle manifest checksum does not match")
            manifest = json.loads(manifest_bytes)
            if not isinstance(manifest, dict):
                raise ValueError("Bundle manifest must be an object")
            if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
                raise ValueError(f"Unsupported bundle schema version: {manifest.get('schema_version')!r}")
            if set(manifest) != {"schema_version", "run", "events", "artifacts"}:
                raise ValueError("Bundle manifest has unexpected fields")
            run = manifest["run"]
            run_id = run.get("run_id") if isinstance(run, dict) else None
            if not isinstance(run_id, str) or not SAFE_ID.fullmatch(run_id):
                raise ValueError("Bundle has an invalid run ID")
            required_run = {"objective", "status", "created_at"}
            if not required_run.issubset(run) or not all(isinstance(run[key], str) for key in required_run):
                raise ValueError("Bundle has incomplete run metadata")
            events = manifest["events"]
            artifacts = manifest["artifacts"]
            if not isinstance(events, list) or not isinstance(artifacts, list):
                raise ValueError("Bundle events and artifacts must be arrays")
            event_fields = {"stage", "agent_id", "event_type", "payload_json", "created_at"}
            for event in events:
                if not isinstance(event, dict) or not event_fields.issubset(event):
                    raise ValueError("Bundle contains invalid event metadata")
                if (not all(isinstance(event[key], str) for key in event_fields - {"agent_id"})
                        or (event["agent_id"] is not None and not isinstance(event["agent_id"], str))):
                    raise ValueError("Bundle contains invalid event metadata")
                try:
                    payload = json.loads(event["payload_json"])
                except json.JSONDecodeError:
                    raise ValueError("Bundle contains invalid event JSON") from None
                if not isinstance(payload, dict):
                    raise ValueError("Bundle event payloads must be JSON objects")
            expected = {"manifest.json", "manifest.sha256"}
            contents: dict[str, str] = {}
            slots = set()
            artifact_ids = set()
            for artifact in artifacts:
                if not isinstance(artifact, dict):
                    raise ValueError("Bundle contains invalid artifact metadata")
                artifact_id = artifact.get("artifact_id")
                member = artifact.get("member")
                if (not isinstance(artifact_id, str) or not SAFE_ID.fullmatch(artifact_id)
                        or not isinstance(member, str) or member != f"artifacts/{artifact_id}.md"):
                    raise ValueError("Bundle contains an invalid artifact ID or path")
                if artifact.get("run_id") != run_id or artifact_id in artifact_ids:
                    raise ValueError("Bundle contains duplicate or mismatched artifacts")
                slot = (artifact.get("stage"), artifact.get("author"))
                if not all(isinstance(value, str) and SAFE_ID.fullmatch(value) for value in slot) or slot in slots:
                    raise ValueError("Bundle contains an invalid or duplicate artifact slot")
                content = _read_member(archive, member, MAX_ARTIFACT_BYTES)
                if len(content) != artifact.get("size") or _digest(content) != artifact.get("sha256"):
                    raise ValueError(f"Artifact checksum does not match: {artifact_id}")
                try:
                    contents[artifact_id] = content.decode("utf-8")
                except UnicodeDecodeError:
                    raise ValueError(f"Artifact is not UTF-8 text: {artifact_id}") from None
                expected.add(member)
                artifact_ids.add(artifact_id)
                slots.add(slot)
            if set(names) != expected:
                raise ValueError("Bundle contains undeclared archive members")
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid run bundle: {exc}") from None
    clean_artifacts = [{key: artifact[key] for key in
                       ("artifact_id", "run_id", "stage", "author", "created_at")} for artifact in artifacts]
    store.import_snapshot(run, clean_artifacts, events, contents)
    return run_id


def markdown_report(store: RunStore, run_id: str) -> str:
    run, artifacts, events = store.export_snapshot(run_id)
    completed = []
    counts = defaultdict(int)
    for event in events:
        counts[event["event_type"]] += 1
        if event["event_type"] == "agent_completed":
            metrics = json.loads(event["payload_json"]).get("metrics")
            if metrics:
                completed.append(metrics)
    controls = store.controls(run_id)
    lines = [
        f"# Run {run_id}", "", f"**Status:** {run['status']}",
        f"**Objective:** {run['objective']}", f"**Created:** {run['created_at']}",
        f"**Completed:** {run['completed_at'] or '-'}", "", "## Metrics", "",
        f"- Agents completed: {len(completed)}", f"- Retries: {sum(item.get('retries', 0) for item in completed)}",
        f"- Repairs: {sum(item.get('repairs', 0) for item in completed)}",
        f"- Input tokens: {sum(item.get('input_tokens') or 0 for item in completed)}",
        f"- Output tokens: {sum(item.get('output_tokens') or 0 for item in completed)}",
        f"- Estimated cost: ${sum(item.get('estimated_cost_usd') or 0 for item in completed):.6f}",
        f"- Token budget used: {controls['used_tokens']}/{controls['max_tokens'] or '-'}",
        f"- Cost budget used: ${controls['used_cost']:.6f}/{controls['max_cost'] if controls['max_cost'] is not None else '-'}",
        "", "## Artifacts", "",
    ]
    lines.extend(f"- `{item['stage']}/{item['author']}` — `{item['artifact_id']}`" for item in artifacts)
    if not artifacts:
        lines.append("- None")
    lines.extend(["", "## Audit summary", ""])
    lines.extend(f"- `{name}`: {count}" for name, count in sorted(counts.items()))
    if not counts:
        lines.append("- No events")
    return "\n".join(lines) + "\n"
