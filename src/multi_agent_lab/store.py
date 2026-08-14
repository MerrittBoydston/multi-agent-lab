from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .models import Artifact, utc_now


class RunStore:
    def __init__(self, root: Path):
        self.root = root
        self.artifact_root = root / "artifacts"
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.db_path = root / "runs.db"
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    objective TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    agent_id TEXT,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    author TEXT NOT NULL,
                    path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS artifacts_run_stage_author
                    ON artifacts (run_id, stage, author);
                """
            )

    def create_run(self, objective: str) -> str:
        run_id = uuid.uuid4().hex[:12]
        with self._connect() as db:
            db.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, NULL)",
                (run_id, objective, "running", utc_now()),
            )
        return run_id

    def resume_run(self, run_id: str) -> str:
        with self._connect() as db:
            row = db.execute(
                "SELECT objective, status FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"Run {run_id!r} does not exist")
            if row["status"] == "completed":
                raise ValueError(f"Run {run_id!r} is already completed")
            db.execute(
                "UPDATE runs SET status = 'running', completed_at = NULL WHERE run_id = ?",
                (run_id,),
            )
            db.execute(
                "INSERT INTO events (run_id, stage, agent_id, event_type, payload_json, created_at) "
                "VALUES (?, 'workflow', NULL, 'resumed', ?, ?)",
                (run_id, json.dumps({"previous_status": row["status"]}), utc_now()),
            )
        return str(row["objective"])

    def list_runs(self) -> list[sqlite3.Row]:
        with self._connect() as db:
            return db.execute(
                "SELECT run_id, status, created_at, completed_at, objective "
                "FROM runs ORDER BY created_at DESC"
            ).fetchall()

    def failed_run_ids(self, *, newest_first: bool = False) -> list[str]:
        order = "DESC" if newest_first else "ASC"
        with self._connect() as db:
            rows = db.execute(
                f"SELECT run_id FROM runs WHERE status = 'failed' ORDER BY created_at {order}"
            ).fetchall()
        return [str(row["run_id"]) for row in rows]

    def run_details(self, run_id: str) -> tuple[sqlite3.Row, list[sqlite3.Row], list[sqlite3.Row]]:
        with self._connect() as db:
            run = db.execute(
                "SELECT run_id, status, created_at, completed_at, objective "
                "FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if run is None:
                raise ValueError(f"Run {run_id!r} does not exist")
            artifacts = db.execute(
                "SELECT stage, author, artifact_id, path, created_at FROM artifacts "
                "WHERE run_id = ? ORDER BY created_at", (run_id,)
            ).fetchall()
            events = db.execute(
                "SELECT stage, agent_id, event_type, payload_json, created_at FROM events "
                "WHERE run_id = ? ORDER BY event_id", (run_id,)
            ).fetchall()
        return run, artifacts, events

    def artifacts_for_run(self, run_id: str) -> dict[tuple[str, str], Artifact]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT artifact_id, run_id, stage, author, path, created_at "
                "FROM artifacts WHERE run_id = ? ORDER BY created_at",
                (run_id,),
            ).fetchall()
        artifacts = {}
        for row in rows:
            artifact = Artifact(**dict(row))
            if Path(artifact.path).is_file():
                artifacts[(artifact.stage, artifact.author)] = artifact
        return artifacts

    def event(self, run_id: str, stage: str, event_type: str, payload: dict, agent_id: str | None = None) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO events (run_id, stage, agent_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, stage, agent_id, event_type, json.dumps(payload), utc_now()),
            )

    def artifact(self, run_id: str, stage: str, author: str, content: str) -> Artifact:
        artifact_id = uuid.uuid4().hex[:12]
        folder = self.artifact_root / run_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{stage}-{author}-{artifact_id}.md"
        path.write_text(content, encoding="utf-8")
        artifact = Artifact(artifact_id, run_id, stage, author, str(path), utc_now())
        try:
            with self._connect() as db:
                db.execute(
                    "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?)",
                    tuple(artifact.to_dict().values()),
                )
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return artifact

    def finish(self, run_id: str, status: str = "completed") -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE runs SET status = ?, completed_at = ? WHERE run_id = ?",
                (status, utc_now(), run_id),
            )
