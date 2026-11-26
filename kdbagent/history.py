"""Local SQLite history of completed QA runs, with snapshot provenance."""

from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import subprocess


SCHEMA = """
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY,
    asked_at TEXT NOT NULL,
    question TEXT NOT NULL,
    answer_json TEXT NOT NULL,
    rendered_answer TEXT NOT NULL,
    model TEXT NOT NULL,
    usage_json TEXT NOT NULL,
    elapsed_seconds REAL NOT NULL,
    git_commit TEXT,
    git_dirty INTEGER,
    lake_snapshot INTEGER NOT NULL,
    conversation_json TEXT NOT NULL,
    analysis_iri TEXT
);
CREATE TABLE IF NOT EXISTS citations (
    question_id INTEGER NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    paper_id TEXT NOT NULL,
    version TEXT,
    citation_json TEXT NOT NULL,
    PRIMARY KEY (question_id, position)
);
CREATE INDEX IF NOT EXISTS citations_paper ON citations(paper_id, question_id);
"""


def history_path(root: Path) -> Path:
    return Path(root).resolve().parent / "lake" / "history.sqlite3"


def git_provenance(root: Path) -> tuple[str | None, bool | None]:
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    try:
        commit = git("rev-parse", "HEAD")
    except (OSError, subprocess.SubprocessError):
        return None, None
    try:
        return commit, bool(git("status", "--porcelain"))
    except (OSError, subprocess.SubprocessError):
        return commit, None


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30)
    try:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        con.executescript(SCHEMA)
        return con
    except BaseException:
        con.close()
        raise


def record(path: Path, question: str, answer, rendered: str, citations: list,
           model: str, snapshot: int, commit: str | None, dirty: bool | None) -> int:
    """Insert the completed answer and citation copies atomically."""
    conversation = answer.conversation_json
    if not isinstance(json.loads(conversation), list):
        raise ValueError("Expected a complete message list in conversation_json")
    with closing(connect(path)) as con, con:
        cur = con.execute("""INSERT INTO questions
            (asked_at, question, answer_json, rendered_answer, model, usage_json, elapsed_seconds,
             git_commit, git_dirty, lake_snapshot, conversation_json)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now(timezone.utc).isoformat(), question, answer.model_dump_json(), rendered,
             model, json.dumps(answer.usage), answer.elapsed_seconds, commit, dirty, snapshot, conversation))
        run_id = cur.lastrowid
        con.executemany("INSERT INTO citations VALUES (?, ?, ?, ?, ?)", [
            (run_id, i, c.paper_id, c.version, json.dumps(c.to_dict(), ensure_ascii=False))
            for i, c in enumerate(citations)
        ])
        return run_id


def link_analysis(path: Path, run_id: int, iri: str) -> None:
    with closing(connect(path)) as con, con:
        con.execute("UPDATE questions SET analysis_iri = ? WHERE id = ?", (iri, run_id))


def read_history(path: Path, paper: str | None = None, limit: int = 50,
                 run_id: int | None = None) -> list[dict]:
    if not 1 <= limit <= 1000:
        raise ValueError("History limit must be between 1 and 1000")
    clauses, args = [], []
    if paper:
        clauses.append("EXISTS (SELECT 1 FROM citations c WHERE c.question_id = q.id "
                       "AND (c.paper_id = ? OR c.paper_id || COALESCE(c.version, '') = ?))")
        args.extend([paper, paper])
    if run_id is not None:
        clauses.append("q.id = ?")
        args.append(run_id)
    where = " AND ".join(clauses) or "TRUE"
    with closing(connect(path)) as con:
        rows = con.execute(f"SELECT q.* FROM questions q WHERE {where} ORDER BY q.id DESC LIMIT ?",
                           [*args, limit]).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["answer"] = json.loads(item.pop("answer_json"))
            item["usage"] = json.loads(item.pop("usage_json"))
            item["conversation"] = json.loads(item.pop("conversation_json"))
            item["citations"] = [json.loads(c[0]) for c in con.execute(
                "SELECT citation_json FROM citations WHERE question_id = ? ORDER BY position", [item["id"]])]
            result.append(item)
        return result


def print_history(path: Path, paper: str | None = None, limit: int = 50,
                  run_id: int | None = None, as_json: bool = False) -> None:
    rows = read_history(path, paper, limit, run_id)
    if as_json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return
    if not rows:
        print("No matching question history.")
        return
    for row in rows:
        print(f"#{row['id']} | {row['asked_at']} | {row['model']}")
        print(f"Question: {row['question']}")
        print(row["rendered_answer"])
        print(f"Git: {row['git_commit'] or 'unavailable'}; dirty: {row['git_dirty']}; lake snapshot: {row['lake_snapshot']}")
        print(f"Usage: {json.dumps(row['usage'])}")
        if row["analysis_iri"]:
            print(f"Saved analysis: {row['analysis_iri']}")
        print()
