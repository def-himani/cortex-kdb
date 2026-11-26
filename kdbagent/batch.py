"""Sequential, resumable ingestion of Submission 1 papers only."""

from pathlib import Path
from tempfile import NamedTemporaryFile
import json
import os

from kdb_tools import config
from kdb_tools.media import sha256_of
from kdb_tools.actions.objects import read_row

from . import ingest
from . import lake as KL
from .library import HELD_OUT_PAPER, paper_origin, submission_one_papers


class BatchError(ValueError):
    """The requested batch cannot safely continue."""


def atomic_json(path: Path, value: dict) -> None:
    """Replace a JSON checkpoint only after fully writing the new contents."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_checkpoint(path: Path, identity: dict) -> dict:
    if not path.exists():
        return {"identity": identity, "papers": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BatchError(f"Unreadable checkpoint {path}: {error}") from error
    if state.get("identity") != identity or not isinstance(state.get("papers"), dict):
        raise BatchError("Checkpoint belongs to another lake, model or configuration; review it before continuing")
    return state


def verify_completed(ctx, path: Path, record: dict) -> None:
    """Check PDF identity and existence of recorded rows before skipping."""
    report = record["report"]
    digest = sha256_of(path.read_bytes())
    if digest != report["sha256"]:
        raise BatchError(f"PDF bytes changed since completion: {path.name}")
    paper = read_row(ctx, "papers", ctx.iri_base + "papers/" + path.stem)
    if not paper or paper.get("media") != digest or paper.get("version") != report["version"]:
        raise BatchError(f"Checkpoint does not match paper row: {path.name}")
    for ref in report["touched_rows"]:
        table, _, iri = ref.partition("/")
        if table not in ("papers", "models", "datasets", "trained_on", "results") or not read_row(ctx, table, iri):
            raise BatchError(f"Checkpoint references a missing row: {ref}")


def run_library(toml: Path, root: Path, *, limit: int | None = None, paper_id: str | None = None) -> int:
    """Ingest selected initial papers, stopping at the first failure.

    Each paper gets a durable record before and after processing. Only a
    completed, validated paper is skipped. A failed/interrupted paper is
    retried from its first chunk; typed row actions are designed for replays.
    Run only one writer process against this lake at a time.
    """
    if limit is not None and limit <= 0:
        raise BatchError("--limit must be positive")
    paths = submission_one_papers(root)
    if paper_id is not None:
        paths = [p for p in paths if p.stem == paper_id]
        if not paths:
            raise BatchError("--paper must be one of the initial 30 papers; LIBERO-Plus is held out")
    cfg = config.load(toml)
    lake_cfg = KL.lake_config(toml)
    checkpoint = Path(lake_cfg.catalog).resolve().parent / "ingestion-checkpoint.json"
    ctx = KL.open_context(toml, root, model=cfg.model)
    try:
        if read_row(ctx, "papers", ctx.iri_base + "papers/" + HELD_OUT_PAPER):
            raise BatchError("LIBERO-Plus is already in this lake; do not build Submission 1 over Submission 2")
        identity = {
            "catalog": str(Path(lake_cfg.catalog).resolve()), "team": ctx.team,
            "iri_base": ctx.iri_base, "model": cfg.model, "context": cfg.context,
            "base_url": str(cfg.base_url),
        }
        state = load_checkpoint(checkpoint, identity)
        # Verify all previously completed entries, even when --paper filters the run.
        for path in submission_one_papers(root):
            record = state["papers"].get(path.stem, {})
            if record.get("status") == "complete":
                verify_completed(ctx, path, record)
        print(f"Checkpoint: {checkpoint}", flush=True)
        attempted = 0
        agent = None
        for path in paths:
            previous = state["papers"].get(path.stem, {})
            if previous.get("status") == "complete":
                print(f"SKIP {path.stem}: previously completed", flush=True)
                continue
            if limit is not None and attempted >= limit:
                break
            attempted += 1
            origin = paper_origin(path.stem)
            state["papers"][path.stem] = {"status": "running", "origin": origin}
            atomic_json(checkpoint, state)
            print(f"INGEST {path.stem}: {origin}", flush=True)
            try:
                if agent is None:
                    agent = ingest.make_agent(config.model_from(cfg), root, config.settings_from(cfg))
                done = ingest.ingest(ctx, path, agent, cfg.context, origin=origin)
                report = done.report()
                # Empty extraction is not a successful paper build. A paper-only
                # write after every fact is rejected also requires review.
                usable = any(not row.startswith("papers/") for row in done.touched_rows)
                complete = not done.checked.errors and usable
                state["papers"][path.stem] = {
                    "status": "complete" if complete else "failed", "origin": origin,
                    "report": report,
                    "reason": None if complete else "Validation errors or no usable instance rows",
                }
                atomic_json(checkpoint, state)
                print(f"{path.stem}: {len(done.facts)} facts, {len(done.rows)} rows written, "
                      f"{len(done.skipped)} skipped, {done.usage['input_tokens']:,} input tokens, "
                      f"{done.usage['output_tokens']:,} output tokens", flush=True)
                if not complete:
                    print(f"FAILED {path.stem}: review {checkpoint}", flush=True)
                    return 1
            except (Exception, KeyboardInterrupt) as error:
                state["papers"][path.stem] = {
                    "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    "origin": origin, "reason": str(error) or type(error).__name__,
                }
                atomic_json(checkpoint, state)
                print(f"STOPPED {path.stem}: {type(error).__name__}: {error}", flush=True)
                return 130 if isinstance(error, KeyboardInterrupt) else 1
        complete_count = sum(record.get("status") == "complete" for record in state["papers"].values())
        print(f"Submission 1 checkpoint: {complete_count}/30 complete; {attempted} attempted this run", flush=True)
        return 0
    finally:
        ctx.lake.close()
