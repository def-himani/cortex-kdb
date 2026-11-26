"""Offline regression tests for ingestion reporting, checkpoints and conflicts."""

from types import SimpleNamespace

import pytest
from fpdf import FPDF

from kdb_tools.media import sha256_of
from kdb_tools.report import Report
from kdb_tools.seed import seed_action
from kdbagent import ingest, batch
from kdbagent.conflicts import record_contested

# Reuse the starter's temporary lake and scripted model fixtures.
from test_starter import (  # noqa: F401
    paper, project, initialized, ctx, scripted, FACT,
    seed_robotics_terms, ROBOTICS,
)


def test_ingest_reports_usage_and_replay_rows(ctx, paper):
    first = ingest.ingest(ctx, paper, ingest.make_agent(scripted({"facts": [FACT]})[0]), 32768)
    assert first.usage["requests"] >= 1
    assert first.chunks[0]["pages"] == [1]
    assert first.report()["sha256"] == sha256_of(paper.read_bytes())
    assert not first.checked.errors
    second = ingest.ingest(ctx, paper, ingest.make_agent(scripted({"facts": [FACT]})[0]), 32768)
    assert second.rows == []
    assert second.touched_rows == first.touched_rows
    assert not second.checked.errors


def test_ingest_reports_skipped_dataset(ctx, paper):
    fact = dict(FACT, kind="dataset", entity="ExampleDataset")
    done = ingest.ingest(ctx, paper, ingest.make_agent(scripted({"facts": [fact]})[0]), 32768)
    assert len(done.skipped) == 1
    assert "collection method" in done.skipped[0]["reason"]
    assert done.skipped[0]["fact"]["kind"] == "dataset"


def test_checkpoint_round_trip_and_identity(tmp_path):
    path = tmp_path / "lake" / "checkpoint.json"
    identity = {"team": "test", "model": "scripted"}
    state = batch.load_checkpoint(path, identity)
    state["papers"]["1234.56789"] = {"status": "complete"}
    batch.atomic_json(path, state)
    assert batch.load_checkpoint(path, identity) == state
    with pytest.raises(batch.BatchError, match="another lake"):
        batch.load_checkpoint(path, dict(identity, model="other"))


def test_completed_paper_requires_matching_pdf(ctx, paper):
    done = ingest.ingest(ctx, paper, ingest.make_agent(scripted({"facts": [FACT]})[0]), 32768)
    record = {"status": "complete", "report": done.report()}
    batch.verify_completed(ctx, paper, record)
    bad = {"status": "complete", "report": dict(done.report(), sha256="0" * 64)}
    with pytest.raises(batch.BatchError, match="PDF bytes changed"):
        batch.verify_completed(ctx, paper, bad)


def test_held_out_rejected_by_library_runner(tmp_path):
    with pytest.raises(batch.BatchError, match="held out"):
        batch.run_library(tmp_path / "kdb.toml", tmp_path / "kdb", paper_id="2510.13626")


def test_library_resume_skips_completed_paper(tmp_path, monkeypatch):
    paths = [tmp_path / "2406.09246.pdf", tmp_path / "2306.03310.pdf"]
    for path in paths:
        path.write_bytes(b"test")
    monkeypatch.setattr(batch, "submission_one_papers", lambda root: paths)
    monkeypatch.setattr(batch.config, "load", lambda toml: SimpleNamespace(
        model="scripted", context=32768, base_url="http://localhost:11434/v1"))
    monkeypatch.setattr(batch.config, "model_from", lambda cfg: None)
    monkeypatch.setattr(batch.config, "settings_from", lambda cfg: {})
    monkeypatch.setattr(batch.KL, "lake_config", lambda toml: SimpleNamespace(catalog=tmp_path / "catalog.ducklake"))
    lake = SimpleNamespace(close=lambda: None)
    context = SimpleNamespace(team="test", iri_base="https://example.test/", lake=lake)
    monkeypatch.setattr(batch.KL, "open_context", lambda *args, **kwargs: context)
    monkeypatch.setattr(batch, "read_row", lambda *args: None)
    verified = []
    monkeypatch.setattr(batch, "verify_completed", lambda ctx, path, record: verified.append(path.stem))
    monkeypatch.setattr(batch.ingest, "make_agent", lambda *args: None)
    calls = []

    def fake_ingest(ctx, path, agent, context, origin=None):
        calls.append(path.stem)
        return SimpleNamespace(
            facts=[1], rows=["models/example"], touched_rows=["models/example"], skipped=[],
            checked=SimpleNamespace(errors=[]), usage={"input_tokens": 10, "output_tokens": 2},
            report=lambda: {"sha256": sha256_of(path.read_bytes())},
        )

    monkeypatch.setattr(batch.ingest, "ingest", fake_ingest)
    toml, root = tmp_path / "kdb.toml", tmp_path / "kdb"
    assert batch.run_library(toml, root, limit=1) == 0
    assert calls == [paths[0].stem]
    assert batch.run_library(toml, root) == 0
    assert calls == [path.stem for path in paths]
    assert verified == [paths[0].stem]


def synthetic_paper(ctx, lines):
    path = ctx.root / "papers" / "2300.00001.pdf"
    pdf = FPDF(unit="pt", format="letter")
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    pdf.set_title("Synthetic study for ingestion tests")
    for n, line in enumerate(lines):
        pdf.text(30, 60 + 20 * n, line)
    pdf.output(str(path))
    return path


def test_dataset_and_training_are_written_in_dependency_order(ctx):
    collection_quote = "ExampleSet was collected by human demonstrations."
    training_quote = "ExampleModel was pretrained on ExampleSet."
    path = synthetic_paper(ctx, [collection_quote, training_quote])
    base = "https://kdb.aegean.ai/datasets/"
    spec = {"domain": "datasets", "base_iri": base, "title": "Datasets", "description": "Test terms.",
            "papers": {"test": path.stem}, "terms": [
                {"slug": "human-collected", "kind": "collection", "pref_label": "Human-collected",
                 "scope": [{"text": "Collection by people.", "quote": {"paper": "test", "page": 1, "text": collection_quote}}]},
                {"slug": "pretraining", "kind": "roles", "pref_label": "Pretraining",
                 "scope": [{"text": "A training role.", "quote": {"paper": "test", "page": 1, "text": training_quote}}]},
            ]}
    report = Report("test", extra={"written": [], "terms": 0, "types": 0})
    seed_action(ctx, "datasets", spec, report)
    assert not report.findings
    facts = [
        {"kind": "training", "entity": "ExampleModel", "dataset": "ExampleSet", "role": "Pretraining",
         "sentence": training_quote, "quote": training_quote, "page": 1},
        {"kind": "dataset", "entity": "ExampleSet", "collection": ["Human-collected"],
         "sentence": collection_quote, "quote": collection_quote, "page": 1},
    ]
    done = ingest.ingest(ctx, path, ingest.make_agent(scripted({"facts": facts})[0]), 32768)
    assert not done.checked.errors
    assert not done.skipped
    assert ctx.lake.con.execute("SELECT collection FROM datasets").fetchall() == [([base + "collection/human-collected"],)]
    assert ctx.lake.con.execute("SELECT role FROM trained_on").fetchall() == [(base + "roles/pretraining",)]


def test_course_name_collision_does_not_merge_variants(ctx):
    quotes = ["OpenVLA-OFT is a baseline.", "OpenVLA-OFT+ is a different variant."]
    path = synthetic_paper(ctx, quotes)
    facts = [{"kind": "model", "entity": name, "sentence": text, "quote": text, "page": 1}
             for name, text in zip(["OpenVLA-OFT", "OpenVLA-OFT+"], quotes)]
    done = ingest.ingest(ctx, path, ingest.make_agent(scripted({"facts": facts})[0]), 32768)
    assert not done.checked.errors
    assert ctx.lake.con.execute("SELECT title FROM models").fetchall() == [("OpenVLA-OFT",)]
    assert "Name collision" in done.skipped[0]["reason"]


def test_candidates_require_review_before_contested_write(ctx):
    seed_robotics_terms(ctx)
    quotes = ["ExampleModel reports 53.7 percent success on LIBERO-Long.",
              "The 53.7 percent value is incorrect; ExampleModel achieved 55.1 percent."]
    path = synthetic_paper(ctx, quotes)
    facts = [{"kind": "result", "entity": "ExampleModel", "sentence": text, "quote": text, "page": 1,
              "value": value, "metric": "success rate (%)", "task": "Pick and place", "benchmark": "LIBERO",
              "suite": "LIBERO-Long", "embodiment": "Franka Panda", "trials": 100}
             for value, text in zip([53.7, 55.1], quotes)]
    done = ingest.ingest(ctx, path, ingest.make_agent(scripted({"facts": facts})[0]), 32768)
    assert not done.checked.errors
    assert len(done.candidates) == 1
    assert ctx.lake.con.execute("SELECT count(*) FROM contested").fetchone()[0] == 0
    item = done.candidates[0]
    row = record_contested(ctx, item["claim"], [item["disputed_by_candidate"]])
    assert row.startswith("contested/")
    assert not ingest.check(ctx, [row]).errors
    assert ctx.lake.con.execute("SELECT count(*) FROM results").fetchone()[0] == 2
    assert ctx.lake.con.execute("SELECT count(*) FROM contested").fetchone()[0] == 1
    assert record_contested(ctx, item["claim"], [item["disputed_by_candidate"]]) == row
    assert ctx.lake.con.execute("SELECT count(*) FROM contested").fetchone()[0] == 1
