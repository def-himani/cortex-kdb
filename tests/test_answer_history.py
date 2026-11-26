"""Offline tests for citations, history, PNG display and typed Analysis storage."""

import json
from pathlib import Path

import pytest
from kdb_tools import config, media
from kdb_tools.actions.objects import UpsertObjects
from kdb_tools.tools import KdbTools

from kdbagent import ask, ingest
from kdbagent import history as H
from kdbagent.analysis_store import save_analysis
from kdbagent.citations import (
    CitationError, collect_citations, render_answer, render_region, show_citations,
)
from test_starter import ctx, initialized, project, paper, scripted, FACT, RESULT_QUOTE, kdb


@pytest.fixture
def grounded_answer(ctx, paper):
    done = ingest.ingest(ctx, paper, ingest.make_agent(scripted({"facts": [FACT]})[0]), 32768,
                         origin="https://arxiv.org/abs/2406.09246v3")
    assert not done.checked.errors
    iri = ctx.lake.con.execute("SELECT iri FROM models WHERE title = ?", ["OpenVLA"]).fetchone()[0]
    out = ask.answer(ask.make_agent(scripted({"answerable": True, "claims": [
        {"text": RESULT_QUOTE, "object": iri, "evidence": [0]}
    ]})[0]), ctx, "What result did OpenVLA report?")
    return out


def test_citations_resolve_paper_version_exact_text_and_page(ctx, grounded_answer):
    [citation] = collect_citations(ctx, grounded_answer)
    assert citation.paper_id == "2406.09246" and citation.version == "v3"
    assert citation.ref.page == 1 and citation.ref.quote == RESULT_QUOTE
    rendered = render_answer(grounded_answer, [citation])
    assert "2406.09246v3, PDF page 1" in rendered
    assert RESULT_QUOTE in rendered and citation.key in rendered


def test_history_records_provenance_and_filters_paper(ctx, grounded_answer, tmp_path):
    path = tmp_path / "history.sqlite3"
    citations = collect_citations(ctx, grounded_answer)
    snapshot = ctx.lake.snapshot()
    run_id = H.record(path, "Question", grounded_answer, render_answer(grounded_answer, citations),
                      citations, "scripted", snapshot, "test-commit", False)
    [row] = H.read_history(path, paper="2406.09246v3")
    assert row["id"] == run_id and row["git_commit"] == "test-commit"
    assert row["lake_snapshot"] == snapshot and row["model"] == "scripted"
    assert row["usage"] == grounded_answer.usage
    assert row["conversation"] == json.loads(grounded_answer.conversation_json)
    assert row["citations"][0]["reference"]["quote"] == RESULT_QUOTE
    assert H.read_history(path, paper="2510.13626") == []
    assert H.read_history(path, paper="2406.09246")[0]["id"] == run_id


def test_refusals_are_recorded_without_citations(tmp_path):
    out = ask.Answer(answerable=False, note="No matching evidence.")
    path = tmp_path / "history.sqlite3"
    run_id = H.record(path, "Unknown question", out, render_answer(out, []), [], "scripted", 1, None, None)
    [row] = H.read_history(path, run_id=run_id)
    assert row["answer"]["answerable"] is False and row["citations"] == []
    assert H.read_history(path, paper="2406.09246") == []


def test_show_outputs_highlighted_png_without_lake_writes(ctx, grounded_answer, tmp_path):
    before = ctx.lake.snapshot()
    [path] = show_citations(ctx, collect_citations(ctx, grounded_answer), tmp_path / "images")
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    from PIL import Image
    with Image.open(path) as image:
        assert image.width > 0 and image.height > 0
    assert ctx.lake.snapshot() == before


def test_render_rejects_changed_source_bytes(ctx, grounded_answer, tmp_path):
    [citation] = collect_citations(ctx, grounded_answer)
    with pytest.raises(CitationError, match="digest"):
        render_region(b"not the source PDF", citation.ref, tmp_path / "bad.png")
    assert not (tmp_path / "bad.png").exists()


def test_save_analysis_copies_exact_evidence_and_drew_on(ctx, grounded_answer):
    [citation] = collect_citations(ctx, grounded_answer)
    iri = save_analysis(ctx, "What result was reported?", grounded_answer)
    tools = KdbTools(ctx.root, ctx=ctx)
    row = tools.get_object(iri)
    assert row["drew_on"] == [citation.row_iri]
    assert row["evidence"] == [citation.reference]
    assert RESULT_QUOTE in row["answer"] and "evidence[0]" in row["answer"]
    assert not tools.check_pages([f"analyses/{iri}"]).errors


def test_save_refuses_ungrounded_citations_before_writing(ctx, grounded_answer):
    [citation] = collect_citations(ctx, grounded_answer)
    wrong = dict(citation.reference, quote="A fabricated quote absent from the PDF.")
    action = UpsertObjects()
    params = {"class_iri": "https://kdb.aegean.ai/vla/classes/model",
              "rows": [{"iri": citation.row_iri, "evidence": [wrong]}]}
    action.submit(ctx, params, action.dry_run(ctx, params))
    before = ctx.lake.snapshot()
    with pytest.raises(CitationError, match="validation or grounding"):
        save_analysis(ctx, "Question", grounded_answer)
    assert ctx.lake.con.execute("SELECT count(*) FROM analyses").fetchone()[0] == 0
    assert ctx.lake.snapshot() == before


def test_save_refuses_unanswerable(ctx):
    with pytest.raises(CitationError, match="unanswerable"):
        save_analysis(ctx, "Unknown", ask.Answer(answerable=False, note="No evidence"))
    assert ctx.lake.con.execute("SELECT count(*) FROM analyses").fetchone()[0] == 0


def test_cli_ask_show_save_and_history(initialized, paper, monkeypatch, capsys):
    # Ingest through the CLI so each command owns and closes its lake connection.
    monkeypatch.setattr(config, "model_from", lambda cfg: scripted({"facts": [FACT]})[0])
    assert kdb(initialized, "ingest", str(paper), "--origin", "https://arxiv.org/abs/2406.09246v3") == 0
    from kdbagent import lake as KL
    c = KL.open_context(initialized / "kdb.toml", initialized / "kdb")
    try:
        iri = c.lake.con.execute("SELECT iri FROM models WHERE title = ?", ["OpenVLA"]).fetchone()[0]
        snapshot = c.lake.snapshot()
    finally:
        c.lake.close()
    monkeypatch.setattr(config, "model_from", lambda cfg: scripted({"answerable": True, "claims": [
        {"text": RESULT_QUOTE, "object": iri, "evidence": [0]}
    ]})[0])
    capsys.readouterr()
    assert kdb(initialized, "ask", "What was reported?", "--show", "--save") == 0
    output = capsys.readouterr().out
    assert "2406.09246v3, PDF page 1" in output and "Saved analysis:" in output
    assert "Highlighted evidence:" in output
    path = H.history_path(initialized / "kdb")
    [row] = H.read_history(path)
    assert row["lake_snapshot"] == snapshot and row["analysis_iri"]
    assert list((path.parent / "evidence-images" / str(row["id"])).glob("*.png"))
    assert kdb(initialized, "history", "--paper", "2406.09246", "--json") == 0
    [history] = json.loads(capsys.readouterr().out)
    assert history["id"] == row["id"] and history["conversation"]
