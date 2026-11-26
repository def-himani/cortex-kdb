"""Offline regression tests for the read-only question-answering pipeline."""

import json
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from kdb_tools.report import Report
from kdb_tools.seed import seed_action
from kdb_tools.tools import ToolInputError
from kdbagent import ask, ingest, resolution
from kdbagent.retrieval import Retrieval, comparison_groups, bounded_limit
from kdbagent.resolution import OntologySearch, ResolutionError, DATASETS, VLA_DATASETS

from test_starter import (  # noqa: F401
    paper, project, initialized, ctx, scripted, FACT, RESULT_QUOTE, TERM_QUOTES, ROBOTICS,
)
from test_ingestion_reliability import synthetic_paper


def test_ontology_search_follows_transitive_broader_links(monkeypatch):
    base = ROBOTICS + "tasks/"
    records = [
        ("manipulation", "Manipulation", [], []),
        ("pick-and-place", "Pick and place", ["PnP"], ["/ontology/robotics/tasks/manipulation.md"]),
        ("stacking", "Stacking", [], ["pick-and-place.md"]),
    ]
    pages = [SimpleNamespace(rel=f"ontology/robotics/tasks/{slug}.md", frontmatter={
        "type": "Term", "iri": base + slug, "pref_label": label, "alt_labels": aliases, "broader": parents,
    }) for slug, label, aliases, parents in records]
    monkeypatch.setattr(resolution.bundle, "load", lambda root: SimpleNamespace(concepts=pages))
    terms = OntologySearch(None)
    assert terms.match("PnP", base) == base + "pick-and-place"
    assert set(terms.narrower(base + "manipulation")) == {base + row[0] for row in records}
    assert terms.search("PnP", "tasks")[0]["iri"] == base + "pick-and-place"
    assert terms.search("PnP", "roles") == []
    with pytest.raises(ResolutionError):
        terms.match("PnP", DATASETS + "roles/")


def test_comparison_groups_separate_metrics_and_conditions():
    row = {"iri": "r1", "task": "pick", "benchmark": "LIBERO", "suite": "Long", "embodiment": "Panda",
           "perturbation": None, "trials": 100, "trials_reported": True, "metric": "success rate (%)"}
    rows = [row, dict(row, iri="r2"), dict(row, iri="r3", suite="Goal"),
            dict(row, iri="r4", trials=None, trials_reported=False),
            dict(row, iri="r5", metric="success rate (fraction)"), dict(row, iri="r6", perturbation="noise")]
    groups = comparison_groups(rows)
    assert [g["rows"] for g in groups] == [["r1", "r2"], ["r3"], ["r4"], ["r5"], ["r6"]]
    assert "Unknown conditions" in groups[0]["caution"]


def test_retrieval_limits_are_enforced():
    for limit in (0, -1, 13, 1000):
        with pytest.raises(ToolInputError):
            bounded_limit(limit)
    assert bounded_limit(8) == 8


def test_read_only_tool_surface():
    agent = ask.make_agent(scripted({"answerable": False})[0])
    names = {name for ts in agent.toolsets for name in getattr(ts, "tools", {})}
    assert names >= {"search_terms", "narrower_terms", "search_entities", "search_results", "search_training",
                     "search_datasets", "search_contested", "query_objects", "get_object", "get_evidence"}
    assert not names & {"submit", "dry_run", "actions", "read_page", "list_pages", "find_quote", "find_value"}


def seed_hierarchical_robotics(ctx):
    spec = {"domain": "robotics", "base_iri": ROBOTICS, "title": "Robotics", "description": "Test terms.",
            "papers": {"kim": "2406.09246"}, "terms": [
                {"slug": slug, "kind": kind, "pref_label": label,
                 **({"broader": ["libero"]} if slug == "libero-long" else {}),
                 "scope": [{"text": f"{label} is a term.", "quote": {"paper": "kim", "page": 1, "text": quote}}]}
                for (kind, slug, label), quote in TERM_QUOTES.items()]}
    report = Report("test", extra={"written": [], "terms": 0, "types": 0})
    seed_action(ctx, "robotics", spec, report)
    assert not report.findings


def test_search_results_expansion_pagination_and_evidence(ctx, paper):
    seed_hierarchical_robotics(ctx)
    # Existing fixture supplies the exact model/result quote and conditions.
    result = dict(FACT, kind="result", sentence=RESULT_QUOTE, value=53.7, metric="success rate (%)",
                  task="Pick and place", benchmark="LIBERO", suite="LIBERO-Long", embodiment="Franka Panda")
    done = ingest.ingest(ctx, paper, ingest.make_agent(scripted({"facts": [result]})[0]), 32768)
    assert not done.checked.errors
    tools = Retrieval(ctx)
    broad = tools.search_results(suite="LIBERO", include_narrower=True)
    assert len(broad["rows"]) == 1
    assert not tools.search_results(suite="LIBERO", include_narrower=False)["rows"]
    assert not tools.search_results(model="Not present")["rows"]
    assert not tools.search_results(metric="%' OR 1=1 --")["rows"]
    row = broad["rows"][0]
    assert row["value"] == 53.7 and row["evidence_count"] == 1
    assert "evidence" not in row
    assert tools.search_results(model="openvla")["rows"][0]["iri"] == row["iri"]
    assert not tools.search_results(after_iri=row["iri"])["rows"]
    assert tools.read_evidence(row["iri"], [0])["entries"][0]["quote"] == RESULT_QUOTE
    with pytest.raises(ToolInputError):
        tools.read_evidence(row["iri"], [1])
    with pytest.raises(ToolInputError):
        tools.search_results(benchmark="Not an ontology term")


def test_training_and_dataset_search_follow_cross_ontology_roles(ctx):
    collection = "ExampleSet was collected by human demonstrations."
    training = "ExampleModel was pretrained on ExampleSet for cross-embodiment learning."
    path = synthetic_paper(ctx, [collection, training])
    for domain, base, extends, entries in [
        ("datasets", DATASETS, [], [("collection", "human-collected", "Human-collected", collection, []),
                                    ("roles", "pretraining", "Pretraining", training, [])]),
        ("vla-datasets", VLA_DATASETS, ["datasets"],
         [("roles", "cross-embodiment-pretraining", "Cross-embodiment pretraining", training,
           ["datasets/roles/pretraining"])])]:
        spec = {"domain": domain, "base_iri": base, "title": domain, "description": "Synthetic test terms.",
                "extends": extends, "papers": {"test": path.stem}, "terms": [
                    {"slug": slug, "kind": kind, "pref_label": label, "broader": broader,
                     "scope": [{"text": label, "quote": {"paper": "test", "page": 1, "text": quote}}]}
                    for kind, slug, label, quote, broader in entries]}
        report = Report("test", extra={"written": [], "terms": 0, "types": 0})
        seed_action(ctx, domain, spec, report)
        assert not report.findings
    facts = [
        {"kind": "dataset", "entity": "ExampleSet", "sentence": collection, "quote": collection,
         "page": 1, "collection": ["Human-collected"]},
        {"kind": "training", "entity": "ExampleModel", "dataset": "ExampleSet", "sentence": training,
         "quote": training, "page": 1, "role": "Cross-embodiment pretraining"},
    ]
    done = ingest.ingest(ctx, path, ingest.make_agent(scripted({"facts": facts})[0]), 32768)
    assert not done.checked.errors and not done.skipped
    tools = Retrieval(ctx)
    assert len(tools.search_training(role="Pretraining")["rows"]) == 1
    assert not tools.search_training(role="Pretraining", include_narrower=False)["rows"]
    assert len(tools.search_training(model="ExampleModel", dataset="ExampleSet")["rows"]) == 1
    assert len(tools.search_datasets(collection="Human-collected")["rows"]) == 1
    assert tools.search_entities("Example", "datasets")["rows"][0]["title"] == "ExampleSet"


def test_answer_refuses_unsupported_question_and_tracks_usage(ctx):
    model, _ = scripted({"answerable": False, "note": "No matching results are stored."})
    out = ask.answer(ask.make_agent(model), ctx, "What is the answer to an absent result?")
    assert out.answerable is False and not out.claims
    assert out.note and out.usage["requests"] >= 1
    assert isinstance(json.loads(out.conversation_json), list)
    assert "usage" not in out.model_dump()


def test_answer_retries_empty_answerable_output(ctx):
    model, seen = scripted({"answerable": True, "claims": []}, {"answerable": False, "note": "Missing evidence."})
    out = ask.answer(ask.make_agent(model), ctx, "Unsupported question")
    assert not out.answerable
    assert any("at least one" in str(message) for message in seen)


def test_scripted_agent_actually_calls_read_tools_and_cites(ctx, paper):
    done = ingest.ingest(ctx, paper, ingest.make_agent(scripted({"facts": [FACT]})[0]), 32768)
    assert not done.checked.errors
    iri = ctx.lake.con.execute("SELECT iri FROM models WHERE title = ?", ["OpenVLA"]).fetchone()[0]
    before = ctx.lake.snapshot()
    calls = []

    def respond(messages, info):
        calls.append(len(messages))
        if len(calls) == 1:
            return ModelResponse(parts=[ToolCallPart("search_entities", {"name": "OpenVLA", "kind": "models"})])
        if len(calls) == 2:
            return ModelResponse(parts=[ToolCallPart("get_evidence", {"iri": iri, "indices": [0]})])
        assert any(isinstance(p, ToolReturnPart) and p.tool_name == "get_evidence"
                   for m in messages for p in m.parts)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {
            "answerable": True, "claims": [{"text": RESULT_QUOTE, "object": iri, "evidence": [0]}],
        })])

    out = ask.answer(ask.make_agent(FunctionModel(respond)), ctx, "What result was reported for OpenVLA?")
    assert out.answerable and out.claims[0].evidence == [0]
    assert out.usage["requests"] >= 3 and out.usage["tool_calls"] >= 2
    assert ctx.lake.snapshot() == before
