"""Offline integration tests for the paper extraction agent."""

import pytest

from fpdf import FPDF
from pydantic_ai.messages import (
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from kdbagent.ingest import (
    Fact,
    TrainingFact,
    ResultFact,
    make_agent,
    extract,
)


MODEL_QUOTE = "ExampleVLA is a vision-language-action model."

TRAINING_QUOTE = (
    "ExampleVLA was pretrained on ExampleDataset."
)

RESULT_QUOTE = (
    "ExampleVLA achieved a success rate of 85%."
)

SECOND_PAGE_QUOTE = (
    "The evaluation used 100 trials."
)


# Synthetic extraction outputs

MODEL_FACT = {
    "kind": "model",
    "entity": "ExampleVLA",
    "sentence": "ExampleVLA is a VLA model.",
    "quote": MODEL_QUOTE,
    "page": 1,
}

TRAINING_FACT = {
    "kind": "training",
    "entity": "ExampleVLA",
    "dataset": "ExampleDataset",
    "role": "pretraining",
    "sentence": TRAINING_QUOTE,
    "quote": TRAINING_QUOTE,
    "page": 1,
}

RESULT_FACT = {
    "kind": "result",
    "entity": "ExampleVLA",
    "sentence": RESULT_QUOTE,
    "quote": RESULT_QUOTE,
    "page": 1,
    "value": 85.0,
    "metric": "success rate (%)",
}


@pytest.fixture
def synthetic_pdf(tmp_path):
    """Create a temporary two-page PDF with known evidence."""

    pdf = FPDF(unit="pt", format="letter")

    pdf.add_page()
    pdf.set_font("Helvetica", size=12)

    pdf.text(50, 100, MODEL_QUOTE)
    pdf.text(50, 130, TRAINING_QUOTE)
    pdf.text(50, 160, RESULT_QUOTE)

    pdf.add_page()
    pdf.text(50, 100, SECOND_PAGE_QUOTE)

    path = tmp_path / "synthetic-paper.pdf"
    pdf.output(str(path))

    return path


def scripted_model(*outputs):
    """Return predefined model responses and record retry messages."""

    retries = []
    calls = 0

    def respond(messages, info: AgentInfo) -> ModelResponse:
        nonlocal calls

        for message in messages:
            for part in message.parts:
                if isinstance(part, RetryPromptPart):
                    retries.append(str(part.content))

        output = outputs[min(calls, len(outputs) - 1)]
        calls += 1

        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    output,
                )
            ]
        )

    return FunctionModel(respond), retries


def test_agent_extracts_multiple_fact_types(synthetic_pdf):
    """The agent accepts correctly grounded mixed fact types."""

    model, retries = scripted_model({
        "facts": [
            MODEL_FACT,
            TRAINING_FACT,
            RESULT_FACT,
        ]
    })

    agent = make_agent(model)

    output = extract(
        agent,
        synthetic_pdf,
        [1],
        "\n".join([
            MODEL_QUOTE,
            TRAINING_QUOTE,
            RESULT_QUOTE,
        ]),
    )

    assert len(output.facts) == 3

    assert isinstance(output.facts[0], Fact)
    assert isinstance(output.facts[1], TrainingFact)
    assert isinstance(output.facts[2], ResultFact)

    assert output.facts[2].value == 85.0

    assert retries == []


def test_agent_rejects_fabricated_result_quote(synthetic_pdf):
    """A fabricated numerical result triggers an evidence retry."""

    fabricated_fact = {
        **RESULT_FACT,
        "value": 99.0,
        "sentence": "ExampleVLA achieved 99% success.",
        "quote": "ExampleVLA achieved a success rate of 99%.",
    }

    model, retries = scripted_model(
        {"facts": [fabricated_fact]},
        {"facts": [RESULT_FACT]},
    )

    agent = make_agent(model)

    output = extract(
        agent,
        synthetic_pdf,
        [1],
        RESULT_QUOTE,
    )

    assert len(output.facts) == 1
    assert output.facts[0].value == 85.0

    assert len(retries) > 0
    assert any("99%" in message for message in retries)


def test_agent_rejects_fabricated_training_quote(synthetic_pdf):
    """An unsupported training relationship triggers a retry."""

    fabricated_fact = {
        **TRAINING_FACT,
        "dataset": "UnknownDataset",
        "sentence": "ExampleVLA was pretrained on UnknownDataset.",
        "quote": "ExampleVLA was pretrained on UnknownDataset.",
    }

    model, retries = scripted_model(
        {"facts": [fabricated_fact]},
        {"facts": [TRAINING_FACT]},
    )

    agent = make_agent(model)

    output = extract(
        agent,
        synthetic_pdf,
        [1],
        TRAINING_QUOTE,
    )

    assert output.facts[0].dataset == "ExampleDataset"

    assert len(retries) > 0
    assert any(
        "UnknownDataset" in message
        for message in retries
    )


def test_agent_rejects_evidence_outside_chunk(synthetic_pdf):
    """Evidence must come from the pages supplied to the agent."""

    outside_chunk_fact = {
        "kind": "benchmark",
        "entity": "Example evaluation",
        "sentence": SECOND_PAGE_QUOTE,
        "quote": SECOND_PAGE_QUOTE,
        "page": 2,
    }

    model, retries = scripted_model(
        {"facts": [outside_chunk_fact]},
        {"facts": []},
    )

    agent = make_agent(model)

    output = extract(
        agent,
        synthetic_pdf,
        [1],
        MODEL_QUOTE,
    )

    assert output.facts == []

    assert len(retries) > 0
    assert any("page 2" in message for message in retries)


def test_agent_accepts_empty_extraction(synthetic_pdf):
    """The agent may return no facts rather than invent evidence."""

    model, retries = scripted_model({"facts": []})

    agent = make_agent(model)

    output = extract(
        agent,
        synthetic_pdf,
        [1],
        MODEL_QUOTE,
    )

    assert output.facts == []
    assert retries == []