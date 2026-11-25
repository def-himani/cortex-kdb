import pytest
from pydantic import ValidationError

from kdbagent.ingest import (
    ChunkFacts,
    Fact,
    TrainingFact,
    ResultFact,
)


def test_model_fact():
    output = ChunkFacts.model_validate({
        "facts": [{
            "kind": "model",
            "entity": "ExampleModel",
            "sentence": "ExampleModel is a VLA model.",
            "quote": "ExampleModel is a VLA model.",
            "page": 1,
        }]
    })

    assert isinstance(output.facts[0], Fact)


def test_training_fact():
    output = ChunkFacts.model_validate({
        "facts": [{
            "kind": "training",
            "entity": "ExampleModel",
            "dataset": "ExampleDataset",
            "role": "pretraining",
            "sentence": "ExampleModel was pretrained on ExampleDataset.",
            "quote": "We pretrained ExampleModel on ExampleDataset.",
            "page": 2,
        }]
    })

    fact = output.facts[0]

    assert isinstance(fact, TrainingFact)
    assert fact.dataset == "ExampleDataset"
    assert fact.role == "pretraining"


def test_result_fact():
    output = ChunkFacts.model_validate({
        "facts": [{
            "kind": "result",
            "entity": "ExampleModel",
            "sentence": "ExampleModel achieved 85 percent success.",
            "quote": "ExampleModel achieved a success rate of 85%.",
            "page": 3,
            "value": 85.0,
            "metric": "success rate (%)",
            "benchmark": "ExampleBenchmark",
            "trials": 100,
        }]
    })

    fact = output.facts[0]

    assert isinstance(fact, ResultFact)
    assert fact.value == 85.0
    assert fact.trials == 100


def test_missing_conditions_remain_unknown():
    fact = ResultFact(
        kind="result",
        entity="ExampleModel",
        sentence="The model achieved 85 percent success.",
        quote="The model achieved 85% success.",
        page=1,
        value=85.0,
        metric="success rate (%)",
    )

    assert fact.task is None
    assert fact.embodiment is None
    assert fact.perturbation is None
    assert fact.trials is None


def test_result_rejects_invalid_trial_count():
    with pytest.raises(ValidationError):
        ResultFact(
            kind="result",
            entity="ExampleModel",
            sentence="The model achieved 85 percent success.",
            quote="The model achieved 85% success.",
            page=1,
            value=85.0,
            metric="success rate (%)",
            trials=-10,
        )


def test_unknown_fact_kind_is_rejected():
    with pytest.raises(ValidationError):
        ChunkFacts.model_validate({
            "facts": [{
                "kind": "unknown",
                "entity": "ExampleModel",
                "sentence": "Example statement.",
                "quote": "Example statement.",
                "page": 1,
            }]
        })


def test_empty_extraction_is_valid():
    output = ChunkFacts(facts=[])

    assert output.facts == []