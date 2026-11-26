"""Offline tests for entity identification and ontology resolution."""

import pytest

from kdbagent.resolution import (
    TermResolver,
    ResolutionError,
    normalize_name,
    entity_iri,
    ROBOTICS,
)


def test_name_normalization():

    assert normalize_name("  OpenVLA  ") == "openvla"


def test_entity_identifier_is_stable():

    base = "https://data.aegean.ai/vla/test/"

    first = entity_iri(
        base,
        "models",
        "OpenVLA",
    )

    second = entity_iri(
        base,
        "models",
        "openvla",
    )

    assert first == second


def test_model_variants_remain_distinct():

    base = "https://data.aegean.ai/vla/test/"

    names = [
        "OpenVLA",
        "OpenVLA-OFT",
        "OpenVLA-OFT+",
        "pi0",
        "pi0.5",
    ]

    identifiers = {
        entity_iri(base, "models", name)
        for name in names
    }

    assert len(identifiers) == len(names)


@pytest.fixture
def resolver():

    resolver = TermResolver.__new__(TermResolver)

    libero = ROBOTICS + "benchmarks/libero"
    long = ROBOTICS + "benchmarks/libero-long"

    resolver.terms = {libero, long}

    resolver.index = {
        "libero": {libero},
        "libero-long": {long},
        "libero-10": {long},
        "ambiguous": {libero, long},
    }

    return resolver


def test_resolve_ontology_alias(resolver):

    assert resolver.resolve(
        "LIBERO-10",
        ROBOTICS + "benchmarks/",
    ) == ROBOTICS + "benchmarks/libero-long"


def test_unknown_term_is_rejected(resolver):

    with pytest.raises(ResolutionError):

        resolver.resolve(
            "UnknownBenchmark",
            ROBOTICS + "benchmarks/",
        )


def test_wrong_term_category_is_rejected(resolver):

    with pytest.raises(ResolutionError):

        resolver.resolve(
            "LIBERO",
            ROBOTICS + "tasks/",
        )


def test_ambiguous_term_is_rejected(resolver):

    with pytest.raises(ResolutionError):

        resolver.resolve(
            "ambiguous",
            ROBOTICS + "benchmarks/",
        )


def test_missing_term_remains_unknown(resolver):

    assert resolver.resolve(
        None,
        ROBOTICS + "benchmarks/",
    ) is None