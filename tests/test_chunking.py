"""Offline tests for context-aware PDF chunking."""

import pytest
from fpdf import FPDF
from kdb_tools import guard

from kdbagent.chunking import (
    REQUEST_OVERHEAD_TOKENS,
    calculate_prompt_budget,
    format_page,
    iter_pdf_chunks,
)

from kdbagent.ingest import get_prompt_budget


@pytest.fixture
def synthetic_pdf(tmp_path):

    pdf = FPDF(unit="pt", format="letter")

    for number in range(1, 5):

        pdf.add_page()
        pdf.set_font("Helvetica", size=12)

        pdf.text(
            50,
            100,
            f"Page {number} contains synthetic research evidence.",
        )

    path = tmp_path / "chunking-test.pdf"

    pdf.output(str(path))

    return path


def test_prompt_budget_calculation():

    context = 16384
    instructions = "Extract supported research facts."
    schema = '{"type": "object"}'

    expected = (
        int(context * guard.SHARE)
        - guard.estimate(instructions)
        - guard.estimate(schema)
        - REQUEST_OVERHEAD_TOKENS
    )

    assert calculate_prompt_budget(
        context,
        instructions,
        schema,
    ) == expected

    assert expected > 0


def test_insufficient_context_is_rejected():

    with pytest.raises(guard.PromptTooLong):

        calculate_prompt_budget(
            context=1024,
            instructions="Extract facts.",
            output_schema="{}",
        )


def test_chunks_respect_page_limit(synthetic_pdf):

    chunks = list(
        iter_pdf_chunks(
            synthetic_pdf,
            token_budget=10000,
            pages_per_chunk=2,
        )
    )

    assert len(chunks) == 2

    assert chunks[0][0].pages == [1, 2]
    assert chunks[1][0].pages == [3, 4]


def test_chunks_respect_token_budget(synthetic_pdf):

    page = format_page(
        1,
        "Page 1 contains synthetic research evidence.",
    )

    budget = guard.estimate(page) + 2

    chunks = list(
        iter_pdf_chunks(
            synthetic_pdf,
            token_budget=budget,
            pages_per_chunk=3,
        )
    )

    assert len(chunks) == 4

    for chunk, prompt in chunks:

        assert len(chunk.pages) == 1
        assert guard.estimate(prompt) <= budget


def test_no_pages_are_lost(synthetic_pdf):

    chunks = list(
        iter_pdf_chunks(
            synthetic_pdf,
            token_budget=10000,
            pages_per_chunk=3,
        )
    )

    pages = [
        page
        for chunk, _ in chunks
        for page in chunk.pages
    ]

    assert pages == [1, 2, 3, 4]

    for chunk, prompt in chunks:

        for page in chunk.pages:

            assert f"=== page {page} ===" in prompt


def test_oversized_page_is_rejected(synthetic_pdf):

    with pytest.raises(
        guard.PromptTooLong,
        match="page 1",
    ):

        list(
            iter_pdf_chunks(
                synthetic_pdf,
                token_budget=1,
            )
        )


def test_configured_context_has_available_budget():

    budget = get_prompt_budget(16384)

    assert 0 < budget < 16384