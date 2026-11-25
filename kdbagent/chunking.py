"""Context-aware PDF chunking for paper extraction."""

from collections.abc import Iterator
from pathlib import Path

import pdfplumber

from kdb_tools import guard
from kdb_tools.agent import Chunk


REQUEST_OVERHEAD_TOKENS = 3072


def calculate_prompt_budget(
    context: int,
    instructions: str,
    output_schema: str,
) -> int:
    """Calculate the estimated token budget available for PDF text."""

    input_budget = int(context * guard.SHARE)

    instruction_tokens = guard.estimate(instructions)
    schema_tokens = guard.estimate(output_schema)

    available = (
        input_budget
        - instruction_tokens
        - schema_tokens
        - REQUEST_OVERHEAD_TOKENS
    )

    if available <= 0:
        raise guard.PromptTooLong(
            f"Context window of {context} tokens is too small "
            "for extraction instructions, schemas and tool overhead."
        )

    return available


def format_page(page_number: int, text: str) -> str:
    """Format a PDF page with its original one-based page number."""

    return f"=== page {page_number} ===\n{text}"


def iter_pdf_chunks(
    pdf: Path,
    token_budget: int,
    pages_per_chunk: int = 3,
) -> Iterator[tuple[Chunk, str]]:
    """Group consecutive PDF pages within the estimated token budget."""

    if token_budget <= 0:
        raise ValueError("token_budget must be positive")

    if pages_per_chunk <= 0:
        raise ValueError("pages_per_chunk must be positive")

    pdf = Path(pdf).absolute()

    chunk_pages = []
    chunk_texts = []

    with pdfplumber.open(pdf) as document:

        for page_number, page in enumerate(
            document.pages,
            start=1,
        ):
            text = page.extract_text() or ""

            formatted = format_page(page_number, text)

            page_tokens = guard.estimate(formatted)

            # Never silently truncate an oversized page.
            if page_tokens > token_budget:
                raise guard.PromptTooLong(
                    f"{pdf.name}, page {page_number}: "
                    f"approximately {page_tokens} tokens exceeds "
                    f"the available budget of {token_budget} tokens."
                )

            candidate = "\n".join(
                chunk_texts + [formatted]
            )

            exceeds_budget = (
                guard.estimate(candidate) > token_budget
            )

            exceeds_page_limit = (
                len(chunk_pages) >= pages_per_chunk
            )

            if chunk_pages and (
                exceeds_budget or exceeds_page_limit
            ):
                yield (
                    Chunk(pdf, chunk_pages.copy()),
                    "\n".join(chunk_texts),
                )

                chunk_pages = []
                chunk_texts = []

            chunk_pages.append(page_number)
            chunk_texts.append(formatted)

        if chunk_pages:
            yield (
                Chunk(pdf, chunk_pages.copy()),
                "\n".join(chunk_texts),
            )