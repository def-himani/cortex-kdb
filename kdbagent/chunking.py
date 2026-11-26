"""Context-aware PDF chunking for paper extraction."""

from collections.abc import Iterator
from pathlib import Path

import pdfplumber
from pdfplumber.utils import cluster_objects

from kdb_tools import guard
from kdb_tools.agent import Chunk

from .tables import render_tables


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


def extract_page_text(page) -> str:
    """Extract text using geometric column detection."""

    words = page.extract_words(
        x_tolerance=1.5,
        y_tolerance=3,
    )

    if not words:
        return page.extract_text() or ""

    x0, top, x1, bottom = page.bbox

    midpoint = (x0 + x1) / 2

    lines = cluster_objects(
        words,
        "bottom",
        tolerance=2.5,
    )

    two_sided = 0
    separated = []

    for line in lines:

        left = [
            word for word in line
            if (word["x0"] + word["x1"]) / 2 < midpoint
        ]

        right = [
            word for word in line
            if (word["x0"] + word["x1"]) / 2 >= midpoint
        ]

        if not left or not right:
            continue

        two_sided += 1

        gap = (
            min(word["x0"] for word in right)
            - max(word["x1"] for word in left)
        )

        if gap >= 4:
            separated.append(line)

    # Detect a persistent central gutter.
    is_two_column = (
        len(separated) >= 6
        and two_sided > 0
        and len(separated) / two_sided >= 0.60
    )

    # Preserve ordinary extraction for single-column pages.
    if not is_two_column:
        return page.extract_text(x_tolerance=1.5) or ""

    # Identify the main two-column region.
    body_top = min(
        word["top"]
        for line in separated
        for word in line
    )

    body_bottom = max(
        word["bottom"]
        for line in separated
        for word in line
    )

    def read_region(bounds):
        region = page.crop(bounds)

        return region.extract_text(
            x_tolerance=1.5
        ) or ""

    sections = []

    # Preserve full-width text above the column region.
    if body_top > top:
        sections.append(
            read_region(
                (x0, top, x1, body_top)
            )
        )

    # Read the complete left column first.
    sections.append(
        read_region(
            (x0, body_top, midpoint, body_bottom)
        )
    )

    # Then read the complete right column.
    sections.append(
        read_region(
            (midpoint, body_top, x1, body_bottom)
        )
    )

    # Preserve full-width text below the column region.
    if body_bottom < bottom:
        sections.append(
            read_region(
                (x0, body_bottom, x1, bottom)
            )
        )

    return "\n\n".join(
        section
        for section in sections
        if section.strip()
    )


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
            # Preserve the reading order of multi-column papers.
            text = extract_page_text(page)

            formatted = format_page(page_number, text)

            tables = render_tables(page)

            if tables:
                formatted += "\n\n" + tables

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