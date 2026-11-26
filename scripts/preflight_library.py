"""Inspect the PDF library without invoking a language model."""

from pathlib import Path

import pdfplumber

from kdb_tools import config, guard

from kdbagent.chunking import iter_pdf_chunks
from kdbagent.ingest import get_prompt_budget
from kdbagent.library import PAPER_VERSIONS, HELD_OUT_PAPER


def main() -> int:

    root = Path(__file__).resolve().parents[1]

    cfg = config.load(
        root / "kdb.toml"
    )

    budget = get_prompt_budget(cfg.context)

    print(f"Model: {cfg.model}")
    print(f"Context: {cfg.context:,}")
    print(f"PDF text budget: {budget:,}")
    print()

    total_pages = 0
    total_chunks = 0
    total_tables = 0
    total_tokens = 0

    failures = []

    for arxiv_id, version in sorted(
        PAPER_VERSIONS.items()
    ):

        path = (
            root
            / "kdb"
            / "papers"
            / f"{arxiv_id}.pdf"
        )

        try:
            with pdfplumber.open(path) as document:

                page_count = len(document.pages)

            chunks = list(
                iter_pdf_chunks(
                    path,
                    token_budget=budget,
                    pages_per_chunk=3,
                )
            )

            estimates = [
                guard.estimate(prompt)
                for _, prompt in chunks
            ]

            tables = sum(
                prompt.count("=== TABLE ")
                for _, prompt in chunks
            )

            largest = max(estimates, default=0)

            total_pages += page_count
            total_chunks += len(chunks)
            total_tables += tables
            total_tokens += sum(estimates)

            label = (
                " [HELD OUT]"
                if arxiv_id == HELD_OUT_PAPER
                else ""
            )

            print(
                f"{arxiv_id}{version}: "
                f"{page_count} pages, "
                f"{len(chunks)} chunks, "
                f"{tables} tables, "
                f"largest {largest:,} tokens"
                f"{label}"
            )

        except Exception as error:

            failures.append(
                (arxiv_id, str(error))
            )

            print(
                f"FAILED {arxiv_id}: {error}"
            )

    print("\n--- Library summary ---")

    print(f"Papers expected: {len(PAPER_VERSIONS)}")
    print(f"Papers passed: {len(PAPER_VERSIONS) - len(failures)}")
    print(f"Pages processed: {total_pages:,}")
    print(f"Total chunks: {total_chunks:,}")
    print(f"Detected tables: {total_tables:,}")
    print(f"Estimated PDF prompt tokens: {total_tokens:,}")
    print(f"Failed papers: {len(failures)}")

    print("\nModel requests: 0")

    if failures:

        print("\n--- Failures ---")

        for arxiv_id, error in failures:
            print(f"{arxiv_id}: {error}")

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())