"""Evidence localization and annotated PDF registration."""

from pathlib import Path
from tempfile import TemporaryDirectory

import shutil
import pdfplumber

from kdb_tools.actions import Context
from kdb_tools.actions.objects import RegisterMedia
from kdb_tools.annotate import highlight
from kdb_tools.ground import Paper as GroundPaper, iou
from kdb_tools.locate import find_quote
from kdb_tools.media import MediaRef, sha256_of

from .tables import TableResultFact, locate_table_result


class EvidenceError(ValueError):
    """Evidence could not be located or annotated reliably."""


def locate_quote_evidence(
    pdf: Path,
    sha256: str,
    page_number: int,
    quote: str,
) -> MediaRef:
    """Create a grounded MediaRef for an exact PDF quotation."""

    with pdfplumber.open(pdf) as document:

        if not 1 <= page_number <= len(document.pages):
            raise EvidenceError(
                f"Invalid PDF page: {page_number}"
            )

        page = document.pages[page_number - 1]

        boxes = find_quote(page, quote)

        if not boxes:
            raise EvidenceError(
                f"Quote not found on page {page_number}: {quote}"
            )

        if len(boxes) != 1:
            raise EvidenceError(
                f"Ambiguous quote on page {page_number}: {quote}"
            )

        return MediaRef(
            sha256=sha256,
            page=page_number,
            bbox=boxes[0],
            quote=quote,
        )


def locate_table_evidence(
    pdf: Path,
    sha256: str,
    fact: TableResultFact,
) -> list[MediaRef]:
    """Create grounded references for a verified table result."""

    return [
        MediaRef(
            sha256=sha256,
            **item,
        )
        for item in locate_table_result(pdf, fact)
    ]


def write_highlights(
    pdf: Path,
    destination: Path,
    references: list[MediaRef],
) -> Path:
    """Add evidence highlights while preserving existing annotations."""

    pdf = Path(pdf).resolve()
    destination = Path(destination).resolve()

    if pdf == destination:
        raise EvidenceError(
            "The original PDF must never be modified"
        )

    if not references:
        raise EvidenceError("No evidence supplied")

    original_hash = sha256_of(pdf.read_bytes())

    existing = None
    marks = []
    seen = set()

    if destination.exists():
        existing = GroundPaper(destination)

    try:
        for index, ref in enumerate(references):

            if ref.sha256 != original_hash:
                raise EvidenceError(
                    "Evidence refers to a different PDF"
                )

            text = (
                ref.quote
                if ref.quote is not None
                else ref.value
            )

            if not text or ref.page is None or ref.bbox is None:
                raise EvidenceError(
                    "Incomplete PDF evidence reference"
                )

            key = (
                ref.page,
                text,
                tuple(ref.bbox),
            )

            if key in seen:
                continue

            seen.add(key)

            # Avoid adding duplicate highlights on repeated ingestion.
            if existing is not None:
                if ref.page <= existing.pages:
                    existing_boxes = existing.highlights(ref.page)

                    if any(
                        iou(ref.bbox, box) >= 0.99
                        for box in existing_boxes
                    ):
                        continue

            marks.append(
                (
                    ref.page,
                    text,
                    f"Cortex KDB evidence {index}",
                    ref.bbox,
                )
            )

    finally:
        if existing is not None:
            existing.close()

    if not marks and destination.exists():
        return destination

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Write to a temporary copy first so an annotation failure
    # cannot damage an existing annotated document.
    with TemporaryDirectory(
        dir=destination.parent
    ) as directory:

        temporary = Path(directory) / destination.name

        if destination.exists():
            shutil.copy2(destination, temporary)

        found = highlight(
            pdf,
            temporary,
            marks,
        )

        if not all(found):
            raise EvidenceError(
                "One or more evidence highlights could not be created"
            )

        temporary.replace(destination)

    return destination


def annotate_and_register(
    ctx: Context,
    pdf: Path,
    references: list[MediaRef],
) -> Path:
    """Create and register an annotated copy using a typed action."""

    original_hash = sha256_of(pdf.read_bytes())

    destination = (
        ctx.root
        / "annotated"
        / pdf.name
    )

    marked = write_highlights(
        pdf,
        destination,
        references,
    )

    RegisterMedia().submit(
        ctx,
        {
            "path": str(marked),
            "origin": "Cortex KDB evidence annotations",
            "annotates": original_hash,
        },
    )

    return marked