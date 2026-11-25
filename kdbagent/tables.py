"""Structured PDF table extraction and cell verification."""

from pathlib import Path
from typing import Literal

import json
import re

import pdfplumber

from kdb_tools.locate import find_quote, inside
from pydantic import BaseModel, Field, FiniteFloat


class TableEvidenceError(ValueError):
    """A reported table result could not be verified."""


class TableResultFact(BaseModel):
    """An experimental result extracted from a PDF table."""

    kind: Literal["table_result"]

    entity: str = Field(min_length=1)
    sentence: str = Field(min_length=1)

    page: int = Field(ge=1)
    table: int = Field(ge=1)
    row: int = Field(ge=1)
    column: int = Field(ge=1)

    model_column: int = Field(default=1, ge=1)
    header_row: int = Field(default=1, ge=1)

    header: str = Field(min_length=1)
    cell_text: str = Field(min_length=1)

    value: FiniteFloat
    metric: str = Field(min_length=1)

    task: str | None = None
    benchmark: str | None = None
    suite: str | None = None
    embodiment: str | None = None
    perturbation: str | None = None

    trials: int | None = Field(default=None, gt=0)


def normalize(text: str) -> str:
    """Normalize whitespace without changing the meaning."""

    return " ".join(text.split())


def detect_tables(page):
    """Detect PDF tables in a consistent order."""

    tables = page.find_tables()

    if not tables:
        tables = page.find_tables(
            table_settings={
                "vertical_strategy": "text",
                "horizontal_strategy": "lines",
            }
        )

    return sorted(
        tables,
        key=lambda table: (
            table.bbox[1],
            table.bbox[0],
        ),
    )


def render_tables(page) -> str:
    """Represent detected tables using explicit row and column indices."""

    sections = []

    for table_number, table in enumerate(
        detect_tables(page),
        start=1,
    ):
        sections.append(
            f"=== TABLE {table_number} ==="
        )

        for row_number, row in enumerate(
            table.extract(),
            start=1,
        ):
            cells = {
                f"C{column}": normalize(text)
                if text is not None
                else None
                for column, text in enumerate(
                    row,
                    start=1,
                )
            }

            sections.append(
                f"R{row_number}: "
                + json.dumps(cells, ensure_ascii=False)
            )

    return "\n".join(sections)


def parse_numeric_cell(text: str) -> float:
    """Parse a simple numerical table cell without changing its scale."""

    cleaned = normalize(text).replace("−", "-")

    if not re.fullmatch(
        r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)\s*%?",
        cleaned,
    ):
        raise TableEvidenceError(
            f"Unsupported numerical cell: {text!r}"
        )

    return float(cleaned.rstrip("%").strip())


def cell_evidence(
    page,
    table,
    row: int,
    column: int,
    expected: str,
    selector: str,
) -> dict:
    """Locate exact text inside a specified table cell."""

    rows = table.extract()

    if not 1 <= row <= len(rows):
        raise TableEvidenceError("Invalid table row")

    if not 1 <= column <= len(rows[row - 1]):
        raise TableEvidenceError("Invalid table column")

    actual = rows[row - 1][column - 1]

    if actual is None:
        raise TableEvidenceError("The selected cell is empty")

    if normalize(actual) != normalize(expected):
        raise TableEvidenceError(
            f"Expected {expected!r}, found {actual!r}"
        )

    raw_bbox = table.rows[row - 1].cells[column - 1]

    if raw_bbox is None:
        raise TableEvidenceError("Cell has no bounding box")

    x0, top, x1, bottom = raw_bbox

    # Convert top-origin pdfplumber coordinates to
    # bottom-origin PDF coordinates required by MediaRef.
    cell_bbox = [
        x0,
        page.height - bottom,
        x1,
        page.height - top,
    ]

    matches = [
        bbox
        for bbox in find_quote(page, actual)
        if inside(bbox, cell_bbox)
    ]

    if len(matches) != 1:
        raise TableEvidenceError(
            f"Cannot uniquely locate {actual!r} in the selected cell"
        )

    return {
        "bbox": matches[0],
        selector: normalize(actual),
    }


def locate_table_result(
    pdf: Path,
    fact: TableResultFact,
) -> list[dict]:
    """Verify a result against its value, model and header cells."""

    with pdfplumber.open(pdf) as document:

        if not 1 <= fact.page <= len(document.pages):
            raise TableEvidenceError("Invalid PDF page")

        page = document.pages[fact.page - 1]

        tables = detect_tables(page)

        if not 1 <= fact.table <= len(tables):
            raise TableEvidenceError("Invalid table number")

        table = tables[fact.table - 1]

        if fact.header_row >= fact.row:
            raise TableEvidenceError(
                "The header must precede the result row"
            )

        if fact.model_column == fact.column:
            raise TableEvidenceError(
                "Model and result cells must be different"
            )

        if parse_numeric_cell(fact.cell_text) != fact.value:
            raise TableEvidenceError(
                "Reported value does not match the numerical cell"
            )

        if "%" in fact.cell_text and "%" not in fact.metric:
            raise TableEvidenceError(
                "The percentage unit is missing from the metric"
            )

        value = cell_evidence(
            page,
            table,
            fact.row,
            fact.column,
            fact.cell_text,
            "value",
        )

        model = cell_evidence(
            page,
            table,
            fact.row,
            fact.model_column,
            fact.entity,
            "quote",
        )

        header = cell_evidence(
            page,
            table,
            fact.header_row,
            fact.column,
            fact.header,
            "quote",
        )

        return [
            {"page": fact.page, **item}
            for item in [value, model, header]
        ]