"""Offline tests for table extraction and PDF evidence."""

import pytest
import pdfplumber

from fpdf import FPDF

from kdb_tools.media import sha256_of
from kdb_tools.ground import Paper as GroundPaper, iou

from kdbagent.tables import (
    TableResultFact,
    TableEvidenceError,
    render_tables,
    locate_table_result,
)

from kdbagent.evidence import (
    EvidenceError,
    locate_quote_evidence,
    locate_table_evidence,
    write_highlights,
)


QUOTE = "ExampleVLA is a robotics model."


@pytest.fixture
def synthetic_pdf(tmp_path):

    pdf = FPDF(unit="pt", format="letter")

    pdf.add_page()
    pdf.set_font("Helvetica", size=12)

    pdf.text(50, 70, QUOTE)

    rows = [
        ["Model", "Success"],
        ["ExampleVLA", "85%"],
        ["OtherModel", "85%"],
    ]

    for row_number, row in enumerate(rows):

        for column_number, text in enumerate(row):

            x = 50 + column_number * 140
            y = 120 + row_number * 30

            pdf.rect(x, y, 140, 30)
            pdf.text(x + 10, y + 19, text)

    path = tmp_path / "synthetic.pdf"

    pdf.output(str(path))

    return path


@pytest.fixture
def result_fact():

    return TableResultFact(
        kind="table_result",
        entity="ExampleVLA",
        sentence="ExampleVLA achieved 85% success.",
        page=1,
        table=1,
        row=2,
        column=2,
        model_column=1,
        header_row=1,
        header="Success",
        cell_text="85%",
        value=85.0,
        metric="success rate (%)",
    )


def test_table_rendering(synthetic_pdf):

    with pdfplumber.open(synthetic_pdf) as pdf:

        text = render_tables(pdf.pages[0])

        assert "TABLE 1" in text
        assert "R2" in text
        assert "ExampleVLA" in text
        assert "85%" in text


def test_table_result_localization(
    synthetic_pdf,
    result_fact,
):

    evidence = locate_table_result(
        synthetic_pdf,
        result_fact,
    )

    assert len(evidence) == 3
    assert evidence[0]["value"] == "85%"
    assert evidence[1]["quote"] == "ExampleVLA"
    assert evidence[2]["quote"] == "Success"


def test_wrong_model_row_is_rejected(
    synthetic_pdf,
    result_fact,
):

    wrong = result_fact.model_copy(
        update={"row": 3}
    )

    with pytest.raises(TableEvidenceError):

        locate_table_result(synthetic_pdf, wrong)


def test_wrong_numeric_value_is_rejected(
    synthetic_pdf,
    result_fact,
):

    wrong = result_fact.model_copy(
        update={"value": 99.0}
    )

    with pytest.raises(TableEvidenceError):

        locate_table_result(synthetic_pdf, wrong)


def test_quote_evidence(synthetic_pdf):

    sha = sha256_of(synthetic_pdf.read_bytes())

    evidence = locate_quote_evidence(
        synthetic_pdf,
        sha,
        1,
        QUOTE,
    )

    assert evidence.sha256 == sha
    assert evidence.page == 1
    assert evidence.quote == QUOTE
    assert len(evidence.bbox) == 4


def test_missing_quote_is_rejected(synthetic_pdf):

    sha = sha256_of(synthetic_pdf.read_bytes())

    with pytest.raises(EvidenceError):

        locate_quote_evidence(
            synthetic_pdf,
            sha,
            1,
            "This sentence does not exist.",
        )


def test_annotation_preserves_original(
    synthetic_pdf,
    result_fact,
    tmp_path,
):

    original_hash = sha256_of(
        synthetic_pdf.read_bytes()
    )

    references = locate_table_evidence(
        synthetic_pdf,
        original_hash,
        result_fact,
    )

    destination = tmp_path / "annotated.pdf"

    write_highlights(
        synthetic_pdf,
        destination,
        references,
    )

    assert destination.exists()

    assert sha256_of(
        synthetic_pdf.read_bytes()
    ) == original_hash

    marked = GroundPaper(destination)

    try:
        boxes = marked.highlights(1)

        for reference in references:

            assert any(
                iou(reference.bbox, box) >= 0.5
                for box in boxes
            )

    finally:
        marked.close()


def test_repeated_annotation_is_idempotent(
    synthetic_pdf,
    result_fact,
    tmp_path,
):

    sha = sha256_of(synthetic_pdf.read_bytes())

    references = locate_table_evidence(
        synthetic_pdf,
        sha,
        result_fact,
    )

    destination = tmp_path / "annotated.pdf"

    write_highlights(
        synthetic_pdf,
        destination,
        references,
    )

    first_hash = sha256_of(destination.read_bytes())

    write_highlights(
        synthetic_pdf,
        destination,
        references,
    )

    assert sha256_of(
        destination.read_bytes()
    ) == first_hash