"""Ingest a research paper and add its knowledge to the lake.

The ingestion pipeline:
1. Prepare context-aware PDF chunks.
2. Register the original PDF.
3. Extract structured facts using Pydantic AI.
4. Validate extracted evidence against the original PDF.
5. Convert extracted facts into database rows.
6. Validate the written rows and their evidence.

All database modifications must use typed actions.
"""

from __future__ import annotations

import json

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from kdb_tools import guard
from kdb_tools.actions import Context
from kdb_tools.actions.objects import RegisterMedia
from kdb_tools.agent import Chunk, Quote, extraction_agent
from kdb_tools.media import sha256_of
from kdb_tools.tools import Checked, KdbTools
from pydantic import BaseModel, Field, FiniteFloat
from pydantic_ai import Agent, ModelRetry, RunContext

from .chunking import calculate_prompt_budget, iter_pdf_chunks
from .writer import write_extracted_rows

from .tables import (
    TableResultFact,
    TableEvidenceError,
    locate_table_result,
)


# --------------------------------------------------
# Structured extraction schemas
# --------------------------------------------------


class EvidenceFact(Quote):
    """Base class for facts supported by PDF evidence."""

    entity: str = Field(
        min_length=1,
        description="Name of the entity exactly as identified in the paper",
    )

    sentence: str = Field(
        min_length=1,
        description="One factual statement supported by the quoted evidence",
    )


class Fact(EvidenceFact):
    """Descriptive knowledge about a model, dataset or benchmark."""

    kind: Literal["model", "dataset", "benchmark"]


class TrainingFact(EvidenceFact):
    """A reported relationship between a model and a training dataset."""

    kind: Literal["training"]

    dataset: str = Field(
        min_length=1,
        description="Name of the dataset used to train the model",
    )

    role: str | None = Field(
        default=None,
        description="Reported dataset role, such as pretraining or fine-tuning",
    )


class ResultFact(EvidenceFact):
    """A quantitative experimental result reported in the paper."""

    kind: Literal["result"]

    value: FiniteFloat = Field(
        description="The reported numerical result without conversion",
    )

    metric: str = Field(
        min_length=1,
        description="Reported metric, including its unit when available",
    )

    task: str | None = None
    benchmark: str | None = None
    suite: str | None = None
    embodiment: str | None = None
    perturbation: str | None = None

    trials: int | None = Field(
        default=None,
        gt=0,
        description="Number of trials if explicitly reported",
    )


ExtractedFact = Annotated[
    Fact | TrainingFact | ResultFact | TableResultFact,
    Field(discriminator="kind"),
]


class ChunkFacts(BaseModel):
    """Collection of structured facts extracted from a PDF chunk."""

    facts: list[ExtractedFact] = Field(default_factory=list)


# --------------------------------------------------
# Extraction instructions
# --------------------------------------------------


INSTRUCTIONS = """
You extract structured knowledge from research papers about
vision-language-action models and robotics.

Extract the following types of facts:

1. model:
   Descriptions of models and their characteristics.

2. dataset:
   Descriptions of datasets and their characteristics.

3. benchmark:
   Descriptions of benchmarks and evaluation protocols.

4. training:
   Explicit relationships between models and training datasets.
   Use entity for the model name.

5. result:
   Reported quantitative experimental results supported by prose.
   Use entity for the evaluated model name.

6. table_result:
   Reported quantitative experimental results from structured tables.
   Use entity for the evaluated model name as written in the model cell.

Evidence requirements:

- Every fact must have supporting evidence from the supplied pages.
- For textual facts, copy the quote verbatim from the paper.
- For table_result facts, provide the exact cell contents and indices.
- Page numbers are PDF page numbers starting from 1.
- Do not invent facts, names, values or evidence.
- Ensure the evidence supports the extracted statement.
- Extract only facts relevant to the knowledge database.

Experimental result requirements:

- Preserve the numerical value as reported.
- Preserve the metric and its unit when available.
- Record the task, benchmark, suite, embodiment and perturbation
  when explicitly supported by the supplied evidence.
- Record the number of trials only when explicitly reported.
- Use null for unknown optional fields.
- Do not assume missing experimental conditions.
- Do not confuse results from different models or configurations.
- Resolve references such as "ours" only when the model identity
  is established by the supplied pages.

Table extraction requirements:

- Use table_result for numerical results in structured tables.
- Table, row and column indices start from 1.
- Use the explicit table numbers and R/C indices provided.
- Copy cell_text exactly from the numerical cell.
- Copy entity exactly from the corresponding model cell.
- Copy header exactly from the corresponding column header.
- Identify the model_column and header_row accurately.
- Preserve the reported numerical scale and metric unit.
- Do not guess the meaning of missing or ambiguous cells.
- Use textual result facts for results supported by prose.
- Extract a table result only when its attribution is clear.
- Do not extract the same result twice from the table and prose
  unless they provide distinct supporting evidence.

If the supplied pages contain no relevant supported facts,
return an empty facts list.
"""


# --------------------------------------------------
# Paper and ingestion data structures
# --------------------------------------------------


@dataclass
class Paper:
    """The paper being ingested, as registered in the lake."""

    pdf: Path
    sha256: str

    @property
    def arxiv_id(self) -> str:
        """Return the unversioned arXiv identifier."""

        return self.pdf.stem


@dataclass
class Ingested:
    """Result of a paper ingestion operation."""

    paper: Paper
    facts: list[ExtractedFact]
    rows: list[str]
    checked: Checked = field(default_factory=Checked)


# --------------------------------------------------
# Extraction agent
# --------------------------------------------------


def make_agent(
    model,
    root: Path | None = None,
    settings: dict | None = None,
) -> Agent[Chunk, ChunkFacts]:
    """Create an extraction agent with quote and table validation."""

    agent = extraction_agent(
        model,
        ChunkFacts,
        INSTRUCTIONS,
        root=root,
        settings=settings,
    )

    @agent.output_validator
    def validate_table_results(
        ctx: RunContext[Chunk],
        output: ChunkFacts,
    ) -> ChunkFacts:
        """Verify numerical results against their actual table cells."""

        problems = []

        for fact in output.facts:

            if not isinstance(fact, TableResultFact):
                continue

            # Evidence must belong to the current PDF chunk.
            if fact.page not in ctx.deps.pages:
                problems.append(
                    f"Page {fact.page} is outside the supplied chunk"
                )
                continue

            # Verify the numerical value, model label and column header.
            try:
                locate_table_result(
                    ctx.deps.document,
                    fact,
                )

            except TableEvidenceError as error:
                problems.append(str(error))

        # Ask the model to correct or remove unsupported facts.
        if problems:
            raise ModelRetry(
                "Correct or remove these unsupported table results "
                "and return the complete output again:\n"
                + "\n".join(problems)
            )

        return output

    return agent


def extract(
    agent: Agent[Chunk, ChunkFacts],
    pdf: Path,
    pages: list[int],
    prompt: str,
) -> ChunkFacts:
    """Extract and validate structured facts from a PDF chunk."""

    return agent.run_sync(
        prompt,
        deps=Chunk(pdf, pages),
    ).output


# --------------------------------------------------
# Context management
# --------------------------------------------------


def get_prompt_budget(context: int) -> int:
    """Calculate the estimated token budget available for PDF text."""

    output_schema = json.dumps(
        ChunkFacts.model_json_schema()
    )

    return calculate_prompt_budget(
        context=context,
        instructions=INSTRUCTIONS,
        output_schema=output_schema,
    )


# --------------------------------------------------
# Media registration
# --------------------------------------------------


def register(
    ctx: Context,
    pdf: Path,
    origin: str | None = None,
) -> Paper:
    """Register the original PDF in the lake's media table."""

    pdf = Path(pdf).absolute()

    RegisterMedia().submit(
        ctx,
        {
            "path": str(pdf),
            "origin": origin or f"https://arxiv.org/abs/{pdf.stem}",
        },
    )

    return Paper(
        pdf=pdf,
        sha256=sha256_of(pdf.read_bytes()),
    )


# --------------------------------------------------
# Ingestion pipeline
# --------------------------------------------------


def ingest(
    ctx: Context,
    pdf: Path,
    agent: Agent[Chunk, ChunkFacts],
    context: int,
    pages_per_chunk: int = 3,
    origin: str | None = None,
) -> Ingested:
    """Extract knowledge from a paper and write it to the lake."""

    pdf = Path(pdf).absolute()

    # Calculate the available context budget for PDF text.
    prompt_budget = get_prompt_budget(context)

    # Prepare all chunks before registration or inference.
    # Reject oversized pages before consuming model resources
    # or modifying the database.
    chunks = list(
        iter_pdf_chunks(
            pdf=pdf,
            token_budget=prompt_budget,
            pages_per_chunk=pages_per_chunk,
        )
    )

    # Register the original PDF after successful preflight.
    paper = register(ctx, pdf, origin)

    facts: list[ExtractedFact] = []

    for chunk, prompt in chunks:

        # Retain the professor's original context guard.
        guard.check(prompt, context)

        # Extract facts and validate their textual or table evidence.
        extracted = extract(
            agent,
            paper.pdf,
            chunk.pages,
            prompt,
        )

        facts.extend(extracted.facts)

    # Convert extracted facts into database rows.
    rows = write_rows(ctx, paper, facts)

    # Validate only the rows written during this ingestion.
    return Ingested(
        paper=paper,
        facts=facts,
        rows=rows,
        checked=check(ctx, rows),
    )


# --------------------------------------------------
# Database validation
# --------------------------------------------------


def check(
    ctx: Context,
    rows: list[str],
) -> Checked:
    """Validate the structure and grounding of written rows."""

    if not rows:
        return Checked()

    return KdbTools(
        ctx.root,
        ctx=ctx,
    ).check_pages(rows)


# --------------------------------------------------
# Database row generation
# --------------------------------------------------


def write_rows(
    ctx: Context,
    paper: Paper,
    facts: list[ExtractedFact],
) -> list[str]:
    """Convert extracted facts into grounded database rows."""

    return write_extracted_rows(
        ctx,
        paper,
        facts,
    )