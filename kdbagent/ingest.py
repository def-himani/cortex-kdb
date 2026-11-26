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
import re
from time import perf_counter

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

    aliases: list[str] = Field(default_factory=list)

    model_class: str | None = None

    collection: list[str] = Field(default_factory=list)

    embodiments: list[str] = Field(default_factory=list)

    tasks: list[str] = Field(default_factory=list)

    modalities: list[str] = Field(default_factory=list)


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
MANDATORY OUTPUT FORMAT:

Return one JSON object containing a "facts" array. Each item in
"facts" must be a flat JSON object with a "kind" field. Never wrap a
fact in a nested object named "model", "dataset", "benchmark",
"training", "result", or "table_result".

For every textual fact, put kind, entity, sentence, quote and page
at the same level. Use the exact field names from the schema:
- entity, not name or model
- sentence, not description
- quote, not evidence
- collection (a list), not collection_method
- role, not training_role
- value (a JSON number), not result_value

These are formatting examples only. Do not copy the placeholder
entities, statements, quotations, page numbers or numerical values.
The actual output must contain only facts supported by the supplied pages.

Model:
{"kind":"model","entity":"ExampleModel","sentence":"Supported model description.","quote":"Exact source quotation.","page":1}

Dataset:
{"kind":"dataset","entity":"ExampleDataset","sentence":"Supported dataset description.","quote":"Exact source quotation.","page":1,"collection":["Teleoperation"]}

Training:
{"kind":"training","entity":"ExampleModel","sentence":"Supported training relationship.","quote":"Exact source quotation.","page":1,"dataset":"ExampleDataset","role":"Fine-tuning"}

Result:
{"kind":"result","entity":"ExampleModel","sentence":"Supported experimental result.","quote":"Exact source quotation.","page":1,"value":53.9,"metric":"success rate (%)"}

For table_result, use the supplied table-result schema with the exact
required table and cell coordinates. All fields must be flat, too.
For benchmark, use the same flat textual format as model, with
"kind":"benchmark".

Use actual PDF page numbers. Copy quotations verbatim, not paraphrases.
Preserve the reported number and unit separately (e.g. 53.9% means
"value":53.9 and "metric":"success rate (%)"). Never invent evidence
or missing conditions. Omit unsupported facts. If no relevant facts
are supported, return {"facts":[]}.

Before returning, check every fact for the correct kind, flat structure,
required fields and exact field names. Return the complete facts array.

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

Entity classification requirements:

- Record aliases only when the paper explicitly identifies them
  as alternative names for the same entity.
- Never treat different model variants as aliases.
- For model facts, record model_class when explicitly supported.
- Use Vision-language-action model for a model explicitly
  identified as belonging to that category.

Dataset requirements:

- For dataset facts, record how the dataset was collected.
- Collection methods may include:
  Human-collected, Programmatically generated, Simulated,
  Pooled from existing datasets, Teleoperation, Scripted policy,
  Simulated teleoperation, and Cross-embodiment pooling.
- Prefer the most specific supported collection method.
- Record explicitly supported embodiments, tasks and modalities.
- Use empty lists for unknown list-valued properties.
- Never infer the collection method merely from a dataset's name.

Training relationship requirements:

- Record the reported training role, such as Pretraining,
  Fine-tuning, Co-training, Cross-embodiment pretraining,
  LoRA fine-tuning, or Co-fine-tuning with web data.
- Extract separate dataset facts when the paper describes
  how the training datasets were collected.
- Do not assume a training role when it is not stated.

Result conditions:

- Look for task, benchmark, suite and embodiment in the supplied
  text before extracting an experimental result.
- Do not invent missing conditions to complete a result.
- A missing trial count must not be replaced with an estimate.

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
    version: str | None = None

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
    chunks: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)
    touched_rows: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def usage(self) -> dict[str, int]:
        """Reported model usage, summed across completed chunk runs."""
        keys = ("input_tokens", "output_tokens", "requests", "tool_calls", "cache_read_tokens")
        return {
            key: sum(chunk["usage"].get(key, 0) for chunk in self.chunks)
            for key in keys
        }

    def report(self) -> dict:
        """Machine-readable report for CLI and batch checkpoints."""
        return {
            "arxiv_id": self.paper.arxiv_id,
            "sha256": self.paper.sha256,
            "version": self.paper.version,
            "facts": len(self.facts),
            "rows": list(self.rows),
            "touched_rows": self.touched_rows,
            "skipped": self.skipped,
            "candidates": self.candidates,
            "chunks": self.chunks,
            "usage": self.usage,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "errors": [str(p) for p in self.checked.errors],
            "warnings": [str(p) for p in self.checked.warnings],
        }


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
        root=None,
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

    version = None

    if origin:
        match = re.search(
            rf"{re.escape(pdf.stem)}(v\d+)(?:\.pdf)?(?:$|[?#])",
            origin,
        )

        if match:
            version = match.group(1)

    return Paper(
        pdf=pdf,
        sha256=sha256_of(pdf.read_bytes()),
        version=version,
    )


# --------------------------------------------------
# Ingestion pipeline
# --------------------------------------------------


def ingest(
    ctx: Context,
    pdf: Path,
    agent: Agent[Chunk, ChunkFacts],
    context: int,
    pages_per_chunk: int = 1,
    origin: str | None = None,
) -> Ingested:
    """Extract knowledge from a paper and write it to the lake."""

    started = perf_counter()
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
    diagnostics = []

    for index, (chunk, prompt) in enumerate(chunks, start=1):

        print(
            f"Processing chunk {index}/{len(chunks)} | "
            f"Pages: {chunk.pages} | "
            f"Estimated PDF tokens: {guard.estimate(prompt):,}",
            flush=True,
        )

        guard.check(prompt, context)
        chunk_started = perf_counter()

        # Keep the public extract() helper unchanged for existing callers.
        # Here retain the run result so reported usage includes retries.
        result = agent.run_sync(prompt, deps=chunk)
        usage = result.usage
        facts.extend(result.output.facts)

        keys = ("input_tokens", "output_tokens", "requests", "tool_calls", "cache_read_tokens")

        diagnostics.append({
            "pages": list(chunk.pages),
            "estimated_pdf_tokens": guard.estimate(prompt),
            "facts": len(result.output.facts),
            "usage": {
                key: int(getattr(usage, key, 0) or 0)
                for key in keys
            },
            "elapsed_seconds": round(perf_counter() - chunk_started, 3),
        })

        print(
            f"Completed chunk {index}/{len(chunks)} | "
            f"Facts: {len(result.output.facts)} | "
            f"Input tokens: {usage.input_tokens:,} | "
            f"Output tokens: {usage.output_tokens:,} | "
            f"Elapsed: {perf_counter() - chunk_started:.1f}s",
            flush=True,
        )

    # Preserve the starter's write_rows hook and its list return contract.
    rows = write_rows(ctx, paper, facts)
    touched = list(getattr(rows, "touched", rows))

    return Ingested(
        paper=paper,
        facts=facts,
        rows=list(rows),
        checked=check(ctx, touched),
        chunks=diagnostics,
        skipped=list(getattr(rows, "skipped", [])),
        candidates=list(getattr(rows, "candidates", [])),
        touched_rows=touched,
        elapsed_seconds=perf_counter() - started,
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