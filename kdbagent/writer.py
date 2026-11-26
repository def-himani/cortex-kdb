"""Convert extracted facts into grounded DuckLake rows."""

from hashlib import sha256

import json
import logging
import re

import pdfplumber

from kdb_tools.actions.objects import UpsertObjects, read_row

from .evidence import (
    EvidenceError,
    locate_quote_evidence,
    locate_table_evidence,
    annotate_and_register,
)

from .tables import TableResultFact, TableEvidenceError

from .resolution import (
    ROBOTICS,
    DATASETS,
    VLA_DATASETS,
    VLA_CLASSES,
    ResolutionError,
    TermResolver,
    EntityResolver,
)


logger = logging.getLogger(__name__)


TABLE_CLASSES = {
    "papers": "paper",
    "models": "model",
    "datasets": "dataset",
    "trained_on": "trained-on",
    "results": "result",
}


def stable_id(*parts) -> str:
    """Generate a deterministic identifier for a relationship or result."""

    text = json.dumps(
        parts,
        sort_keys=True,
        ensure_ascii=False,
    )

    return sha256(text.encode("utf-8")).hexdigest()


def unique_items(items: list) -> list:
    """Remove duplicates while preserving existing list order."""

    result = []

    for item in items:

        if item not in result:
            result.append(item)

    return result


def existing_evidence(row: dict) -> list:
    """Read evidence from a previously stored database row."""

    evidence = row.get("evidence") or []

    if isinstance(evidence, str):
        return json.loads(evidence)

    return evidence


def paper_metadata(paper) -> dict:
    """Read available PDF title metadata."""

    with pdfplumber.open(paper.pdf) as document:

        metadata = document.metadata or {}

        title = metadata.get("Title")

    if not isinstance(title, str) or not title.strip():
        title = f"arXiv:{paper.arxiv_id}"

    return {
        "title": title.strip(),
        "arxiv_id": paper.arxiv_id,
        "media": paper.sha256,
    }


def fact_evidence(paper, fact):
    """Resolve an extracted fact to its source evidence."""

    if isinstance(fact, TableResultFact):

        return locate_table_evidence(
            paper.pdf,
            paper.sha256,
            fact,
        )

    return [
        locate_quote_evidence(
            paper.pdf,
            paper.sha256,
            fact.page,
            fact.quote,
        )
    ]


def numeric_value_present(value: float, text: str) -> bool:
    """Check that a reported numerical value occurs in prose."""

    text = text.replace("−", "-")

    numbers = re.findall(
        r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)",
        text,
    )

    return any(
        float(number) == value
        for number in numbers
    )


def resolve_conditions(resolver, fact) -> dict:
    """Convert reported result conditions to ontology IRIs."""

    return {
        "task": resolver.resolve(
            fact.task,
            ROBOTICS + "tasks/",
        ),
        "benchmark": resolver.resolve(
            fact.benchmark,
            ROBOTICS + "benchmarks/",
        ),
        "suite": resolver.resolve(
            fact.suite,
            ROBOTICS + "benchmarks/",
        ),
        "embodiment": resolver.resolve(
            fact.embodiment,
            ROBOTICS + "embodiments/",
        ),
        "perturbation": resolver.resolve(
            fact.perturbation,
            ROBOTICS + "perturbations/",
        ),
    }


class RowWriter:
    """Prepare and submit grounded rows without overwriting old evidence."""

    def __init__(self, ctx, paper):

        self.ctx = ctx
        self.paper = paper

        self.terms = TermResolver(ctx.root)
        self.entities = EntityResolver(ctx)

        self.rows = {
            table: {}
            for table in TABLE_CLASSES
        }

        self.references = []

    def add_row(
        self,
        table: str,
        iri: str,
        fields: dict,
        references=None,
    ) -> None:
        """Merge a proposed row with previously recorded information."""

        pending = self.rows[table].get(iri)

        if pending is None:

            previous = read_row(
                self.ctx,
                table,
                iri,
            ) or {}

            pending = {"iri": iri}

            if table != "papers":

                pending["evidence"] = (
                    existing_evidence(previous)
                )

            if table in ("models", "datasets"):

                pending["aliases"] = (
                    previous.get("aliases") or []
                )

                if previous.get("description"):
                    pending["description"] = (
                        previous["description"]
                    )

                if previous.get("title"):
                    pending["title"] = previous["title"]

        for key, value in fields.items():

            # Unknown values must not erase existing values.
            if value is None:
                continue

            if key == "aliases":

                pending[key] = unique_items(
                    pending.get(key, []) + value
                )

            elif key == "description":

                descriptions = (
                    pending.get(key, "").split("\n")
                )

                pending[key] = "\n".join(
                    unique_items([
                        item
                        for item in descriptions + [value]
                        if item
                    ])
                )

            elif key == "title" and table in (
                "models",
                "datasets",
            ):

                pending.setdefault(key, value)

            else:
                pending[key] = value

        if references:

            evidence = [
                ref.model_dump(exclude_none=True)
                for ref in references
            ]

            pending["evidence"] = unique_items(
                pending.get("evidence", []) + evidence
            )

            self.references.extend(references)

        self.rows[table][iri] = pending

    def add_paper(self) -> None:
        """Prepare the paper's bibliographic row."""

        iri = (
            self.ctx.iri_base
            + "papers/"
            + self.paper.arxiv_id
        )

        self.add_row(
            "papers",
            iri,
            paper_metadata(self.paper),
        )

    def add_entity(
        self,
        table: str,
        name: str,
        references,
        description: str | None = None,
    ) -> str:
        """Resolve or create a model or dataset."""

        iri = self.entities.resolve(table, name)

        self.add_row(
            table,
            iri,
            {
                "title": name,
                "aliases": [name],
                "description": description,
            },
            references,
        )

        return iri

    def add_training(self, fact, references) -> None:
        """Prepare a grounded model-dataset relationship."""

        role = self.terms.resolve(
            fact.role,
            DATASETS + "roles/",
            VLA_DATASETS + "roles/",
        )

        if role is None:
            raise ResolutionError(
                "Training relationship has no reported role"
            )

        # Resolve both identities before preparing rows.
        model = self.entities.resolve(
            "models",
            fact.entity,
        )

        dataset = self.entities.resolve(
            "datasets",
            fact.dataset,
        )

        self.add_entity(
            "models",
            fact.entity,
            references,
        )

        self.add_entity(
            "datasets",
            fact.dataset,
            references,
        )

        iri = (
            self.ctx.iri_base
            + "trained-on/"
            + stable_id(model, dataset, role)
        )

        self.add_row(
            "trained_on",
            iri,
            {
                "model": model,
                "dataset": dataset,
                "role": role,
            },
            references,
        )

    def add_result(self, fact, references) -> None:
        """Prepare an experimental result under reported conditions."""

        conditions = resolve_conditions(
            self.terms,
            fact,
        )

        if not isinstance(fact, TableResultFact):

            if not numeric_value_present(
                fact.value,
                fact.quote,
            ):
                raise EvidenceError(
                    "Reported number is absent from its quotation"
                )

        model = self.add_entity(
            "models",
            fact.entity,
            references,
        )

        # Keep results from different papers and evidence
        # locations separate instead of overwriting them.
        identifier = stable_id(
            self.paper.arxiv_id,
            model,
            fact.metric,
            fact.value,
            conditions,
            fact.trials,
            references[0].page,
            references[0].bbox,
        )

        iri = (
            self.ctx.iri_base
            + "results/"
            + identifier
        )

        self.add_row(
            "results",
            iri,
            {
                "title": f"{fact.entity}: {fact.metric}",
                "model": model,
                **conditions,
                "metric": fact.metric,
                "value": fact.value,
                "trials": fact.trials,
                "trials_reported": fact.trials is not None,
            },
            references,
        )

    def add_fact(self, fact) -> None:
        """Dispatch an extracted fact to its appropriate table."""

        # Benchmarks currently exist as ontology terms,
        # rather than a separate instance table.
        if fact.kind == "benchmark":
            return

        references = fact_evidence(
            self.paper,
            fact,
        )

        if fact.kind == "model":

            self.add_entity(
                "models",
                fact.entity,
                references,
                description=fact.sentence,
            )

        elif fact.kind == "dataset":

            self.add_entity(
                "datasets",
                fact.entity,
                references,
                description=fact.sentence,
            )

        elif fact.kind == "training":

            self.add_training(
                fact,
                references,
            )

        elif fact.kind in ("result", "table_result"):

            self.add_result(
                fact,
                references,
            )

    def submit(self) -> list[str]:
        """Register evidence and write rows through typed actions."""

        if self.references:

            annotate_and_register(
                self.ctx,
                self.paper.pdf,
                self.references,
            )

        written = []

        for table, class_name in TABLE_CLASSES.items():

            rows = list(
                self.rows[table].values()
            )

            if not rows:
                continue

            params = {
                "class_iri": VLA_CLASSES + class_name,
                "rows": rows,
            }

            action = UpsertObjects()

            plan = action.dry_run(
                self.ctx,
                params,
            )

            action.submit(
                self.ctx,
                params,
                plan,
            )

            written.extend(
                f"{table}/{edit.iri}"
                for edit in plan.row_edits
            )

        return written


def write_extracted_rows(ctx, paper, facts) -> list[str]:
    """Write supported extracted facts to the lake."""

    writer = RowWriter(ctx, paper)

    writer.add_paper()

    for fact in facts:

        try:
            writer.add_fact(fact)

        except (
            ResolutionError,
            EvidenceError,
            TableEvidenceError,
        ) as error:

            logger.warning(
                "Skipped %s fact about %s on page %s: %s",
                fact.kind,
                fact.entity,
                fact.page,
                error,
            )

    return writer.submit()