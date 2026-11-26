"""Convert extracted facts into grounded DuckLake rows."""

from hashlib import sha256

import json
import logging
import re

import pdfplumber

from kdb_tools.actions.objects import UpsertObjects, read_row
from kdb_tools.validate import normalize_name as profile_name

from .conflicts import result_candidates

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
    normalize_name,
    entity_iri,
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
        "version": paper.version,
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


class WrittenRows(list):
    """A list of changed row paths, with diagnostics for the caller."""

    def __init__(self, changed, *, touched, skipped, candidates):
        super().__init__(changed)
        self.touched = touched
        self.skipped = skipped
        self.candidates = candidates


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
        self.skipped = []
        # Match the course's cross-table uniqueness rule, but do not use
        # its lossy normalization to silently merge different variants.
        self.name_owners = {}
        for table in ("papers", "models", "datasets"):
            columns = "iri, title" if table == "papers" else "iri, title, aliases"
            for row in ctx.lake.con.execute(f"SELECT {columns} FROM {table}").fetchall():
                iri, title = row[:2]
                aliases = row[2] or [] if len(row) == 3 else []
                self.remember_names(table, iri, [title, *aliases])

    def remember_names(self, table, iri, names):
        for name in names:
            if name and profile_name(name):
                self.name_owners.setdefault(profile_name(name), set()).add((table, iri))

    def check_names(self, table, iri, names):
        for name in names:
            if not name or not profile_name(name):
                continue
            owners = self.name_owners.get(profile_name(name), set())
            if owners - {(table, iri)}:
                raise ResolutionError(
                    f"Name collision under course normalization: {name!r}; "
                    "explicit identity review required"
                )

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

                for key in (
                    "collection",
                    "embodiments",
                    "tasks",
                    "modalities",
                    "model_class",
                ):
                    if previous.get(key) is not None:
                        pending[key] = previous[key]

        for key, value in fields.items():

            # Unknown values must not erase existing values.
            if value is None:
                continue

            if key in (
                "aliases",
                "collection",
                "embodiments",
                "tasks",
                "modalities",
            ):

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

        previous = read_row(self.ctx, "papers", iri) or {}
        if previous.get("media") and previous["media"] != self.paper.sha256:
            raise ResolutionError(
                f"Paper {self.paper.arxiv_id} already has different registered bytes"
            )
        fields = paper_metadata(self.paper)
        if previous.get("version") and fields.get("version") and previous["version"] != fields["version"]:
            raise ResolutionError(f"Conflicting paper version for {self.paper.arxiv_id}")
        self.check_names("papers", iri, [fields["title"]])
        self.add_row("papers", iri, fields)
        self.remember_names("papers", iri, [fields["title"]])

    def add_entity(
        self,
        table: str,
        name: str,
        references,
        description: str | None = None,
        properties: dict | None = None,
    ) -> str:
        """Resolve or create a grounded model or dataset."""

        if table == "models" and normalize_name(name) in {
            "ours", "our model", "our method", "proposed model", "proposed method", "this work",
        }:
            raise ResolutionError(f"Unresolved model identity: {name!r}")
        properties = dict(properties or {})

        aliases = unique_items(
            [name] + properties.pop("aliases", [])
        )

        # Only an exact normalized name or an explicitly extracted alias
        # can establish identity. Refuse ambiguous merges.
        matches = set()
        for alias in aliases:
            matches.update(self.entities.index.get((table, normalize_name(alias)), set()))
        if len(matches) > 1:
            raise ResolutionError(f"Aliases identify multiple {table} rows: {aliases!r}")
        iri = next(iter(matches)) if matches else entity_iri(self.ctx.iri_base, table, name)
        self.check_names(table, iri, aliases)

        previous = (
            self.rows[table].get(iri)
            or read_row(self.ctx, table, iri)
            or {}
        )

        if table == "datasets":

            collection = (
                previous.get("collection") or []
            ) + (
                properties.get("collection") or []
            )

            if not collection:

                raise ResolutionError(
                    f"Dataset {name!r} has no supported "
                    "collection method"
                )

        if table == "models":

            old_class = previous.get("model_class")
            new_class = properties.get("model_class")

            if old_class and new_class and old_class != new_class:

                raise ResolutionError(
                    f"Conflicting model classification for {name!r}"
                )

        self.add_row(
            table,
            iri,
            {
                "title": name,
                "aliases": aliases,
                "description": description,
                **properties,
            },
            references,
        )

        for alias in aliases:

            self.entities.remember(
                table,
                alias,
                iri,
            )

        self.remember_names(table, iri, aliases)
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

        # Validate the dataset first: no placeholder dataset without collection.
        dataset = self.add_entity("datasets", fact.dataset, references)
        model = self.add_entity("models", fact.entity, references)

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

        # Required experimental conditions must be established
        # before creating a Result row.
        required_conditions = (
            "task",
            "benchmark",
            "suite",
            "embodiment",
        )

        missing = [
            name
            for name in required_conditions
            if not conditions.get(name)
        ]

        if missing:
            raise ResolutionError(
                "Missing required result conditions: "
                + ", ".join(missing)
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

            model_class = self.terms.resolve(
                fact.model_class,
                ROBOTICS + "models/",
            )

            self.add_entity(
                "models",
                fact.entity,
                references,
                description=fact.sentence,
                properties={
                    "aliases": fact.aliases,
                    "model_class": model_class,
                },
            )

        elif fact.kind == "dataset":

            collection = [
                self.terms.resolve(
                    name,
                    DATASETS + "collection/",
                    VLA_DATASETS + "collection/",
                )
                for name in fact.collection
            ]

            embodiments = [
                self.terms.resolve(
                    name,
                    ROBOTICS + "embodiments/",
                )
                for name in fact.embodiments
            ]

            tasks = [
                self.terms.resolve(
                    name,
                    ROBOTICS + "tasks/",
                )
                for name in fact.tasks
            ]

            # Only resolve modalities that exist as ontology terms.
            modalities = []

            for name in fact.modalities:

                try:
                    modalities.append(
                        self.terms.resolve(
                            name,
                            DATASETS + "modalities/",
                            VLA_DATASETS + "modalities/",
                            ROBOTICS + "modalities/",
                        )
                    )

                except ResolutionError as error:

                    logger.warning(
                        "Unresolved modality: %s",
                        error,
                    )

            self.add_entity(
                "datasets",
                fact.entity,
                references,
                description=fact.sentence,
                properties={
                    "aliases": fact.aliases,
                    "collection": [
                        item for item in collection if item
                    ],
                    "embodiments": [
                        item for item in embodiments if item
                    ],
                    "tasks": [
                        item for item in tasks if item
                    ],
                    "modalities": [
                        item for item in modalities if item
                    ],
                },
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
        touched = []

        # Dry-run every table before beginning row writes. Re-plan at
        # submission because each committed action advances the snapshot.
        for table, class_name in TABLE_CLASSES.items():
            rows = list(self.rows[table].values())
            if rows:
                UpsertObjects().dry_run(self.ctx, {
                    "class_iri": VLA_CLASSES + class_name,
                    "rows": rows,
                })

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
            touched.extend(f"{table}/{row['iri']}" for row in rows)

        candidates = result_candidates(self.ctx, set(self.rows["results"]))
        return WrittenRows(written, touched=touched, skipped=self.skipped, candidates=candidates)


def write_extracted_rows(ctx, paper, facts) -> list[str]:
    """Write supported extracted facts to the lake."""

    writer = RowWriter(ctx, paper)

    writer.add_paper()

    # Establish models and datasets before creating relationships.
    priority = {
        "model": 0,
        "dataset": 1,
        "benchmark": 2,
        "training": 3,
        "result": 4,
        "table_result": 4,
    }

    ordered_facts = sorted(
        facts,
        key=lambda fact: priority.get(fact.kind, 5),
    )

    for fact in ordered_facts:

        if fact.kind == "benchmark":
            writer.skipped.append({
                "kind": fact.kind, "entity": fact.entity, "page": fact.page,
                "reason": "Benchmark facts are ontology-only in the current schema",
                "fact": fact.model_dump(mode="json"),
            })
            continue
        try:
            writer.add_fact(fact)

        except (
            ResolutionError,
            EvidenceError,
            TableEvidenceError,
        ) as error:

            writer.skipped.append({
                "kind": fact.kind, "entity": fact.entity, "page": fact.page,
                "reason": str(error), "fact": fact.model_dump(mode="json"),
            })
            logger.warning(
                "Skipped %s fact about %s on page %s: %s",
                fact.kind,
                fact.entity,
                fact.page,
                error,
            )

    return writer.submit()
