"""Ontology term resolution and stable entity identification."""

import unicodedata
from collections import defaultdict
from urllib.parse import quote

from kdb_tools import bundle


ROBOTICS = "https://kdb.aegean.ai/robotics/"
DATASETS = "https://kdb.aegean.ai/datasets/"
VLA_DATASETS = "https://kdb.aegean.ai/vla-datasets/"
VLA_CLASSES = "https://kdb.aegean.ai/vla/classes/"


class ResolutionError(ValueError):
    """An entity or ontology term cannot be resolved safely."""


def normalize_name(value: str) -> str:
    """Normalize names without discarding meaningful punctuation."""

    value = unicodedata.normalize("NFKC", value)

    return " ".join(value.split()).casefold()


def entity_iri(
    iri_base: str,
    table: str,
    name: str,
) -> str:
    """Generate a stable team-specific entity IRI."""

    normalized = normalize_name(name)

    if not normalized:
        raise ResolutionError("Entity name cannot be empty")

    identifier = quote(normalized, safe="-._")

    return f"{iri_base}{table}/{identifier}"


class TermResolver:
    """Resolve names and aliases against seeded ontology terms."""

    def __init__(self, root):

        self.index = defaultdict(set)
        self.terms = set()

        for page in bundle.load(root).concepts:

            metadata = page.frontmatter or {}

            if metadata.get("type") != "Term":
                continue

            iri = metadata.get("iri")

            if not isinstance(iri, str):
                continue

            self.terms.add(iri)

            labels = [
                metadata.get("pref_label"),
                metadata.get("title"),
                iri.rsplit("/", 1)[-1],
                *metadata.get("alt_labels", []),
            ]

            for label in labels:

                if isinstance(label, str):
                    self.index[
                        normalize_name(label)
                    ].add(iri)

    def resolve(
        self,
        name: str | None,
        *prefixes: str,
    ) -> str | None:
        """Resolve a term within the expected ontology category."""

        if name is None or not name.strip():
            return None

        if name in self.terms:
            candidates = {name}

        else:
            candidates = set(
                self.index.get(normalize_name(name), set())
            )

        candidates = {
            iri for iri in candidates
            if any(iri.startswith(p) for p in prefixes)
        }

        if len(candidates) == 1:
            return next(iter(candidates))

        if not candidates:
            raise ResolutionError(
                f"Unknown ontology term: {name!r}"
            )

        raise ResolutionError(
            f"Ambiguous ontology term: {name!r}"
        )


class EntityResolver:
    """Reuse existing model and dataset rows by title or alias."""

    def __init__(self, ctx):

        self.ctx = ctx
        self.index = defaultdict(set)

        for table in ("models", "datasets"):

            rows = ctx.lake.con.execute(
                f"SELECT iri, title, aliases FROM {table}"
            ).fetchall()

            for iri, title, aliases in rows:

                self.remember(table, title, iri)

                for alias in aliases or []:
                    self.remember(table, alias, iri)

    def remember(
        self,
        table: str,
        name: str,
        iri: str,
    ) -> None:

        self.index[
            (table, normalize_name(name))
        ].add(iri)

    def resolve(self, table: str, name: str) -> str:

        if table not in ("models", "datasets"):
            raise ResolutionError(
                f"Unsupported entity table: {table}"
            )

        key = (table, normalize_name(name))

        matches = self.index.get(key, set())

        if len(matches) > 1:
            raise ResolutionError(
                f"Ambiguous {table} entity: {name!r}"
            )

        if matches:
            return next(iter(matches))

        iri = entity_iri(
            self.ctx.iri_base,
            table,
            name,
        )

        self.remember(table, name, iri)

        return iri