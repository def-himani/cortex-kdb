"""Ontology term resolution and stable entity identification."""

import unicodedata
from collections import defaultdict
from urllib.parse import quote
from pathlib import PurePosixPath
import posixpath

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


def entity_iri(iri_base: str, table: str, name: str) -> str:
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
            labels = [metadata.get("pref_label"), metadata.get("title"),
                      iri.rsplit("/", 1)[-1], *metadata.get("alt_labels", [])]
            for label in labels:
                if isinstance(label, str):
                    self.index[normalize_name(label)].add(iri)

    def resolve(self, name: str | None, *prefixes: str) -> str | None:
        """Resolve a term within the expected ontology category."""
        if name is None or not name.strip():
            return None
        if name in self.terms:
            candidates = {name}
        else:
            candidates = set(self.index.get(normalize_name(name), set()))
        candidates = {iri for iri in candidates if any(iri.startswith(p) for p in prefixes)}
        if len(candidates) == 1:
            return next(iter(candidates))
        if not candidates:
            raise ResolutionError(f"Unknown ontology term: {name!r}")
        raise ResolutionError(f"Ambiguous ontology term: {name!r}")


class EntityResolver:
    """Reuse existing model and dataset rows by title or alias."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.index = defaultdict(set)
        for table in ("models", "datasets"):
            rows = ctx.lake.con.execute(f"SELECT iri, title, aliases FROM {table}").fetchall()
            for iri, title, aliases in rows:
                self.remember(table, title, iri)
                for alias in aliases or []:
                    self.remember(table, alias, iri)

    def remember(self, table: str, name: str, iri: str) -> None:
        self.index[(table, normalize_name(name))].add(iri)

    def resolve(self, table: str, name: str) -> str:
        if table not in ("models", "datasets"):
            raise ResolutionError(f"Unsupported entity table: {table}")
        key = (table, normalize_name(name))
        matches = self.index.get(key, set())
        if len(matches) > 1:
            raise ResolutionError(f"Ambiguous {table} entity: {name!r}")
        if matches:
            return next(iter(matches))
        iri = entity_iri(self.ctx.iri_base, table, name)
        self.remember(table, name, iri)
        return iri


class OntologySearch:
    """Read-only term index with reverse traversal of broader links."""

    def __init__(self, root):
        self.records = {}
        self.children = defaultdict(set)
        paths = {}
        pending = []
        for page in bundle.load(root).concepts:
            fm = page.frontmatter or {}
            iri = fm.get("iri")
            if fm.get("type") != "Term" or not isinstance(iri, str):
                continue
            path = str(page.rel).lstrip("/")
            labels = [x for x in fm.get("alt_labels", []) if isinstance(x, str)]
            category = PurePosixPath(path).parent.name
            self.records[iri] = {
                "iri": iri,
                "label": fm.get("pref_label") or fm.get("title") or iri.rsplit("/", 1)[-1],
                "aliases": labels,
                "category": category,
            }
            paths[path] = iri
            pending.append((iri, path, fm.get("broader") or []))

        for child, path, parents in pending:
            for parent in parents:
                if not isinstance(parent, str):
                    continue
                if parent in self.records:
                    target = parent
                else:
                    rel = parent.lstrip("/") if parent.startswith("/") else posixpath.join(posixpath.dirname(path), parent)
                    target = paths.get(posixpath.normpath(rel))
                if target is not None:
                    self.children[target].add(child)

    def narrower(self, iri: str, include_self: bool = True) -> list[str]:
        """Return transitive descendants, handling cycles and multiple parents."""
        if iri not in self.records:
            raise ResolutionError(f"Unknown ontology term: {iri!r}")
        seen = set()
        stack = [iri]
        while stack:
            current = stack.pop()
            if current not in seen:
                seen.add(current)
                stack.extend(self.children[current] - seen)
        if not include_self:
            seen.discard(iri)
        return sorted(seen)

    @staticmethod
    def _names(record: dict) -> list[str]:
        return [record["label"], *record["aliases"], record["iri"].rsplit("/", 1)[-1]]

    def match(self, name: str, *prefixes: str) -> str:
        """Require an unambiguous label, alias or IRI in the requested category."""
        wanted = normalize_name(name)
        matches = [iri for iri, row in self.records.items()
                   if (not prefixes or any(iri.startswith(p) for p in prefixes))
                   and (iri == name or wanted in {normalize_name(x) for x in self._names(row)})]
        if len(matches) != 1:
            status = "Unknown" if not matches else "Ambiguous"
            raise ResolutionError(f"{status} ontology term: {name!r}; search_terms for the exact IRI")
        return matches[0]

    def search(self, query: str, category: str | None = None) -> list[dict]:
        """Search names, not raw paper text; prefer exact matches."""
        wanted = normalize_name(query)
        found = []
        for iri, row in self.records.items():
            if category and row["category"] != category:
                continue
            names = [normalize_name(x) for x in self._names(row)]
            exact = iri == query or wanted in names
            if exact or not wanted or any(wanted in x for x in names):
                found.append((not exact, iri, row))
        return [row for _, _, row in sorted(found, key=lambda item: item[:2])]
