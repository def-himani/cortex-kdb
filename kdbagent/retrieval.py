"""Bounded read-only retrieval over the VLA lake, without model-supplied SQL."""

from collections import defaultdict
import json

from kdb_tools.tools import KdbTools, ToolInputError

from .resolution import (
    DATASETS, ROBOTICS, VLA_DATASETS, OntologySearch,
    ResolutionError, normalize_name,
)

MAX_ROWS = 12
MAX_EVIDENCE = 3
MAX_TEXT = 800

CONDITIONS = ("task", "benchmark", "suite", "embodiment", "perturbation", "trials", "trials_reported", "metric")
TERM_PREFIXES = {
    "task": (ROBOTICS + "tasks/",),
    "benchmark": (ROBOTICS + "benchmarks/",),
    "suite": (ROBOTICS + "benchmarks/",),
    "embodiment": (ROBOTICS + "embodiments/",),
    "perturbation": (ROBOTICS + "perturbations/",),
    "role": (DATASETS + "roles/", VLA_DATASETS + "roles/"),
    "collection": (DATASETS + "collection/", VLA_DATASETS + "collection/"),
}


def bounded_limit(limit: int) -> int:
    if not 1 <= limit <= MAX_ROWS:
        raise ToolInputError(f"limit must be between 1 and {MAX_ROWS}")
    return limit


def evidence_list(row: dict) -> list:
    entries = row.get("evidence") or []
    if isinstance(entries, str):
        entries = json.loads(entries)
    return entries if isinstance(entries, list) else []


def compact_text(text):
    if isinstance(text, str) and len(text) > MAX_TEXT:
        return text[:MAX_TEXT] + " [truncated; retrieve exact evidence]"
    return text


def row_summary(row: dict) -> dict:
    """Omit long evidence arrays, retaining row identity and the evidence count."""
    out = {k: compact_text(v) if k in ("description", "answer", "question", "code") else v
           for k, v in row.items() if k != "evidence" and v is not None}
    out["evidence_count"] = len(evidence_list(row)) if "evidence" in row else row.get("evidence_count", 0)
    return out


def page_result(rows: list, limit: int) -> dict:
    more = len(rows) > limit
    visible = rows[:limit]
    return {"rows": visible, "truncated": more,
            "next_after_iri": visible[-1].get("iri") if more and visible else None}


def comparison_groups(rows: list[dict]) -> list[dict]:
    """Group by exact stored conditions, never calculate a ranking or a new value."""
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in CONDITIONS)].append(row["iri"])
    return [{"conditions": dict(zip(CONDITIONS, key)), "rows": iris,
             "caution": "Matching recorded fields only. Unknown conditions or differing protocols may prevent comparison."
             if key[4] is None or key[5] is None or not key[6] else
             "Matching recorded fields only; confirm identical evaluation protocols from the evidence."}
            for key, iris in groups.items()]


class Retrieval(KdbTools):
    """Use the existing lake context but expose only the explicitly registered read tools."""

    def __init__(self, ctx):
        super().__init__(ctx.root, ctx=ctx)
        self.terms = OntologySearch(ctx.root)

    def _select(self, sql: str, parameters: list) -> list[dict]:
        # SQL is authored in this module. All model-provided values are bound parameters.
        cur = self.ctx.lake.con.execute(sql, parameters)
        names = [d[0] for d in cur.description]
        return [dict(zip(names, row)) for row in cur.fetchall()]

    def _term_ids(self, field: str, value: str, include_narrower: bool) -> list[str]:
        try:
            iri = self.terms.match(value, *TERM_PREFIXES[field])
            ids = self.terms.narrower(iri) if include_narrower else [iri]
            return [x for x in ids if any(x.startswith(p) for p in TERM_PREFIXES[field])]
        except ResolutionError as exc:
            raise ToolInputError(str(exc)) from exc

    def _entity_ids(self, table: str, name: str) -> list[str]:
        # Do not call the ingestion resolver: it may mint a new IRI on a miss.
        if table not in ("models", "datasets"):
            raise ToolInputError("entity table must be models or datasets")
        rows = self._select(f"SELECT iri, title, aliases FROM {table}", [])
        wanted = normalize_name(name)
        matches = [r["iri"] for r in rows if name == r["iri"] or wanted in
                   {normalize_name(x) for x in [r["title"], *(r["aliases"] or [])] if isinstance(x, str)}]
        if len(matches) > 1:
            raise ToolInputError(f"Ambiguous {table} name {name!r}; choose an exact IRI using search_entities")
        return matches

    def search_entities(self, name: str, kind: str, limit: int = 8) -> dict:
        bounded_limit(limit)
        if kind not in ("models", "datasets", "papers"):
            raise ToolInputError("kind must be models, datasets or papers")
        columns = "iri, title, arxiv_id, version" if kind == "papers" else "iri, title, aliases"
        rows = self._select(f"SELECT {columns} FROM {kind} ORDER BY iri", [])
        wanted = normalize_name(name)
        if not wanted:
            raise ToolInputError("supply a non-empty entity name or arXiv ID")
        found = []
        for row in rows:
            labels = [row["title"], row.get("arxiv_id"), *(row.get("aliases") or [])]
            names = [normalize_name(x) for x in labels if isinstance(x, str)]
            exact = row["iri"] == name or wanted in names
            if exact or any(wanted in x for x in names):
                found.append((not exact, row["iri"], row))
        visible = [r for _, _, r in sorted(found, key=lambda x: x[:2])]
        return {"rows": visible[:limit], "truncated": len(visible) > limit}

    @staticmethod
    def _in(clauses, args, field, ids):
        # field is from the module's fixed whitelist, never directly from the model.
        if not ids:
            clauses.append("FALSE")
        else:
            clauses.append(f"{field} IN ({', '.join('?' for _ in ids)})")
            args.extend(ids)

    def search_results(self, model=None, task=None, benchmark=None, suite=None,
                       embodiment=None, perturbation=None, metric=None,
                       include_narrower=True, limit=8, after_iri=None) -> dict:
        bounded_limit(limit)
        clauses, args = [], []
        if model is not None:
            self._in(clauses, args, "model", self._entity_ids("models", model))
        for field, value in (("task", task), ("benchmark", benchmark), ("suite", suite),
                             ("embodiment", embodiment), ("perturbation", perturbation)):
            if value is not None:
                self._in(clauses, args, field, self._term_ids(field, value, include_narrower))
        if metric is not None:
            clauses.append("contains(lower(metric), lower(?))")
            args.append(metric)
        if after_iri is not None:
            clauses.append("iri > ?")
            args.append(after_iri)
        where = " AND ".join(clauses) or "TRUE"
        rows = self._select("SELECT iri, title, model, task, benchmark, suite, embodiment, perturbation, "
                            "trials, trials_reported, value, metric, json_array_length(evidence) AS evidence_count "
                            f"FROM results WHERE {where} ORDER BY iri LIMIT ?", [*args, limit + 1])
        out = page_result(rows, limit)
        out["groups"] = comparison_groups(out["rows"])
        out["note"] = "Groups cover this returned page only. Retrieve evidence before comparing or citing results."
        return out

    def search_training(self, model=None, dataset=None, role=None,
                        include_narrower=True, limit=8, after_iri=None) -> dict:
        bounded_limit(limit)
        clauses, args = [], []
        if model is not None:
            self._in(clauses, args, "model", self._entity_ids("models", model))
        if dataset is not None:
            self._in(clauses, args, "dataset", self._entity_ids("datasets", dataset))
        if role is not None:
            self._in(clauses, args, "role", self._term_ids("role", role, include_narrower))
        if after_iri is not None:
            clauses.append("iri > ?")
            args.append(after_iri)
        where = " AND ".join(clauses) or "TRUE"
        return page_result(self._select("SELECT iri, model, dataset, role, json_array_length(evidence) AS evidence_count "
                                      f"FROM trained_on WHERE {where} ORDER BY iri LIMIT ?", [*args, limit+1]), limit)

    def search_datasets(self, collection=None, task=None, embodiment=None,
                        include_narrower=True, limit=8, after_iri=None) -> dict:
        bounded_limit(limit)
        clauses, args = [], []
        for field, column, value in (("collection", "collection", collection), ("task", "tasks", task),
                                     ("embodiment", "embodiments", embodiment)):
            if value is not None:
                ids = self._term_ids(field, value, include_narrower)
                clauses.append("(" + " OR ".join(f"list_contains({column}, ?)" for _ in ids) + ")" if ids else "FALSE")
                args.extend(ids)
        if after_iri is not None:
            clauses.append("iri > ?")
            args.append(after_iri)
        where = " AND ".join(clauses) or "TRUE"
        return page_result(self._select("SELECT iri, title, collection, tasks, embodiments, modalities, "
                                      "json_array_length(evidence) AS evidence_count "
                                      f"FROM datasets WHERE {where} ORDER BY iri LIMIT ?", [*args, limit+1]), limit)

    def search_contested(self, row_iri: str, limit: int = 8) -> dict:
        bounded_limit(limit)
        prefix = row_iri + "#evidence/"
        rows = self._select("SELECT iri, claim, disputed_by, json_array_length(evidence) AS evidence_count "
                            "FROM contested ORDER BY iri", [])
        matches = [r for r in rows if any(isinstance(x, str) and x.startswith(prefix)
                   for x in [r["claim"], *(r["disputed_by"] or [])])]
        return {"rows": matches[:limit], "truncated": len(matches) > limit}

    def read_evidence(self, iri: str, indices: list[int]) -> dict:
        if not 1 <= len(indices) <= MAX_EVIDENCE or len(set(indices)) != len(indices):
            raise ToolInputError(f"supply 1 to {MAX_EVIDENCE} distinct evidence indices")
        row = self.get_object(iri)
        if row is None:
            raise ToolInputError(f"No row {iri!r}")
        entries = evidence_list(row)
        if any(not 0 <= i < len(entries) for i in indices):
            raise ToolInputError(f"{iri} has {len(entries)} evidence entries; indices start at 0")
        found = []
        for i in indices:
            item = dict(entries[i])
            for field in ("quote", "value"):
                if isinstance(item.get(field), str) and len(item[field]) > MAX_TEXT:
                    item[field] = compact_text(item[field])
                    item["text_truncated"] = True
            found.append({"index": i, **item})
        return {"iri": iri, "evidence_count": len(entries), "entries": found}
