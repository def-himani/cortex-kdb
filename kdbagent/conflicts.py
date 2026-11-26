"""Conservative conflict review and explicit contested-row recording."""

from hashlib import sha256
import json
import re

from kdb_tools.actions.objects import UpsertObjects, read_row
from kdb_tools.tools import KdbTools

from .resolution import VLA_CLASSES


EVIDENCE_REF = re.compile(r"(.+)#evidence/([0-9]+)")
CONDITIONS = ("model", "task", "benchmark", "suite", "embodiment", "perturbation", "trials", "trials_reported")
EVIDENCE_TABLES = ("models", "datasets", "trained_on", "results", "contested", "computations", "analyses")


def evidence_list(row):
    value = row.get("evidence") or []
    return json.loads(value) if isinstance(value, str) else value


def result_key(row):
    """Compare the reported fields, without converting metric units."""
    metric = " ".join((row.get("metric") or "").split()).casefold()
    if not metric or any(not row.get(k) for k in CONDITIONS[:5]):
        return None
    return (*[row.get(k) for k in CONDITIONS], metric)


def result_candidates(ctx, touched: set[str]) -> list[dict]:
    """Find unequal values with matching recorded conditions for review.

    Matching recorded fields do not prove equivalent experimental protocols.
    In particular, missing trials or perturbation information is ambiguous.
    No contested rows are automatically created by this function.
    """
    if not touched:
        return []
    cursor = ctx.lake.con.execute("SELECT * FROM results ORDER BY iri")
    columns = [col[0] for col in cursor.description]
    rows = [dict(zip(columns, values)) for values in cursor.fetchall()]
    groups = {}
    for row in rows:
        key = result_key(row)
        if key is not None and evidence_list(row):
            groups.setdefault(key, []).append(row)
    candidates = []
    for group in groups.values():
        for i, left in enumerate(group):
            for right in group[i + 1:]:
                if not ({left["iri"], right["iri"]} & touched):
                    continue
                if left.get("value") is None or right.get("value") is None or left["value"] == right["value"]:
                    continue
                candidates.append({
                    "claim": left["iri"] + "#evidence/0",
                    "disputed_by_candidate": right["iri"] + "#evidence/0",
                    "values": [left["value"], right["value"]],
                    "metric": left["metric"],
                    "conditions": {k: left.get(k) for k in CONDITIONS},
                    "requires_review": True,
                })
    return candidates


def resolve_evidence(ctx, reference):
    match = EVIDENCE_REF.fullmatch(reference)
    if not match:
        raise ValueError("Use <row IRI>#evidence/<zero-based index>")
    iri, index = match[1], int(match[2])
    found = []
    for table in EVIDENCE_TABLES:
        row = read_row(ctx, table, iri)
        if row is not None:
            found.append((table, row))
    if len(found) != 1:
        raise ValueError(f"Evidence reference must identify exactly one row: {reference}")
    table, row = found[0]
    refs = evidence_list(row)
    if index >= len(refs):
        raise ValueError(f"Evidence index out of range: {reference}")
    return table, row, refs[index], f"{table}/{iri}"


def record_contested(ctx, claim: str, disputed_by: list[str]) -> str:
    """Record a reviewed contradiction, preserving all original rows.

    The caller must establish semantic contradiction from the sources.
    This function validates references and rejects unlike numerical results.
    """
    disputed_by = sorted(set(disputed_by))
    if not disputed_by or claim in disputed_by:
        raise ValueError("A contested claim requires distinct disputing evidence")
    table, row, evidence, path = resolve_evidence(ctx, claim)
    refs = [evidence]
    paths = [path]
    for item in disputed_by:
        other_table, other, ref, other_path = resolve_evidence(ctx, item)
        if ref == evidence:
            raise ValueError("Disputing evidence must not be the same source passage")
        if table == "results" or other_table == "results":
            if table != other_table or result_key(row) is None or result_key(row) != result_key(other):
                raise ValueError("Results have different or incomplete recorded conditions or metrics")
            if row.get("value") is None or other.get("value") is None or row["value"] == other["value"]:
                raise ValueError("Results must have distinct reported values")
        if ref not in refs:
            refs.append(ref)
        paths.append(other_path)
    checked = KdbTools(ctx.root, ctx=ctx).check_pages(paths)
    if checked.errors:
        raise ValueError("Referenced rows failed validation; review grounding before recording a conflict")
    key = sha256(claim.encode("utf-8")).hexdigest()
    iri = ctx.iri_base + "contested/" + key
    old = read_row(ctx, "contested", iri) or {}
    # Preserve existing evidence order and previously recorded disputing references.
    all_refs = list(evidence_list(old))
    for ref in refs:
        if ref not in all_refs:
            all_refs.append(ref)
    disputes = list(old.get("disputed_by") or [])
    for item in disputed_by:
        if item not in disputes:
            disputes.append(item)
    params = {"class_iri": VLA_CLASSES + "contested", "rows": [{
        "iri": iri, "claim": claim, "disputed_by": disputes, "evidence": all_refs,
    }]}
    action = UpsertObjects()
    plan = action.dry_run(ctx, params)
    action.submit(ctx, params, plan)
    return f"contested/{iri}"
