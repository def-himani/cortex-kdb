"""Save a grounded answer via typed actions, never via an agent tool."""

from uuid import uuid4

from kdb_tools.actions.objects import DeleteObjects, UpsertObjects
from kdb_tools.tools import KdbTools

from .citations import CitationError, collect_citations, validate_citations
from .resolution import VLA_CLASSES


def save_analysis(ctx, question: str, answer) -> str:
    """Validate cited sources before writing an Analysis and preserve citation order."""
    if not answer.answerable or not answer.claims:
        raise CitationError("Cannot save an unanswerable response as an Analysis")
    citations = collect_citations(ctx, answer)
    validate_citations(ctx, citations)
    evidence, drew_on, lines = [], [], []
    for number, claim in enumerate(answer.claims, 1):
        if claim.object not in drew_on:
            drew_on.append(claim.object)
        pointers = []
        for c in citations:
            if c.claim_index != number:
                continue
            if c.reference not in evidence:
                evidence.append(c.reference)
            pointers.append(f"evidence[{evidence.index(c.reference)}] ({c.key})")
        lines.append(f"{claim.text} [{'; '.join(pointers)}]")
    if answer.note:
        lines.append(f"Limitation: {answer.note}")
    iri = ctx.iri_base + "analyses/" + uuid4().hex
    class_iri = VLA_CLASSES + "analysis"
    params = {"class_iri": class_iri, "rows": [{"iri": iri, "title": question,
              "question": question, "answer": "\n".join(lines), "drew_on": drew_on, "evidence": evidence}]}
    action = UpsertObjects()
    plan = action.dry_run(ctx, params)
    action.submit(ctx, params, plan)
    # Sources were checked before the insert. If the new row unexpectedly fails,
    # compensate through a logged typed deletion rather than leaving an invalid analysis.
    try:
        checked = KdbTools(ctx.root, ctx=ctx).check_pages([f"analyses/{iri}"])
        if checked.errors:
            raise CitationError("Saved analysis failed validation or grounding: " +
                                "; ".join(p.message for p in checked.errors))
    except Exception:
        delete = DeleteObjects()
        removal = {"class_iri": class_iri, "iris": [iri], "reason": "Analysis post-write validation failed"}
        delete.submit(ctx, removal, delete.dry_run(ctx, removal))
        raise
    return iri
