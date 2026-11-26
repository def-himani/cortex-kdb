"""Resolve and render immutable citations from the lake, outside the QA agent."""

from dataclasses import asdict, dataclass
from pathlib import Path
import io
import re

import pdfplumber
from kdb_tools import media
from kdb_tools.tools import KdbTools

from .retrieval import evidence_list


class CitationError(ValueError):
    """An answer cites missing, malformed or ungrounded evidence."""


@dataclass(frozen=True)
class Citation:
    claim_index: int
    row_iri: str
    row_path: str
    evidence_index: int
    paper_id: str
    paper_title: str
    version: str | None
    origin: str
    reference: dict

    def to_dict(self):
        return asdict(self)

    @property
    def ref(self):
        return media.MediaRef.model_validate(self.reference)

    @property
    def key(self):
        return f"{self.row_iri}#evidence/{self.evidence_index}"


def collect_citations(ctx, answer) -> list[Citation]:
    """Resolve exact evidence from stored rows, not truncated tool responses."""
    if not answer.answerable:
        if answer.claims:
            raise CitationError("An unanswerable response cannot contain claims")
        return []
    if not answer.claims:
        raise CitationError("An answerable response needs claims")

    tools = KdbTools(ctx.root, ctx=ctx)
    tables = {r["iri"]: r["table"] for r in tools.list_classes() if r.get("table")}
    papers = {sha: (pid, title, version) for sha, pid, title, version in ctx.lake.con.execute(
        "SELECT media, arxiv_id, title, version FROM papers ORDER BY iri").fetchall()}
    rows, registered = {}, {}
    citations = []
    for number, claim in enumerate(answer.claims, 1):
        if not claim.text.strip() or not claim.evidence or len(set(claim.evidence)) != len(claim.evidence):
            raise CitationError("Each claim needs text and distinct evidence indices")
        if claim.object not in rows:
            rows[claim.object] = tools.get_object(claim.object)
        row = rows[claim.object]
        if row is None or row.get("type") not in tables:
            raise CitationError(f"Unknown bound row: {claim.object}")
        entries = evidence_list(row)
        for index in claim.evidence:
            if not 0 <= index < len(entries):
                raise CitationError(f"{claim.object} has no evidence[{index}]")
            ref = media.MediaRef.model_validate(entries[index])
            if not (ref.quote or ref.value):
                raise CitationError(f"No quoted evidence at {claim.object}#evidence/{index}")
            if ref.sha256 not in registered:
                registered[ref.sha256] = media.lookup(ctx.lake, ref.sha256)
            source = registered[ref.sha256]
            if source is None:
                raise CitationError(f"Unregistered source: {ref.sha256}")
            if source.get("annotates"):
                raise CitationError("Citations must refer to original source bytes, not annotated copies")
            if why := media.unchecked(ref, source):
                raise CitationError(why)
            name = source.get("name") or ref.sha256
            origin = source.get("origin") or ""
            pid, title, version = papers.get(ref.sha256, (Path(name).stem, name, None))
            # For registered sources that predate the papers table, retain explicit origin version.
            if not version:
                match = re.search(rf"{re.escape(pid)}(v\d+)(?:\.pdf)?(?:$|[?#])", origin)
                version = match.group(1) if match else None
            citations.append(Citation(number, claim.object, f"{tables[row['type']]}/{claim.object}", index,
                                      pid, title, version, origin, ref.model_dump(exclude_none=True)))
    return citations


def validate_citations(ctx, citations: list[Citation]) -> None:
    """Run the official structural and grounding checks on each cited row."""
    paths = sorted({c.row_path for c in citations})
    if not paths:
        return
    checked = KdbTools(ctx.root, ctx=ctx).check_pages(paths)
    if checked.errors:
        details = "; ".join(f"{p.rule} {p.path}: {p.message}" for p in checked.errors)
        raise CitationError(f"Cited rows failed validation or grounding: {details}")


def render_answer(answer, citations: list[Citation]) -> str:
    if not answer.answerable:
        return "Cannot answer from the knowledge database.\n" + answer.note
    lines = []
    for index, claim in enumerate(answer.claims, 1):
        lines.append(f"{index}. {claim.text}")
        for citation in citations:
            if citation.claim_index != index:
                continue
            ref = citation.ref
            paper = citation.paper_id + (citation.version or "")
            location = f"PDF page {ref.page}" if ref.page is not None else f"sheet {ref.sheet or '(default)'}, cell {ref.cell}"
            selector = "Quote" if ref.quote is not None else "Value"
            text = ref.quote if ref.quote is not None else ref.value
            lines.append(f"   [{paper}, {location}] {selector}: {text}")
            lines.append(f"   Source: {citation.key}")
            lines.append(f"   SHA-256: {ref.sha256}{ref.fragment()}")
        lines.append("")
    if answer.note:
        lines.append(f"Limitation: {answer.note}")
    return "\n".join(lines).rstrip()


def render_region(pdf_bytes: bytes, ref, destination: Path) -> Path:
    """Render a crop of the verified original with the cited bbox highlighted."""
    if media.sha256_of(pdf_bytes) != ref.sha256:
        raise CitationError("Source bytes do not match the evidence digest")
    if ref.page is None or ref.bbox is None:
        raise CitationError("PDF evidence needs a page and bounding box")
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as document:
        if not 1 <= ref.page <= len(document.pages):
            raise CitationError("Evidence page is outside the source PDF")
        page = document.pages[ref.page - 1]
        x0, y0, x1, y1 = ref.bbox
        box = (x0, page.height - y1, x1, page.height - y0)
        left, top, right, bottom = page.bbox
        if not (left <= box[0] < box[2] <= right and top <= box[1] < box[3] <= bottom):
            raise CitationError("Evidence bounding box is outside the PDF page")
        crop = (max(left, box[0] - 18), max(top, box[1] - 18),
                min(right, box[2] + 18), min(bottom, box[3] + 18))
        image = page.crop(crop).to_image(resolution=150)
        image.draw_rect(box, fill=(255, 230, 102, 75), stroke=(180, 130, 0, 255), stroke_width=1)
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(str(destination), format="PNG", quantize=False)
    return destination


def show_citations(ctx, citations: list[Citation], directory: Path) -> list[Path]:
    """Output one highlighted PNG per distinct cited region; no lake writes."""
    validate_citations(ctx, citations)
    paths, seen = [], set()
    for citation in citations:
        ref = citation.ref
        if ref.page is None or ref.bbox is None:
            raise CitationError("--show currently supports PDF citations only")
        key = (ref.sha256, ref.page, tuple(ref.bbox))
        if key in seen:
            continue
        seen.add(key)
        row = media.lookup(ctx.lake, ref.sha256)
        if row is None:
            raise CitationError("Source media is missing")
        data = media.fetch(ctx.store, row)
        if isinstance(data, str):
            raise CitationError(data)
        path = directory / f"{len(paths) + 1:03d}-{ref.sha256[:12]}-page-{ref.page}.png"
        paths.append(render_region(data, ref, path))
    return paths
