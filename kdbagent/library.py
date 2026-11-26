"""Versioned paper library for the Physical AI kDB."""

from pathlib import Path


PAPER_VERSIONS = {
    "2112.03227": "v4",
    "2212.06817": "v2",
    "2303.04137": "v5",
    "2304.13705": "v1",
    "2306.03310": "v2",
    "2307.15818": "v1",
    "2308.12952": "v3",
    "2310.08864": "v9",
    "2403.12945": "v2",
    "2405.05941": "v1",
    "2405.12213": "v2",
    "2406.09246": "v3",
    "2409.12514": "v5",
    "2410.07864": "v2",
    "2410.24164": "v4",
    "2411.19650": "v1",
    "2412.10345": "v3",
    "2412.14058": "v4",
    "2501.09747": "v1",
    "2501.15830": "v5",
    "2502.19645": "v2",
    "2503.14734": "v2",
    "2503.19757": "v2",
    "2503.22020": "v1",
    "2504.16054": "v1",
    "2504.19854": "v1",
    "2505.06111": "v3",
    "2505.19789": "v4",
    "2506.01844": "v1",
    "2509.09674": "v1",
    "2510.13626": "v3",
}


HELD_OUT_PAPER = "2510.13626"


def paper_origin(arxiv_id: str) -> str:
    """Return the exact versioned arXiv source URL."""

    version = PAPER_VERSIONS[arxiv_id]

    return f"https://arxiv.org/abs/{arxiv_id}{version}"


def submission_one_papers(root: Path) -> list[Path]:
    """Return the 30 initial papers, excluding the held-out paper."""

    return [
        root / "papers" / f"{arxiv_id}.pdf"
        for arxiv_id in sorted(PAPER_VERSIONS)
        if arxiv_id != HELD_OUT_PAPER
    ]


def submission_two_paper(root: Path) -> Path:
    """Return the held-out incremental ingestion paper."""

    return root / "papers" / f"{HELD_OUT_PAPER}.pdf"