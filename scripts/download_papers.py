from pathlib import Path
from urllib.request import urlopen, Request
import shutil
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = PROJECT_ROOT / "kdb" / "papers"

# Exact arXiv versions specified in the assignment.
PAPERS = {
    "CALVIN": "2112.03227v4",
    "RT-1": "2212.06817v2",
    "ACT": "2304.13705v1",
    "BridgeData V2": "2308.12952v3",
    "Diffusion Policy": "2303.04137v5",
    "LIBERO": "2306.03310v2",
    "Open X-Embodiment": "2310.08864v9",
    "RT-2": "2307.15818v1",
    "CogACT": "2411.19650v1",
    "DROID": "2403.12945v2",
    "Octo": "2405.12213v2",
    "OpenVLA": "2406.09246v3",
    "RDT-1B": "2410.07864v2",
    "RoboVLMs": "2412.14058v4",
    "SIMPLER": "2405.05941v1",
    "TinyVLA": "2409.12514v5",
    "TraceVLA": "2412.10345v3",
    "pi0": "2410.24164v4",
    "CoT-VLA": "2503.22020v1",
    "Dita": "2503.19757v2",
    "FAST": "2501.09747v1",
    "GR00T N1": "2503.14734v2",
    "NORA": "2504.19854v1",
    "OpenVLA-OFT": "2502.19645v2",
    "RL4VLA": "2505.19789v4",
    "SimpleVLA-RL": "2509.09674v1",
    "SmolVLA": "2506.01844v1",
    "SpatialVLA": "2501.15830v5",
    "UniVLA": "2505.06111v3",
    "pi0.5": "2504.16054v1",

    # Reserved for Submission 2 ingestion.
    "LIBERO-Plus": "2510.13626v3",
}


def is_pdf(path: Path) -> bool:
    if not path.is_file():
        return False

    with path.open("rb") as file:
        return file.read(5) == b"%PDF-"


def download_paper(name: str, versioned_id: str) -> str:
    arxiv_id = versioned_id.split("v")[0]

    destination = PAPERS_DIR / f"{arxiv_id}.pdf"
    temporary = PAPERS_DIR / f"{arxiv_id}.pdf.part"

    if is_pdf(destination):
        print(f"[SKIP] {name}: already exists")
        return "skipped"

    url = f"https://arxiv.org/pdf/{versioned_id}"

    print(f"[DOWNLOAD] {name}: {versioned_id}")

    request = Request(
        url,
        headers={"User-Agent": "Cortex-KDB-Academic-Project/1.0"},
    )

    try:
        with urlopen(request, timeout=120) as response:
            with temporary.open("wb") as file:
                shutil.copyfileobj(response, file)

        if not is_pdf(temporary):
            raise ValueError("Downloaded file is not a PDF")

        temporary.replace(destination)

        size_mb = destination.stat().st_size / (1024 * 1024)
        print(f"[OK] {destination.name} ({size_mb:.2f} MB)")

        return "downloaded"

    finally:
        temporary.unlink(missing_ok=True)


def main():
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed = []

    for name, versioned_id in PAPERS.items():
        try:
            status = download_paper(name, versioned_id)

            if status == "downloaded":
                downloaded += 1
                time.sleep(3)
            else:
                skipped += 1

        except Exception as error:
            print(f"[FAILED] {name}: {error}")
            failed.append(name)
            time.sleep(3)

    print("\n--- Download Summary ---")
    print(f"Required:   {len(PAPERS)}")
    print(f"Downloaded: {downloaded}")
    print(f"Skipped:    {skipped}")
    print(f"Failed:     {len(failed)}")
    print(f"Available:  {downloaded + skipped}/{len(PAPERS)}")

    if failed:
        print("\nFailed papers:")
        for name in failed:
            print(f"- {name}")

        raise SystemExit(1)


if __name__ == "__main__":
    main()