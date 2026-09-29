from __future__ import annotations

from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "Ebook-Agentic-AI.pdf"
SOURCE_URL = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading source PDF to {OUTPUT} ...")
    with requests.get(SOURCE_URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        with OUTPUT.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
    print("Download complete.")


if __name__ == "__main__":
    main()
