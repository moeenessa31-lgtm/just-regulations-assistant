"""Extract text from the PDFs in data/raw into data/text.

Text PDFs are read with `pdftotext`. If a PDF has (almost) no text layer, it is
treated as a scan and read with Tesseract OCR (Arabic). OCR output can contain
mistakes, especially in numbers, so always check important numbers by hand.

Usage:  python3 src/extract_text.py [raw_dir] [out_dir]
Needs:  poppler-utils (pdftotext, pdftoppm) and tesseract with the `ara` language.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

MIN_CHARS = 200  # fewer characters than this -> assume a scanned PDF


def pdftotext(pdf: Path) -> str:
    out = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True)
    return out.stdout


def ocr(pdf: Path) -> str:
    text = []
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-r", "200", "-png", str(pdf), f"{tmp}/pg"], check=True)
        for page in sorted(Path(tmp).glob("pg-*.png")):
            res = subprocess.run(["tesseract", str(page), "-", "-l", "ara"],
                                 capture_output=True, text=True)
            text.append(res.stdout)
    return "\n".join(text)


def main() -> None:
    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/text")
    out.mkdir(parents=True, exist_ok=True)
    for pdf in sorted(raw.glob("*.pdf")):
        text = pdftotext(pdf)
        method = "text"
        if len(text.strip()) < MIN_CHARS:
            text, method = ocr(pdf), "ocr"
        (out / f"{pdf.stem}.txt").write_text(text, encoding="utf-8")
        print(f"{pdf.name}: {len(text)} chars via {method}")


if __name__ == "__main__":
    main()
