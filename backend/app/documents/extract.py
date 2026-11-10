"""Modular bounded readers; no document code, macros, or external links are executed."""

import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path
from PIL import Image
from pypdf import PdfReader
from defusedxml import ElementTree
from striprtf.striprtf import rtf_to_text
from . import office_readers as office
from app.core.settings import ROOT

SUPPORTED = office.SUPPORTED
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}


def ocr(path, config):
    bundled = ROOT / "document-runtime/tesseract/tesseract.exe"
    command = config["ocr"]["executable"] or (
        str(bundled) if os.name == "nt" and bundled.is_file() else next((p for p in [shutil.which("tesseract"), "/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"] if p and Path(p).is_file()), None)
    )
    if not command or not Path(command).is_file():
        raise ValueError(
            "OCR is not configured. Install Tesseract with Arabic and English language data, then set its path in Settings."
        )
    process = subprocess.run(
        [command, str(path), "stdout", "-l", config["ocr"]["language"], "--psm", "3"],
        capture_output=True,
        timeout=120,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if process.returncode:
        raise ValueError("OCR failed. Check the installed Arabic/English language data and image quality.")
    return process.stdout.decode("utf8").strip()


def extract(path, config):
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError("Unsupported file extension.")
    if path.stat().st_size > config["max_file_mb"] * 1024**2:
        raise ValueError("File exceeds the configured upload limit. Split it into smaller files.")
    if config["docling"]["enabled"] and ext in (".pdf", ".docx", ".pptx", ".xlsx"):
        return docling_extract(path, config)
    if ext == ".pdf":
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ValueError("Use a decrypted copy of this PDF.")
        if len(reader.pages) > config["max_pages"]:
            raise ValueError("PDF exceeds the configured page limit.")
        parts, warnings, total = [], [], 0
        raster = None
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if len(text.strip()) < 30:
                try:
                    import pypdfium2

                    if raster is None:
                        raster = pypdfium2.PdfDocument(path)
                    with tempfile.TemporaryDirectory(prefix="ems-ocr-") as tmp:
                        image = Path(tmp) / "page.png"
                        raster[i].render(scale=2).to_pil().save(image)
                        text = ocr(image, config)
                except ValueError as exc:
                    warnings.append(f"Page {i + 1}: {exc}")
            if text.strip():
                parts.append((f"Page {i + 1}", text, True))
            total += len(text)
            if total > config["max_extracted_chars"]:
                raise ValueError("Extracted PDF text exceeds the configured safe limit.")
        if raster is not None:
            raster.close()
        return parts, list(dict.fromkeys(warnings))
    if ext in IMAGE_TYPES:
        with Image.open(path) as image:
            if image.width * image.height > 40_000_000:
                raise ValueError("Image exceeds 40 megapixels. Resize it first.")
            with tempfile.TemporaryDirectory(prefix="ems-ocr-") as tmp:
                target = Path(tmp) / "image.png"
                image.convert("RGB").save(target)
                return [("Image OCR", ocr(target, config), True)], [
                    "OCR can misread numbers or handwriting; verify against the original."
                ]
    if ext == ".xml":
        root = ElementTree.fromstring(path.read_bytes())
        return office.paragraphs("\n".join(root.itertext())), []
    if ext == ".rtf":
        return office.paragraphs(rtf_to_text(path.read_text(encoding="utf8", errors="replace"))), []
    return office.extract_extra(path)


def docling_extract(path, config):
    settings = config["docling"]
    if not settings["python"] or not settings["artifacts_path"]:
        raise ValueError("Configure the optional Docling Python and offline artifacts path first.")
    from app.core.settings import ROOT

    env = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        DOCLING_ARTIFACTS_PATH=settings["artifacts_path"],
    )
    with tempfile.TemporaryDirectory(prefix="ems-docling-") as tmp:
        output = Path(tmp) / "document.md"
        result = subprocess.run(
            [settings["python"], str(ROOT / "scripts/docling_worker.py"), str(path), str(output)],
            env=env,
            capture_output=True,
            timeout=600,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode:
            raise ValueError("Docling failed. Validate its offline models using the diagnostics script.")
        if output.stat().st_size > config["max_extracted_chars"] * 4:
            raise ValueError("Docling output exceeds the extraction limit.")
        return office.paragraphs(output.read_text(encoding="utf8"), "Section"), []


def chunks(parts, config):
    size, overlap = config["chunk_chars"], config["chunk_overlap"]
    for location, text, _ in parts:
        text = text.strip()
        offset = 0
        while offset < len(text):
            end = min(offset + size, len(text))
            if end < len(text):
                boundary = max(
                    text.rfind("\n", offset + size // 2, end),
                    text.rfind(". ", offset + size // 2, end),
                    text.rfind(" ", offset + size // 2, end),
                )
                if boundary > offset:
                    end = boundary + 1
            value = text[offset:end].strip()
            if value:
                yield (
                    location
                    + (f" / part {offset // max(1, size - overlap) + 1}" if len(text) > size else ""),
                    value,
                )
            if end == len(text):
                break
            offset = max(offset + 1, end - overlap)
