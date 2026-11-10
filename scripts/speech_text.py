"""Local text segmentation for speech; no model calls or transcript rewriting."""

import re

LATIN = re.compile(
    r"[A-Za-z][A-Za-z0-9]*(?:[-./_+][A-Za-z0-9]+)*(?:\s+[A-Za-z][A-Za-z0-9]*(?:[-./_+][A-Za-z0-9]+)*)*"
)
NUMBER = re.compile(r"\d{1,3}(?:[,٬]\d{3})+(?:[.٫]\d+)?|\d+(?:[.٫]\d+)?")


def verbalize_numbers(text, convert):
    text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    return NUMBER.sub(
        lambda match: convert(match.group().replace(",", "").replace("٬", "").replace("٫", ".")), text
    )
