"""Optional isolated Docling runtime. All assets must be installed before offline conversion."""

import os
import sys
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

if __name__ == "__main__":
    options = PdfPipelineOptions(artifacts_path=Path(os.environ["DOCLING_ARTIFACTS_PATH"]))
    options.enable_remote_services = False
    result = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    ).convert(sys.argv[1])
    Path(sys.argv[2]).write_text(result.document.export_to_markdown(), encoding="utf8")
