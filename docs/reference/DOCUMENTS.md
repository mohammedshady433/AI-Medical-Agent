# Document processing and retrieval

## Supported routes

| Formats | Extraction route | Practical limits |
|---|---|---|
| TXT, Markdown, CSV, JSON, XML, HTML | Bounded Python readers | UTF-8/UTF-16 text; no HTML script execution or link fetching |
| DOCX, PPTX, XLSX and related OOXML variants | ZIP/XML readers | Macros ignored; spreadsheets use saved values, not recalculation |
| DOC, PPT, XLS | Bundled restricted Apache Tika/Java | Corrupt/encrypted formats can fail; formatting differs from the original |
| RTF | striprtf | Plain text output; layout is not preserved |
| ODT, ODS, ODP, EPUB, supported message formats | Tika | Images and embedded attachments need separate handling |
| PDF | pypdf; local raster OCR for sparse pages | Encrypted PDFs rejected; page/character limits apply |
| PNG, JPEG, TIFF, BMP, GIF, WebP, scanned PDF | Tesseract Arabic/English OCR when configured | OCR is not general visual reasoning; handwriting, charts and low-quality scans need review |
| PDF/DOCX/PPTX/XLSX via optional Docling | Isolated parser with preinstalled assets | Disabled until its environment and offline artifacts are configured |

The baseline supports 100 MB per file, 2,000 PDF pages and 12 million extracted characters. These are safety limits, not a claim that every file at the limit will parse quickly. Split larger archives/books into meaningful parts. The core queues documents and reports extraction/indexing progress, errors and warnings through `GET /documents`. It never requires an entire document to fit in the model's context.

## Extraction lifecycle

The upload is streamed to a temporary file while calculating its digest. An identical digest reuses the existing library entry. A new document is stored under an opaque ID outside the installation directory. Parsing runs off the API event loop. Successful text is split into located chunks; rows, slides and pages keep their source labels. Large passages overlap to preserve sentence context. Embeddings are processed in batches; completed vectors persist even if a later batch fails.

`ready` means semantic indexing completed. `keyword_only` means text is available but local embeddings failed or were not installed; Retry fills missing vectors. `error` indicates extraction failed; read the reason and fix the input/setup before retrying. An interrupted job is visible after restart. Document deletion removes its text, vector index and stored original, not other conversations or files.

## Retrieval behavior

RAG means answering with retrieved evidence. A multilingual embedding maps English and Arabic passages into comparable numerical vectors. A query vector finds related passages even when words differ. FTS5 supplies exact-word matches. Reciprocal-rank fusion combines their rankings, then the context budget limits the evidence passed to the model. Stored vectors are tagged with the embedding model so different embedding spaces are not mixed.

The model can search several times for a multi-part task. It can use an alternate-language query if necessary. Source references map to filenames and page/slide/row labels. The core removes nonexistent reference IDs but does not claim that every model sentence is proven merely because its citation ID exists. Test actual facts and translations against known fixtures.

An overview tool returns the library inventory and representative beginning/middle/end excerpts within a bounded budget. The result labels its coverage. For a full document review, ask a focused series of questions rather than treating a few excerpts as a complete audit. General questions do not require document evidence and are answered normally.

## OCR and Docling setup

Install [Tesseract for Windows](https://github.com/UB-Mannheim/tesseract/wiki) with `ara` and `eng` trained data. Set its path with `PATCH /settings {"documents": {"ocr": {"executable": "..."}}}` if it is not detected. Test one clean printed image and a scanned PDF; inspect numbers in the original. A scan with no configured OCR produces a clear error or page warning, not fabricated extracted content.

For advanced structured parsing, create `.docling-env` with the pinned version in `runtime-lock.json`, then explicitly prefetch its models using `docling-tools models download`. Set `documents.docling.python`, `artifacts_path` and `enabled` in saved settings. The worker sets offline environment flags and disables remote services. Do not enable Docling before assets are present. See [Docling's offline instructions](https://github.com/docling-project/docling/blob/main/docs/usage/advanced_options.md).

Adding a format means implementing a pure local parser in `documents/extract.py` or a new module, registering its suffix, bounding its input/output, and adding a fixture test. Do not launch Office, execute macros, follow arbitrary embedded URLs or turn document instructions into agent permissions.

The Windows bundle includes Tesseract with Arabic and English data. Leaving the OCR executable setting empty uses that bundled runtime before checking PATH. A configured custom executable still takes precedence. English text-image extraction has been exercised with the real bundled executable; Arabic OCR accuracy needs representative user scans.
