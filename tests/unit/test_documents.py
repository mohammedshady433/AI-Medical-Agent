from pathlib import Path
import zipfile
import pytest
from app.documents.extract import extract, chunks


@pytest.mark.parametrize(
    "suffix,text,expected",
    [
        (".txt", "Training budget: 8000 EGP", "8000"),
        (".md", "# Budget\n\n8000 EGP", "8000"),
        (".csv", "Department,Budget\nTraining,8000", "Budget: 8000"),
        (".json", '{"budget":8000}', "8000"),
        (".xml", "<budget><amount>8000</amount></budget>", "8000"),
        (".html", "<script>UNSAFE</script><p>Budget 8000</p>", "Budget 8000"),
        (".rtf", r"{\rtf1\ansi Training budget 8000}", "8000"),
    ],
)
def test_text_formats(tmp_path, settings, suffix, text, expected):
    path = tmp_path / ("document" + suffix)
    path.write_text(text, encoding="utf8")
    parts, _ = extract(path, settings.snapshot()["documents"])
    output = " ".join(p[1] for p in parts)
    assert expected in output and "UNSAFE" not in output


def package(path, parts):
    with zipfile.ZipFile(path, "w") as z:
        for name, text in parts.items():
            z.writestr(name, text)


def test_docx(tmp_path, settings):
    path = tmp_path / "budget.docx"
    package(
        path,
        {
            "word/document.xml": '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Training budget 8000</w:t></w:r></w:p></w:body></w:document>'
        },
    )
    parts, _ = extract(path, settings.snapshot()["documents"])
    assert "8000" in parts[0][1]


def test_pptx_slide_order(tmp_path, settings):
    path = tmp_path / "slides.pptx"
    package(
        path,
        {
            "ppt/presentation.xml": '<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><p:sldIdLst><p:sldId r:id="r2"/><p:sldId r:id="r1"/></p:sldIdLst></p:presentation>',
            "ppt/_rels/presentation.xml.rels": '<Relationships><Relationship Id="r1" Target="slides/slide1.xml"/><Relationship Id="r2" Target="slides/slide2.xml"/></Relationships>',
            **{
                f"ppt/slides/slide{i}.xml": f'<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:p><a:r><a:t>Slide {i} budget</a:t></a:r></a:p></p:sld>'
                for i in (1, 2)
            },
        },
    )
    parts, _ = extract(path, settings.snapshot()["documents"])
    assert "Slide 2" in parts[0][1] and "Slide 1" in parts[1][1]


def test_xlsx_saved_values(tmp_path, settings):
    path = tmp_path / "sheet.xlsx"
    package(
        path,
        {
            "xl/workbook.xml": '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Budget" r:id="r1"/></sheets></workbook>',
            "xl/_rels/workbook.xml.rels": '<Relationships><Relationship Id="r1" Target="worksheets/sheet1.xml"/></Relationships>',
            "xl/worksheets/sheet1.xml": '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1"><v>8000</v></c><c r="B1"><f>A1*2</f><v>16000</v></c></row></sheetData></worksheet>',
        },
    )
    parts, warnings = extract(path, settings.snapshot()["documents"])
    assert "8000" in parts[0][1] and "cached value" in parts[0][1] and warnings


@pytest.mark.parametrize(
    "filename,expected",
    [("testWORD.doc", "Sample"), ("testPPT.ppt", "PowerPoint"), ("testEXCEL.xls", "Sample")],
)
def test_legacy_real_office_fixtures(settings, filename, expected):
    import os, subprocess
    if os.name != 'nt':
        found = subprocess.run(['/usr/libexec/java_home', '-v', '17'], capture_output=True)
        if found.returncode and not os.environ.get('JAVA_HOME'):
            pytest.skip('Optional native JDK 17 is not installed on this Mac')
    path = Path(__file__).resolve().parents[1] / "fixtures" / filename
    parts, _ = extract(path, settings.snapshot()["documents"])
    assert expected.lower() in " ".join(p[1] for p in parts).lower()


def test_xml_entities_rejected(tmp_path, settings):
    path = tmp_path / "bad.xml"
    path.write_text('<!DOCTYPE x [<!ENTITY ext SYSTEM "file:///private">]><x>&ext;</x>')
    with pytest.raises(Exception):
        extract(path, settings.snapshot()["documents"])


def test_chunking_preserves_tail(settings):
    cfg = settings.snapshot()["documents"]
    text = "Start " + ("detail " * 600) + " END-OF-DOCUMENT"
    result = list(chunks([("Page 1", text, True)], cfg))
    assert all(len(t) <= cfg["chunk_chars"] for _, t in result)
    assert result[-1][1].endswith("END-OF-DOCUMENT")


def test_corrupt_file_fails(tmp_path, settings):
    path = tmp_path / "broken.docx"
    path.write_bytes(b"not a zip file")
    with pytest.raises(zipfile.BadZipFile):
        extract(path, settings.snapshot()["documents"])


def test_pdf_text(tmp_path, settings):
    from pypdf import PdfWriter
    from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

    writer = PdfWriter()
    page = writer.add_blank_page(612, 792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
    )
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 50 700 Td (Training budget for this project is 8000 EGP.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    path = tmp_path / "budget.pdf"
    writer.write(path)
    parts, _ = extract(path, settings.snapshot()["documents"])
    assert "8000" in parts[0][1] and parts[0][0] == "Page 1"


@pytest.mark.parametrize('suffix,content,message', [
    ('.csv', 'name,name\na,b', 'different, non-empty'),
    ('.csv', 'name,price\na,4,5', 'row 2'),
    ('.csv', 'name,price\n"broken,4', 'quoted field'),
    ('.jsonl', '{"ok":1}\n{"broken":', 'JSONL line 2'),
    ('.json', '{broken', 'near line 1'),
])
def test_dataset_errors_explain_how_to_fix(tmp_path, settings, suffix, content, message):
    path = tmp_path / ('dataset' + suffix)
    path.write_text(content, encoding='utf8')
    with pytest.raises(ValueError, match=message):
        extract(path, settings.snapshot()['documents'])


@pytest.mark.parametrize('suffix,content,expected', [
    ('.tsv', 'المنتج\tالسعر\nمكتب\t7250', 'السعر: 7250'),
    ('.jsonl', '{"product":"Cedar","price":7250}\n{"product":"Palm","price":4300}', '4300'),
    ('.csv', 'name,details\nCedar,"includes desk, chair"', 'includes desk, chair'),
])
def test_more_dataset_formats_keep_rows(tmp_path, settings, suffix, content, expected):
    path = tmp_path / ('dataset' + suffix)
    path.write_text(content, encoding='utf8')
    parts, _ = extract(path, settings.snapshot()['documents'])
    assert expected in ' '.join(p[1] for p in parts)
