"""Bounded, local document readers. Never open Office, run macros, or fetch links."""

import os
import csv, io, json, posixpath, re, subprocess, tempfile, zipfile
from pathlib import Path
from html.parser import HTMLParser
from email import policy
from email.parser import BytesParser
from defusedxml import ElementTree as ET

WORD = {".docx", ".docm", ".dotx", ".dotm"}
SLIDES = {".pptx", ".pptm", ".ppsx", ".ppsm", ".potx", ".potm"}
SHEETS = {".xlsx", ".xlsm", ".xltx", ".xltm"}
LEGACY = {
    ".doc",
    ".dot",
    ".ppt",
    ".pps",
    ".pot",
    ".xls",
    ".rtf",
    ".odt",
    ".odp",
    ".ods",
    ".epub",
    ".msg",
    ".mht",
    ".mhtml",
}
TEXT = {".txt", ".md", ".markdown", ".log", ".rst"}
EXTRA = WORD | SLIDES | SHEETS | LEGACY | TEXT | {".csv", ".tsv", ".json", ".jsonl", ".html", ".htm", ".eml"}
SUPPORTED = EXTRA | {".pdf", ".xml", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
FILE_FILTER = "Documents (" + " ".join("*" + s for s in sorted(SUPPORTED)) + ")"
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
PARTIAL = "الصور والمخططات والملفات المرفقة مش داخلة في النص المستخرج؛ راجع الأصل."
LIMIT = "محتوى الملف أكبر من حدود القراءة الآمنة. قسّمه لأجزاء أصغر."


def text_file(path):
    data = Path(path).read_bytes()
    try:
        return data.decode("utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("احفظ النص بترميز UTF-8 وجرب تاني.")


def paragraphs(text, location="فقرة", safe=True):
    # Keep each source line intact; never cut a number or qualification.
    return [
        (f"{location} {i}", p.strip(), safe and len(p.strip()) <= 750)
        for i, p in enumerate(re.split(r"\n\s*\n", text), 1)
        if p.strip()
    ]


class Package:
    def __init__(self, path):
        self.z = zipfile.ZipFile(path)
        entries = self.z.infolist()
        if len(entries) > 20000 or sum(i.file_size for i in entries) > 512 * 1024**2:
            self.z.close()
            raise ValueError(LIMIT)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.z.close()

    def xml(self, name):
        info = self.z.getinfo(name)
        if info.file_size > 16 * 1024**2:
            raise ValueError(LIMIT)
        return ET.fromstring(self.z.read(name))

    def links(self, part):
        folder, leaf = posixpath.split(part)
        name = folder + "/_rels/" + leaf + ".rels"
        if name not in self.z.namelist():
            return {}
        return {
            r.get("Id"): posixpath.normpath(posixpath.join(folder, r.get("Target", ""))).lstrip("/")
            for r in self.xml(name)
            if r.get("TargetMode") != "External"
        }


def text_nodes(node, tag):
    return "".join(
        (t.text or "")
        if t.tag == tag
        else "\t"
        if t.tag.endswith("}tab")
        else "\n"
        if t.tag.endswith("}br")
        else ""
        for t in node.iter()
    )


def office(path):
    ext = Path(path).suffix.lower()
    chunks = []
    warnings = [PARTIAL]
    with Package(path) as pack:
        if any(n.endswith("vbaProject.bin") for n in pack.z.namelist()):
            warnings.append("الماكرو متجاهل ولم يتم تشغيله.")
        if ext in WORD:
            body = pack.xml("word/document.xml").find("w:body", NS)
            para = 0
            table = 0
            for block in body:
                if block.tag == "{" + NS["w"] + "}p":
                    para += 1
                    text = text_nodes(block, "{" + NS["w"] + "}t")
                    if text.strip():
                        chunks.append((f"فقرة {para}", text, len(text) <= 750))
                elif block.tag == "{" + NS["w"] + "}tbl":
                    table += 1
                    for r, row in enumerate(block.findall("w:tr", NS), 1):
                        cells = [
                            " / ".join(
                                text_nodes(p, "{" + NS["w"] + "}t") for p in cell.findall(".//w:p", NS)
                            )
                            for cell in row.findall("w:tc", NS)
                        ]
                        text = " | ".join(cells)
                        if text.strip(" |"):
                            chunks.append((f"جدول {table} / صف {r}", text, False))
                    warnings.append("الجداول مستخرجة للمراجعة فقط؛ راجع ارتباط العناوين بالقيم في الأصل.")
            for name in sorted(pack.z.namelist()):
                if re.fullmatch(r"word/(?:header\d+|footer\d+|footnotes|endnotes)\.xml", name):
                    for i, p in enumerate(pack.xml(name).findall(".//w:p", NS), 1):
                        text = text_nodes(p, "{" + NS["w"] + "}t")
                        if text.strip():
                            chunks.append((f"{name} / فقرة {i}", text, len(text) <= 750))
        elif ext in SLIDES:
            part = "ppt/presentation.xml"
            links = pack.links(part)
            for number, item in enumerate(pack.xml(part).findall(".//p:sldId", NS), 1):
                slide = links.get(item.get("{" + NS["r"] + "}id"))
                if not slide:
                    raise ValueError("تعذر قراءة ترتيب الشرائح في الملف.")
                slide_root = pack.xml(slide)
                table_paragraphs = {
                    id(p) for tbl in slide_root.findall(".//a:tbl", NS) for p in tbl.findall(".//a:p", NS)
                }
                if table_paragraphs:
                    warnings.append("الجداول مستخرجة للمراجعة فقط؛ راجع ارتباط العناوين بالقيم في الأصل.")
                for i, p in enumerate(slide_root.findall(".//a:p", NS), 1):
                    text = text_nodes(p, "{" + NS["a"] + "}t")
                    if text.strip():
                        chunks.append(
                            (
                                f"شريحة {number} / فقرة {i}",
                                text,
                                len(text) <= 750 and id(p) not in table_paragraphs,
                            )
                        )
                for note in pack.links(slide).values():
                    if "/notesSlides/" not in note:
                        continue
                    for i, p in enumerate(pack.xml(note).findall(".//a:p", NS), 1):
                        text = text_nodes(p, "{" + NS["a"] + "}t")
                        if text.strip() and text.strip() != str(number):
                            chunks.append((f"شريحة {number} / ملاحظات {i}", text, len(text) <= 750))
        else:
            shared = []
            if "xl/sharedStrings.xml" in pack.z.namelist():
                shared = [text_nodes(si, "{" + NS["s"] + "}t") for si in pack.xml("xl/sharedStrings.xml")]
            links = pack.links("xl/workbook.xml")
            warnings.append(
                "خلايا Excel مستخرجة بالقيم المحفوظة للمراجعة؛ المعادلات مش بتتحسب وتنسيق التواريخ والنسب قد يختلف."
            )
            for sheet in pack.xml("xl/workbook.xml").findall(".//s:sheet", NS):
                name = sheet.get("name", "")
                target = links.get(sheet.get("{" + NS["r"] + "}id"))
                if not target:
                    continue
                for row in pack.xml(target).findall(".//s:sheetData/s:row", NS):
                    cells = []
                    for cell in row.findall("s:c", NS):
                        kind = cell.get("t")
                        val = cell.findtext("s:v", "", NS)
                        if kind == "s":
                            val = shared[int(val)] if val else ""
                        elif kind == "inlineStr":
                            val = text_nodes(cell, "{" + NS["s"] + "}t")
                        elif kind == "b":
                            val = "TRUE" if val == "1" else "FALSE"
                        formula = cell.findtext("s:f", None, NS)
                        if formula is not None:
                            val = f"{val} [formula: {formula}; cached value]"
                        if val:
                            cells.append(f"{cell.get('r', '')}: {val}")
                    if cells:
                        chunks.append((f"ورقة {name} / صف {row.get('r', '')}", " | ".join(cells), False))
    return chunks, list(dict.fromkeys(warnings))


class WebText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "template"):
            self.hidden += 1
        if not self.hidden and tag in ("p", "div", "h1", "h2", "h3", "li", "tr", "br", "section", "article"):
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "template"):
            self.hidden = max(0, self.hidden - 1)
        if not self.hidden and tag in ("p", "div", "li", "tr", "section", "article"):
            self.parts.append("\n\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def html_text(text):
    parser = WebText()
    parser.feed(text)
    return "".join(parser.parts)


def legacy(path):
    from app.core.settings import ROOT

    base = ROOT / "document-runtime"
    java = base / "jre/bin/java.exe"
    if os.name != "nt":
        homes = [os.environ.get("JAVA_HOME", "")]
        try:
            homes.append(subprocess.run(["/usr/libexec/java_home", "-v", "17"], capture_output=True, text=True, timeout=5, check=True).stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass
        java = None
        for home in homes:
            candidate = Path(home) / "bin/java" if home else None
            if candidate and candidate.is_file():
                version = subprocess.run([str(candidate), "-version"], capture_output=True, text=True, timeout=5)
                if re.search(r'version "17[.\"]', version.stdout + version.stderr):
                    java = candidate
                    break
        if java is None:
            raise ValueError("Legacy Office documents need Java 17 on macOS. Install a native JDK 17 or save as DOCX/PPTX/XLSX.")
    jar = base / "tika-app.jar"
    if not java.exists() or not jar.exists():
        raise ValueError("قارئ المستندات الإضافي مش موجود. استخدم الحزمة الكاملة.")
    with tempfile.TemporaryDirectory(prefix="ems-read-") as temp:
        # Local file URI is never interpreted as a remote URL or CLI option.
        out = Path(temp) / "text.txt"
        err = Path(temp) / "error.txt"
        try:
            with out.open("wb") as stdout, err.open("wb") as stderr:
                proc = subprocess.run(
                    [
                        str(java),
                        "-Xmx512m",
                        "-Dfile.encoding=UTF-8",
                        "-Djava.awt.headless=true",
                        "-Djava.io.tmpdir=" + temp,
                        "-Djava.security.manager",
                        "-Djava.security.policy==" + str(base / "offline.policy"),
                        "-cp",
                        str(base) + os.pathsep + str(jar),
                        "EmsRead",
                        str(Path(path).resolve()),
                        str(base / "tika-config.xml"),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=90,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
        except subprocess.TimeoutExpired:
            raise ValueError("قراءة الملف أخدت وقت أطول من الحد المسموح. جرّب تقسيمه أو حفظه بصيغة أحدث.")
        if proc.returncode:
            raise ValueError("تعذر قراءة الملف. ممكن يكون تالف أو محمي بكلمة سر؛ جرّب نسخة غير محمية.")
        if out.stat().st_size > 32 * 1024**2:
            raise ValueError(LIMIT)
        text = out.read_text(encoding="utf8")
    review = Path(path).suffix.lower() in (".xls", ".ods")
    return paragraphs(text, safe=not review), [
        PARTIAL,
        "التنسيق الأصلي قد يختلف؛ راجع الجداول والتواريخ في الملف الأصلي.",
    ]


def extract_extra(path):
    path = Path(path)
    ext = path.suffix.lower()
    if ext in WORD | SLIDES | SHEETS:
        return office(path)
    if ext in LEGACY:
        return legacy(path)
    if ext in TEXT:
        return paragraphs(text_file(path)), []
    if ext in (".csv", ".tsv"):
        text = text_file(path)
        try:
            rows = list(csv.reader(io.StringIO(text), delimiter="\t" if ext == ".tsv" else ",", strict=True))
        except csv.Error as exc:
            raise ValueError("This CSV/TSV has a broken quoted field or an oversized cell. Export a clean UTF-8 CSV with column names in the first row.") from exc
        if not rows:
            return [], []
        headers = rows[0]
        chunks = []
        valid = all(h.strip() for h in headers) and len(set(headers)) == len(headers)
        if not valid:
            raise ValueError("Give every CSV/TSV column a different, non-empty name in the first row, then add the file again.")
        for i, row in enumerate(rows[1:], 2):
            if not any(row):
                continue
            if len(row) != len(headers):
                raise ValueError(f"CSV/TSV row {i} has {len(row)} cells but the header has {len(headers)}. Check delimiters and quote values containing commas.")
            cells = [
                f"{headers[j] if j < len(headers) else 'Column ' + str(j + 1)}: {value}"
                for j, value in enumerate(row)
            ]
            val = "\n".join(cells)
            if any(row):
                chunks.append((f"صف {i}", val, valid and len(row) == len(headers) and len(val) <= 750))
        return chunks, ["أول صف في CSV هو أسماء الأعمدة. راجعها لو الملف مش فيه عناوين."]
    if ext in (".json", ".jsonl"):
        try:
            if ext == ".jsonl":
                data = []
                for line_number, line in enumerate(text_file(path).splitlines(), 1):
                    if not line.strip():
                        continue
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"JSONL line {line_number} is not valid JSON. Put one complete JSON object on each line.") from exc
            else:
                data = json.loads(text_file(path))
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON could not be read near line {exc.lineno}. Check quotes, commas and brackets, then add it again.") from exc
        except RecursionError as exc:
            raise ValueError("This JSON is nested too deeply. Flatten it before importing.") from exc
        chunks = []

        def walk(value, loc, depth=0):
            if depth > 40:
                raise ValueError(LIMIT)
            if isinstance(value, list):
                for i, v in enumerate(value):
                    walk(v, loc + f"[{i}]", depth + 1)
            elif isinstance(value, dict) and any(isinstance(v, (dict, list)) for v in value.values()):
                for k, v in value.items():
                    walk(v, loc + "." + k, depth + 1)
            else:
                text = json.dumps(value, ensure_ascii=False, indent=2)
                chunks.append((loc, text, len(text) <= 750))

        walk(data, "$")
        return chunks, []
    if ext in (".html", ".htm"):
        return paragraphs(html_text(text_file(path))), ["الروابط والصور الخارجية مش بيتم تحميلها."]
    if ext == ".eml":
        msg = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
        body = msg.get_body(preferencelist=("plain", "html"))
        content = body.get_content() if body else ""
        if body and body.get_content_subtype() == "html":
            content = html_text(content)
        headers = "\n".join(
            f"{key}: {str(msg[key])}" for key in ("From", "To", "Subject", "Date") if msg[key]
        )
        return paragraphs(headers + "\n\n" + content), ["مرفقات البريد مش مستوردة. ضيف كل مرفق كملف منفصل."]
    raise ValueError("الصيغة دي مش مدعومة. جرّب PDF نصي أو Word أو PowerPoint أو Excel أو CSV.")
