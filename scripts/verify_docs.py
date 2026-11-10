"""Check repository documentation links without network access."""
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r'!?\[[^\]]*\]\(([^\s)]+)(?:\s+"[^"]*")?\)')

def check_links():
    files = [ROOT / 'README.md', ROOT / 'CONTRIBUTING.md', ROOT / 'scripts/README.md']
    files += list((ROOT / 'docs').rglob('*.md'))
    files += list((ROOT / 'examples').rglob('*.md'))
    sources = [(p, p.read_text(encoding='utf-8')) for p in files]
    errors = []
    checked = 0
    for path, content in sources:
        for match in LINK.finditer(content):
            url = match.group(1).strip('<>')
            if urlsplit(url).scheme or url.startswith('#'):
                continue
            target = unquote(urlsplit(url).path)
            if not target:
                continue
            checked += 1
            if not (path.parent / target).exists():
                errors.append(f'{path.relative_to(ROOT)}: missing {target}')
    for error in errors:
        print(error)
    if errors:
        raise SystemExit(1)
    print(f'{checked} local documentation links passed across {len(files)} guides.')

if __name__ == '__main__':
    check_links()
