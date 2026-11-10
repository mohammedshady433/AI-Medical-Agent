"""Explicit setup downloads only. Resume interrupted files and verify remote repository hashes."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
import httpx

ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads((ROOT / "config/runtime-lock.json").read_text())


def digest(path, algorithm="sha256"):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, algorithm).hexdigest()


def download(client, url, target, expected=None, size=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    if (
        target.exists()
        and (not size or target.stat().st_size == size)
        and (not expected or digest(target) == expected)
    ):
        return
    partial = target.with_name(target.name + ".part")
    if target.exists() and size and target.stat().st_size < size and not partial.exists():
        target.replace(partial)
    if size and size > 64 * 1024**2:
        download_ranges(client, url, partial, size)
        if expected and digest(partial) != expected:
            partial.unlink(missing_ok=True)
            raise ValueError("Downloaded checksum mismatch: " + target.name)
        partial.replace(target)
        print(f"Verified {target.name}", flush=True)
        return
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            with client.stream(
                "GET", url, headers={"Range": f"bytes={offset}-"} if offset else {}
            ) as response:
                if response.status_code == 416:
                    partial.unlink(missing_ok=True)
                    continue
                response.raise_for_status()
                append = response.status_code == 206 and offset > 0
                if append and not response.headers.get("content-range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Invalid resumed range.")
                with partial.open("ab" if append else "wb") as output:
                    for chunk in response.iter_bytes(1024 * 1024):
                        output.write(chunk)
            if size and partial.stat().st_size != size:
                raise ValueError("Incomplete download.")
            if expected and digest(partial) != expected:
                partial.unlink(missing_ok=True)
                raise ValueError("Downloaded checksum mismatch.")
            partial.replace(target)
            print(f"Verified {target.name}", flush=True)
            return
        except (httpx.HTTPError, ValueError):
            if attempt == 4:
                raise
            time.sleep(min(2**attempt, 10))


def download_ranges(client, url, partial, size, chunk_size=1024**2, workers=16):
    """Bound concurrent transfers; append in order so an interrupted file is resumable."""
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        raise ValueError("Partial download is larger than the model file.")

    def fetch(start):
        end = min(start + chunk_size, size) - 1
        for attempt in range(4):
            try:
                response = client.get(url, headers={"Range": f"bytes={start}-{end}"})
                response.raise_for_status()
                if (
                    response.status_code != 206
                    or response.headers.get("content-range") != f"bytes {start}-{end}/{size}"
                ):
                    raise ValueError("Server returned an unexpected model download range.")
                if len(response.content) != end - start + 1:
                    raise ValueError("Incomplete model range.")
                return response.content
            except (httpx.HTTPError, ValueError):
                if attempt == 3:
                    raise
                time.sleep(min(2**attempt, 4))

    positions = iter(range(offset, size, chunk_size))
    with ThreadPoolExecutor(max_workers=workers) as pool, partial.open("ab") as output:
        pending = [
            pool.submit(fetch, start)
            for start in list(next(positions, None) for _ in range(workers))
            if start is not None
        ]
        progress = offset // (64 * 1024**2)
        while pending:
            data = pending.pop(0).result()
            output.write(data)
            output.flush()
            offset += len(data)
            if offset // (64 * 1024**2) != progress or offset == size:
                progress = offset // (64 * 1024**2)
                print(f"{partial.stem}: {offset / 1024**2:.0f}/{size / 1024**2:.0f} MB", flush=True)
            start = next(positions, None)
            if start is not None:
                pending.append(pool.submit(fetch, start))


def huggingface(component, target):
    spec = LOCK[component]
    repo = spec["repository"]
    rev = spec["revision"]
    with httpx.Client(
        timeout=httpx.Timeout(120, connect=30), follow_redirects=True, trust_env=False
    ) as client:
        response = client.get(f"https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true")
        response.raise_for_status()
        files = response.json()["siblings"]
        manifest = []
        for entry in files:
            name = entry["rfilename"]
            selected = (
                (
                    name.startswith(("kemetone/", "voices/"))
                    or name in ("kemetone.pth", "config.json", "README.md")
                )
                if component == "kemetone"
                else name
                in (
                    "model.bin",
                    "config.json",
                    "tokenizer.json",
                    "vocabulary.json",
                    "preprocessor_config.json",
                    "README.md",
                )
            )
            if component == "chatterbox":
                selected = name in spec["files"]
            if not selected:
                continue
            path = (target / name).resolve()
            if not path.is_relative_to(target.resolve()):
                raise ValueError("Unsafe model path.")
            lfs = entry.get("lfs") or {}
            expected = lfs.get("sha256")
            size = entry.get("size")
            download(client, f"https://huggingface.co/{repo}/resolve/{rev}/{name}", path, expected, size)
            if not expected and entry.get("blobId"):
                content = path.read_bytes()
                git_hash = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
                if git_hash != entry["blobId"]:
                    raise ValueError("Repository file hash mismatch: " + name)
            manifest.append({"path": name, "sha256": digest(path), "size": path.stat().st_size})
        (target / "ems-manifest.json").write_text(
            json.dumps({"repository": repo, "revision": rev, "files": manifest}, indent=2), encoding="utf8"
        )


def ollama(target):
    spec = LOCK["ollama_macos" if sys.platform == "darwin" else "ollama"]
    archive = target / ("ollama.tgz" if sys.platform == "darwin" else "ollama.zip")
    with httpx.Client(timeout=120, follow_redirects=True, trust_env=False) as client:
        download(client, spec["url"], archive, spec["sha256"])
    if sys.platform == "darwin":
        with tarfile.open(archive) as tar:
            tar.extractall(target / "ollama", filter="data")
        return
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not (target / "ollama" / name).resolve().is_relative_to((target / "ollama").resolve()):
                raise ValueError("Unsafe archive path.")
        z.extractall(target / "ollama")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("component", choices=["kemetone", "whisper", "ollama", "chatterbox"])
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    target = Path(args.destination).resolve()
    target.mkdir(parents=True, exist_ok=True)
    ollama(target) if args.component == "ollama" else huggingface(args.component, target)
