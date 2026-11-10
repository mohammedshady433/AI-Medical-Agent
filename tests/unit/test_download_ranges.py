import hashlib
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from download_runtime import download_ranges


def test_parallel_download_resumes_exact_byte_order(tmp_path):
    payload = bytes(range(256)) * 500
    partial = tmp_path / "model.part"
    partial.write_bytes(payload[:3333])

    def response(request):
        start, end = map(int, request.headers["Range"].removeprefix("bytes=").split("-"))
        assert start >= 3333
        return httpx.Response(
            206,
            content=payload[start : end + 1],
            headers={"content-range": f"bytes {start}-{end}/{len(payload)}"},
        )

    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        download_ranges(client, "https://example.com/model", partial, len(payload), chunk_size=8192)
    assert hashlib.sha256(partial.read_bytes()).digest() == hashlib.sha256(payload).digest()


def test_invalid_resume_size_cannot_append(tmp_path):
    partial = tmp_path / "model.part"
    partial.write_bytes(b"too long")
    with httpx.Client() as client, pytest.raises(ValueError, match="larger"):
        download_ranges(client, "https://example.com/model", partial, 2)
    assert partial.read_bytes() == b"too long"
