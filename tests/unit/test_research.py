import asyncio
import pytest
from app.extensions.research import Research
from app.extensions.web import public_url, SearchResults, PageText, PublicConnection


class Web:
    def __init__(self): self.searches = self.pages = 0
    async def search(self, query, count):
        self.searches += 1
        return [{"title": "مصدر", "url": "https://example.org/article"}]
    async def page(self, url, max_chars):
        self.pages += 1
        return {"url": url, "text": "معلومات موثقة"}


class Model:
    calls = []
    def __init__(self, config, secrets): self.config = config
    async def stream(self, messages, tools):
        self.calls.append(messages)
        assert tools == []
        assert self.config["models"][self.config["mode"]]["max_response_tokens"] <= 1200
        if self.config["models"]["offline"]["id"] == "qwen2.5:7b-instruct":
            assert self.config["models"]["offline"]["reasoning_effort"] is None
        if "JSON array" in messages[-1]["content"]:
            yield {"type": "token", "text": '["موضوع", "topic"]'}
        else:
            yield {"type": "token", "text": "ملخص [R1] [R999]"}


async def test_research_zero_model_and_reopen(settings, secrets):
    settings.update({"assistant": {"research": {"enabled": True}}})
    web = Web()
    def forbidden(*args): raise AssertionError("No model request allowed")
    research = Research(settings, secrets, provider_factory=forbidden, web=web)
    with pytest.raises(ValueError): research.start("سؤال البحث")
    row = research.start("سؤال البحث", confirm_network=True)
    await research.tasks[row["id"]]
    result = research.load(row["id"])
    assert result["status"] == "completed" and result["model_calls"] == 0
    assert web.searches == web.pages == 1
    assert "معلومات موثقة" in result["report"]
    assert Research(settings, secrets).load(row["id"])["report"] == result["report"]


async def test_research_completed_checkpoint_survives_brief_windows_file_lock(settings, secrets, monkeypatch):
    import os
    settings.update({"assistant": {"research": {"enabled": True}}})
    research = Research(settings, secrets, web=Web())
    replace = os.replace
    locked = False
    def once(source, target):
        nonlocal locked
        if not locked and target.suffix == ".json":
            locked = True
            raise PermissionError("Temporary Windows sharing violation")
        return replace(source, target)
    monkeypatch.setattr(os, "replace", once)
    row = research.start("checkpoint", confirm_network=True)
    await research.tasks[row["id"]]
    assert locked and research.load(row["id"])["status"] == "completed"


async def test_research_keeps_distinct_publishers_on_same_host(settings, secrets):
    settings.update({"assistant": {"research": {"enabled": True}}})
    class SharedHost(Web):
        async def search(self, query, count):
            return [{"title": author, "url": "https://example.org/" + author} for author in ["community", "official", "official"]]
    web = SharedHost(); research = Research(settings, secrets, web=web)
    row = research.start("official project", confirm_network=True)
    await research.tasks[row["id"]]
    result = research.load(row["id"])
    assert len(result["sources"]) == web.pages == 2


async def test_research_deep_budgets_and_consent(settings, secrets):
    settings.update({"assistant": {"research": {"enabled": True, "worker_model": "qwen2.5:7b-instruct"}}})
    Model.calls = []
    r = Research(settings, secrets, provider_factory=Model, web=Web())
    assert settings.snapshot()["mode"] == "online"  # chat mode must not change local research
    row = r.start("موضوع", True, True, True, True)
    await r.tasks[row["id"]]
    result = r.load(row["id"])
    assert result["status"] == "completed" and result["mode"] == "offline"
    assert len(Model.calls) == result["model_calls"] == 3
    assert "[R999]" not in result["report"] and "[R1]" in result["report"]
    with pytest.raises(ValueError): await r.model(result, "one more")
    assert len(Model.calls) == 3


async def test_pause_resume_does_not_refetch_completed_steps(settings, secrets):
    settings.update({"assistant": {"research": {"enabled": True}}})
    entered, release = asyncio.Event(), asyncio.Event()
    class WaitingWeb(Web):
        async def search(self, query, count):
            entered.set(); await release.wait()
            return await super().search(query, count)
    web = WaitingWeb(); r = Research(settings, secrets, web=web)
    row = r.start("موضوع", confirm_network=True)
    await entered.wait(); r.pause(row["id"]); release.set()
    await r.tasks[row["id"]]
    assert r.load(row["id"])["status"] == "paused"
    r.resume(row["id"]); await r.tasks[row["id"]]
    assert web.searches == web.pages == 1
    assert r.load(row["id"])["status"] == "completed"


@pytest.mark.parametrize("url", ["http://example.com", "https://localhost/", "https://127.0.0.1/", "https://169.254.169.254/", "https://[::1]/", "https://user:pass@example.com", "https://example.com:8443", "file:///C:/secret"])
def test_research_private_urls_rejected(url):
    with pytest.raises(ValueError): public_url(url)


def test_dns_private_resolution_and_html(monkeypatch):
    monkeypatch.setattr("socket.getaddrinfo", lambda *a, **kw: [(2, 1, 6, "", ("10.0.0.1", 443))])
    with pytest.raises(ValueError): PublicConnection("example.com").connect()
    parser = SearchResults(); parser.feed('<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org">Arabic source</a>')
    assert parser.results[0]["url"] == "https://example.org"
    text = PageText(); text.feed('<p>مرحبا</p><script>ignore this</script>world')
    assert text.parts == ["مرحبا", "world"]
