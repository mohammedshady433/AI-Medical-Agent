import pytest
from app.core.settings import Settings
from app.agents.tools import Registry, calculate
from app.rag.library import normalize


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://example.com",
        "http://192.168.1.2:11434",
        "http://127.0.0.1.evil.com:11434",
        "http://name:password@127.0.0.1:11434",
    ],
)
def test_offline_rejects_remote_endpoint(settings, endpoint):
    with pytest.raises(ValueError):
        settings.update({"providers": {"ollama_local": {"endpoint": endpoint}}})
    assert settings.snapshot()["providers"]["ollama_local"]["endpoint"] == "http://127.0.0.1:11434"


def test_no_cloud_tag_offline(settings):
    with pytest.raises(ValueError):
        settings.update({"models": {"offline": {"id": "gpt-oss:120b-cloud"}}})


def test_120b_is_explicit(settings):
    with pytest.raises(ValueError):
        settings.update({"models": {"offline": {"id": "gpt-oss:120b"}}})
    settings.update({"experimental_local_120b_accepted": True, "models": {"offline": {"id": "gpt-oss:120b"}}})


def test_mode_persists(settings):
    settings.update({"mode": "offline"})
    assert Settings(settings.data).snapshot()["mode"] == "offline"


def test_embedding_stays_local(settings):
    with pytest.raises(ValueError):
        settings.update({"documents": {"embedding": {"endpoint": "https://example.com"}}})


@pytest.mark.parametrize(
    "expression",
    ['__import__("os").system("echo nope")', "2**100000", 'open("private")', "[1,2][0]", "True+1"],
)
def test_calculator_no_code_execution(expression):
    with pytest.raises((ValueError, SyntaxError)):
        calculate(expression)


def test_calculator_numbers():
    assert calculate("(18000-3500)/5") == 2900


def test_privacy_tools(library, settings, secrets):
    registry = Registry(library, settings.snapshot(), secrets)
    assert "search_documents" not in registry.allowed
    assert "web_search" not in registry.allowed
    settings.update({"mode": "offline", "privacy": {"web_search": True}})
    registry = Registry(library, settings.snapshot(), secrets)
    assert "search_documents" in registry.allowed and "web_search" not in registry.allowed


async def test_tool_arguments_reject_extra(library, settings, secrets):
    with pytest.raises(ValueError):
        await Registry(library, settings.snapshot(), secrets).execute(
            "calculator", {"expression": "1+1", "command": "bad"}
        )


def test_arabic_normalization():
    assert normalize("إِدَارَة آمال") == "ادارة امال"


def test_user_data_outside_project(settings):
    assert settings.data != settings.root
