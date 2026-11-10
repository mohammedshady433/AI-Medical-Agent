import pytest
from app.core.settings import Settings
from app.storage.database import Database
from app.rag.library import Library


@pytest.fixture
def settings(tmp_path):
    return Settings(tmp_path)


@pytest.fixture
def database(settings):
    return Database(settings.data)


@pytest.fixture
def library(database, settings):
    return Library(database, settings)


class NoSecrets:
    def get(self, name, env=""):
        return "test-credential-not-a-real-key"


@pytest.fixture
def secrets():
    return NoSecrets()
