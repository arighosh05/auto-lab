import pytest
from pathlib import Path


@pytest.fixture
def session_dir(tmp_path: Path) -> Path:
    return tmp_path
