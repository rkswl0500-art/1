"""테스트를 개발자의 .env 로부터 격리합니다.

이게 없으면 저장소 루트에 .env 가 있을 때 테스트가 그 값을 읽습니다. 실제로
스텁 실험용 .env 를 두고 돌렸다가 타임아웃 경고 테스트가 조용히 깨졌습니다 —
테스트는 실패했지만, 반대로 **깨진 코드가 통과하는** 방향으로도 똑같이
일어날 수 있습니다.
"""

from __future__ import annotations

import os

import pytest

from debate.config import Settings


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch, tmp_path):
    """DEBATE_* 환경변수를 지우고 .env 탐색 경로를 빈 디렉터리로 돌립니다."""
    for key in list(os.environ):
        if key.startswith("DEBATE_"):
            monkeypatch.delenv(key, raising=False)

    empty = tmp_path / "no-env"
    empty.mkdir()
    monkeypatch.setitem(Settings.model_config, "env_file", str(empty / ".env"))
    yield
