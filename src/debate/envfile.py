""".env 읽기/쓰기.

브라우저에서 프로바이더 슬롯을 편집하면 여기로 내려옵니다. PowerShell 로 .env 를
만드는 게 새 환경 세팅에서 제일 자주 실패하는 단계라(BOM, 따옴표, 한글 깨짐)
그 단계를 아예 없애는 게 목적입니다.

쓰기 규칙:
  - **BOM 없는 UTF-8**, LF 개행. PowerShell 의 `Out-File -Encoding utf8` 이
    붙이는 BOM 이 바로 그 실패 원인이었습니다.
  - 프로바이더 슬롯 외의 줄은 **그대로 보존**합니다. 주석도, 다른 설정도.
  - 원자적 교체(temp → os.replace). 쓰다가 죽어도 반쪽짜리 .env 가 남지 않습니다.
  - 파일 권한 0600. 키가 들어가는 파일입니다.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

#: UI 가 관리하는 구역. 이 표식 사이만 갈아끼우므로 반복 저장해도 쌓이지 않습니다.
BEGIN = "# >>> debate: providers (UI 가 관리합니다 — 직접 편집해도 되지만 저장 시 덮어씁니다)"
END = "# <<< debate: providers"

_SLOT_LINE = re.compile(r"^\s*DEBATE_PROVIDER_\d+_(NAME|BASE_URL|API_KEY)\s*=", re.I)


@dataclass(frozen=True, slots=True)
class SlotInput:
    name: str
    base_url: str
    api_key: str = ""


def mask_key(key: str | None) -> str:
    """화면에 보여줄 마스킹. 원본은 절대 브라우저로 나가지 않습니다."""
    if not key:
        return ""
    if len(key) < 12:
        return "*" * 8          # 짧은 키는 앞뒤를 보여주면 남는 게 없습니다
    return f"{key[:5]}****...****{key[-3:]}"


def read_raw(path: Path | str) -> str:
    """BOM 이 있든 없든 읽습니다."""
    p = Path(path)
    if not p.is_file():
        return ""
    return p.read_text(encoding="utf-8-sig")


def render_slots(slots: list[SlotInput]) -> str:
    lines = [BEGIN]
    for i, slot in enumerate(slots, start=1):
        lines.append(f"DEBATE_PROVIDER_{i}_NAME={slot.name}")
        lines.append(f"DEBATE_PROVIDER_{i}_BASE_URL={slot.base_url}")
        lines.append(f"DEBATE_PROVIDER_{i}_API_KEY={slot.api_key}")
    lines.append(END)
    return "\n".join(lines)


def merge(existing: str, slots: list[SlotInput]) -> str:
    """기존 내용에서 슬롯만 갈아끼운 전체 텍스트를 만듭니다.

    UI 가 관리하는 구역과, 그 밖에 손으로 적어둔 DEBATE_PROVIDER_* 줄을 모두
    걷어낸 뒤 새 구역을 붙입니다. 나머지 줄(다른 설정·주석)은 순서까지 그대로
    둡니다 — 사용자가 적어둔 것을 UI 가 지우면 안 됩니다.
    """
    kept: list[str] = []
    inside = False
    for line in existing.splitlines():
        stripped = line.strip()
        if stripped == BEGIN:
            inside = True
            continue
        if inside:
            if stripped == END:
                inside = False
            continue
        if _SLOT_LINE.match(line):
            continue              # 구역 밖에 흩어진 옛 슬롯 줄
        kept.append(line)

    while kept and not kept[-1].strip():
        kept.pop()

    body = "\n".join(kept)
    block = render_slots(slots)
    return (f"{body}\n\n{block}\n" if body else f"{block}\n")


def write(path: Path | str, text: str) -> None:
    """BOM 없는 UTF-8 로 원자적 교체."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".env.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.chmod(tmp, 0o600)      # 키가 들어가는 파일
        os.replace(tmp, p)        # 같은 파일시스템 내 원자적 교체
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def save_slots(path: Path | str, slots: list[SlotInput]) -> None:
    write(path, merge(read_raw(path), slots))
