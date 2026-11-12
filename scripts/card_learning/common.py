from __future__ import annotations

import hashlib
import json
from pathlib import Path

from flylatro.learning.choice_credit_pilot import implementation_hash


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def identity() -> dict:
    root = Path(__file__).parent
    files = {p.name: digest(p.read_bytes()) for p in sorted(root.glob('*.py'))}
    launcher = root.parent / 'run_card_learning.py'
    files[launcher.name] = digest(launcher.read_bytes())
    return {'production_source_sha256': implementation_hash(), 'research_files_sha256': files}
