import json
import pickle
from pathlib import Path
from typing import Any, Type


def load_pickle(path: str, mode: str = "rb") -> Any:
    with open(path, mode) as f:
        return pickle.load(f)


def dump_pickle(path: str, data: Any, mode: str = "wb") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode) as f:
        pickle.dump(data, f)


def load_json(path: str, mode: str = "r") -> dict:
    with open(path, mode) as f:
        return json.load(f)


def dump_json(
    path,
    data,
    *,
    indent: int = 2,
    sort_keys: bool = True,
    encoder_cls: Type[json.JSONEncoder] | None = None,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(data, cls=encoder_cls, indent=indent, sort_keys=sort_keys)
        + "\n",
        encoding="utf-8",
    )


def dump_jsonl(path, rows) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
