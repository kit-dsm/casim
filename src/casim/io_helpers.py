import json
import pickle
from pathlib import Path
from typing import Any, Type


def load_pickle(
        path: str,
        mode: str = "rb"
) -> Any:
    with open(path, mode) as f:
        return pickle.load(f)


def dump_pickle(
        path: str,
        data: Any,
        mode: str = "wb"
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode) as f:
        pickle.dump(data, f)


def load_json(
        path: str,
        mode: str = "r"

) -> dict:
    with open(path, mode) as f:
        return json.load(f)


def dump_json(
        path: str,
        data: dict,
        encoder_cls: Type[json.JSONEncoder] | None = None,
        mode: str = "w",
        indent: int = 4

) -> None:
    with open(path, mode) as f:
        json.dump(data, f, cls=encoder_cls, indent=indent)