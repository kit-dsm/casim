import pickle
from pathlib import Path
from typing import Any


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


