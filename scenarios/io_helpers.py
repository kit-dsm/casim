import pickle
from typing import Any


def dump_pickle(
        path: str,
        data: Any,
        mode: str = "wb"
) -> None:
    with open(path, mode) as f:
        pickle.dump(data, f)
