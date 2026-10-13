"""Shared loader support for CASIM scenarios."""

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from casim.domain_objects.sim_domain import SimWarehouseDomain


class DataLoader(ABC):
    """Read scenario data from a configured directory."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.cache_path: Path | None = None

    @abstractmethod
    def load(self, *args, **kwargs) -> SimWarehouseDomain:
        pass

    def _load_text(self, filename: str, encoding: str = "utf-8") -> list[str]:
        path = self.data_dir / filename if not Path(filename).is_absolute() else Path(filename)
        with open(path, "r", encoding=encoding) as f:
            return [line.strip() for line in f if line.strip()]

    def _load_csv(self, filename: str, sep: str = ",", **kwargs) -> pd.DataFrame:
        path = self.data_dir / filename if not Path(filename).is_absolute() else Path(filename)
        return pd.read_csv(path, sep=sep, **kwargs)
