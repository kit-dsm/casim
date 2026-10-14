"""Download benchmark data into ignored directories and check every file's SHA-256.

Usage: python scripts/fetch_data.py {oobp,dhaen,radar,all} [--check]

Each dataset directory holds a committed SOURCE.md and SHA256SUMS.txt; the data
itself is never committed (Section 3.1 of docs/revision_scopes_corrections.md).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import logging
from pathlib import Path
import shutil
import sys
import tarfile
from typing import Callable, IO
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 1 << 20

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Download:
    """One published file and the SHA-256 it must have."""

    url: str
    sha256: str
    name: str


@dataclass(frozen=True)
class Dataset:
    """A directory with SHA256SUMS.txt and the rule that fills it."""

    directory: Path
    downloads: tuple[Download, ...]
    unpack: Callable[[Path, dict[str, Path]], None] | None
    cache: str = ".download"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_sums(directory: Path) -> dict[str, str]:
    """Map each relative path of SHA256SUMS.txt to its SHA-256."""
    sums = {}
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name.lstrip("*")] = digest
    return sums


def download(item: Download, cache: Path) -> Path:
    """Download a file into the cache unless a copy with the right hash exists."""
    target = cache / item.name
    if not (target.exists() and file_sha256(target) == item.sha256):
        target.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s", item.url)
        partial = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(item.url, timeout=600) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out, CHUNK)
        partial.replace(target)
    actual = file_sha256(target)
    if actual != item.sha256:
        raise ValueError(f"{item.url}: SHA-256 {actual}, expected {item.sha256}")
    return target


def write_member(source: IO[bytes], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as out:
        shutil.copyfileobj(source, out, CHUNK)


def unpack_oobp(directory: Path, files: dict[str, Path]) -> None:
    """Extract the Henn groups to the top level and the Albareda tree to albareda/."""
    wanted = expected_sums(directory)
    prefixes = (("OOBP_instances/henn/", ""), ("OOBP_instances/albareda/", "albareda/"))
    with zipfile.ZipFile(files["oobp_instances.zip"]) as archive:
        for member in archive.infolist():
            for prefix, replacement in prefixes:
                if not member.filename.startswith(prefix):
                    continue
                name = replacement + member.filename[len(prefix):]
                if name in wanted:
                    with archive.open(member) as source:
                        write_member(source, directory / name)


def unpack_radar(directory: Path, files: dict[str, Path]) -> None:
    """Extract the listed files of Instances.tar, which the RADAR bag contains."""
    wanted = expected_sums(directory)
    with tarfile.open(files["radar_bag.tar"]) as bag:
        inner = next(member for member in bag.getmembers()
                     if member.name.endswith("/Instances.tar"))
        with tarfile.open(fileobj=bag.extractfile(inner)) as instances:
            for member in instances:
                name = f"selected/{member.name}"
                if member.isfile() and name in wanted:
                    write_member(instances.extractfile(member), directory / name)


DATASETS = {
    "oobp": Dataset(
        ROOT / "scenarios/scenario_henn_online/data/oobp",
        (Download("https://grafo.etsii.urjc.es/optsicom/oobp/oobp-files/oobp_instances.zip",
                  "c8c7e1096ef515d373c770a705ad8aeda975d5874cafcefab2fdb4dcea29e20d",
                  "oobp_instances.zip"),),
        unpack_oobp,
    ),
    "dhaen": Dataset(
        ROOT / "scenarios/scenario_dhaen_online/data",
        tuple(Download(f"https://osf.io/download/{guid}/", digest, f"osf/{name}") for guid, digest, name in (
            ("h4ys6", "c95c1b52df544c10747b68a556d90852a6064ab8ba4debacce207534a2b35b33",
             "README.txt"),
            ("4p9jn", "83eab8a7781b683276aa00b88184975821aaf02053b618a7b5c1d8b55b721d79",
             "Distance matrix/DistanceMatrix_12SubAisles.txt"),
            ("5vfs9", "37f0ebaf4d5e955c1feb685e3c12318c96d6447b23b1fc7d71b29fe0aa5f7fbd",
             "Online problem/Instances/Instances.zip"),
            ("d73kr", "8cd8174132dbbf262f6f0c4961bf3c9ca3a3af2c37d9e8ab0c86840835c6ccbf",
             "Online problem/Results online problem instances.xlsx"),
            ("nru89", "b1f5b16d3c589bc00017c209612b73696a2da5f97757c82b195ee148b29cd402",
             "Online problem/Parameter tuning/Instances.zip"),
            ("nc7gp", "37fff3236247062acd10f758ea4f9bb742721f33220a2ac0e0d4e92e2b95fd4f",
             "Online problem/Parameter tuning/Results parameter tuning.xlsx"),
        )),
        None,
        cache=".",
    ),
    "radar": Dataset(
        ROOT / "scenarios/scenario_walk_or_wait/data/radar",
        (Download("https://radar.kit.edu/radar-backend/archives/mwsv59v8sk9sqaan/versions/1/content",
                  "5d866dae22333c1e0b0dc3277e68dfe5ee8ab7d0267dd9fe3aa60713d9c8b93f",
                  "radar_bag.tar"),),
        unpack_radar,
    ),
}


def check(dataset: Dataset) -> list[str]:
    """Return every listed file that is missing or has another SHA-256."""
    problems = []
    for name, digest in expected_sums(dataset.directory).items():
        path = dataset.directory / name
        if not path.exists():
            problems.append(f"missing {path}")
        elif file_sha256(path) != digest:
            problems.append(f"SHA-256 differs {path}")
    return problems


def fetch(dataset: Dataset) -> None:
    cache = dataset.directory / dataset.cache
    files = {item.name: download(item, cache) for item in dataset.downloads}
    if dataset.unpack is not None:
        dataset.unpack(dataset.directory, files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=[*DATASETS, "all"])
    parser.add_argument("--check", action="store_true",
                        help="only check the files already present")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    failed = False
    for name in names:
        dataset = DATASETS[name]
        if not args.check:
            fetch(dataset)
        problems = check(dataset)
        for problem in problems:
            logger.error(problem)
        count = len(expected_sums(dataset.directory))
        logger.info("%s: %d of %d files verified", name, count - len(problems), count)
        failed |= bool(problems)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
