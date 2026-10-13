"""Extract the published OOBP values and compare the Henn source files.

Usage: python oobp_reference.py experiments.xlsx oobp_instances.zip
The official downloads are linked at https://grafo.etsii.urjc.es/optsicom/oobp.html.
"""

from __future__ import annotations

import csv
import sys
from decimal import Decimal
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile


NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
DATA = Path(__file__).resolve().parents[1] / "data"
COLUMNS = (
    "instance", "gil_completion_s", "gil_turnover_s",
    "best_known_completion_s", "best_known_turnover_s",
)


def _shared_strings(book: ZipFile) -> list[str]:
    root = ET.fromstring(book.read("xl/sharedStrings.xml"))
    return ["".join(part.text or "" for part in item.iter(f"{NS}t"))
            for item in root.iter(f"{NS}si")]


def _rows(book: ZipFile, sheet: str, strings: list[str]):
    root = ET.fromstring(book.read(sheet))
    for row in root.iter(f"{NS}row"):
        if int(row.attrib["r"]) < 5:
            continue
        values = {}
        for cell in row.iter(f"{NS}c"):
            column = cell.attrib["r"].rstrip("0123456789")
            if column not in "ABCDE":
                continue
            value = cell.find(f"{NS}v")
            if value is None:
                continue
            values[column] = (strings[int(value.text)]
                              if cell.attrib.get("t") == "s" else value.text)
        if "A" in values:
            if set(values) != set("ABCDE"):
                raise ValueError(f"Incomplete row {row.attrib['r']} in {sheet}")
            yield (values["A"], *(format(Decimal(values[column]).quantize(
                Decimal("0.001")).normalize(), "f")
                    for column in "BCDE"))


def extract(workbook: Path, output: Path = DATA / "oobp_reference.csv") -> int:
    with ZipFile(workbook) as book:
        strings = _shared_strings(book)
        records = [*_rows(book, "xl/worksheets/sheet1.xml", strings),
                   *_rows(book, "xl/worksheets/sheet2.xml", strings)]
    if len(records) != 128 or len({row[0] for row in records}) != len(records):
        raise ValueError(f"Expected 128 distinct instances, got {len(records)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        writer.writerows(records)
    return len(records)


def _arrival_key(name: str) -> str:
    stem = Path(name).stem
    stem = stem.removeprefix("Arrival_Time_abc1_").removeprefix("Arrival_Time_abc1-")
    return stem.removesuffix("-0")


def compare_archive(archive: Path) -> tuple[int, list[str]]:
    """Compare every local Henn abc1 instance and arrival file byte for byte."""
    with ZipFile(archive) as book:
        names = book.namelist()
        instances = {Path(name).name: name for name in names
                     if name.startswith("OOBP_instances/henn/abc1/")
                     and Path(name).name.endswith(".txt")
                     and not Path(name).name.startswith("sett")}
        arrivals = {_arrival_key(name): name for name in names
                    if name.startswith("OOBP_instances/henn/abc1/Arrival_Time_")}
        differences = []
        compared = 0
        for local in sorted(DATA.glob("*.txt")):
            source = instances.get(local.name)
            if source is None or local.read_bytes() != book.read(source):
                differences.append(str(local.relative_to(DATA)))
            compared += 1
        for local in sorted((DATA / "HennOnline").glob("Arrival_Time_abc1_*.csv")):
            source = arrivals.get(_arrival_key(local.name))
            if source is None or local.read_bytes() != book.read(source):
                differences.append(str(local.relative_to(DATA)))
            compared += 1
    return compared, differences


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: oobp_reference.py experiments.xlsx oobp_instances.zip")
    count = extract(Path(sys.argv[1]))
    compared, differences = compare_archive(Path(sys.argv[2]))
    print(f"Extracted {count} results; compared {compared} files; "
          f"{len(differences)} byte differences")
    for name in differences:
        print(f"DIFF {name}")
