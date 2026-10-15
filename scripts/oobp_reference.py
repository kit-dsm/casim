"""Extract the published OOBP values and compare the Henn source files.

Usage: python scripts/oobp_reference.py experiments.xlsx oobp_instances.zip
The official downloads are linked at https://grafo.etsii.urjc.es/optsicom/oobp.html.
Writes ``scenarios/oobp/reference.csv`` in the format of ``casim.reference``: one
row per instance, objective and method, selected by ``source.filepath``.
"""

from __future__ import annotations

import csv
import sys
from decimal import Decimal
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile


NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "scenarios" / "henn_waiting" / "data"
REFERENCE = ROOT / "scenarios" / "oobp" / "reference.csv"
RUNLIST = ROOT / "docs" / "literature_benchmarks" / "l3_instances.csv"
COLUMNS = ("source.filepath", "kpi", "published", "kind", "source")
# (column of the workbook rows, KPI of the scenario, method in the source column)
VALUES = ((1, "max_completion_time", "GRASP+VND"), (2, "max_turnover_time", "GRASP+VND"),
          (3, "max_completion_time", "best known"), (4, "max_turnover_time", "best known"))


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


def instance_key(group: str, stratum: str, path: str) -> str:
    """The instance's name in ``experiments.xlsx``."""
    if group == "Henn":
        setting, orders, _, _ = Path(path).stem.split("-")
        return f"H_{stratum}_{orders}_{setting.rstrip('sl')}"
    stem = Path(path)
    warehouse = stem.name.split("_")[-2]
    return f"A_{int(warehouse)}_{stem.parent.name}_{stem.name.split('_')[-1].split('.')[0]}"


def filepaths() -> dict[str, str]:
    """Workbook instance name -> the instance's path in the oobp data directory."""
    with RUNLIST.open(newline="", encoding="utf-8") as stream:
        return {instance_key(row["group"], row["stratum"], row["path_in_data_oobp"]):
                row["path_in_data_oobp"] for row in csv.DictReader(stream)}


def extract(workbook: Path, output: Path = REFERENCE) -> int:
    with ZipFile(workbook) as book:
        strings = _shared_strings(book)
        records = [*_rows(book, "xl/worksheets/sheet1.xml", strings),
                   *_rows(book, "xl/worksheets/sheet2.xml", strings)]
    if len(records) != 128 or len({row[0] for row in records}) != len(records):
        raise ValueError(f"Expected 128 distinct instances, got {len(records)}")
    paths = filepaths()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        for record in sorted(records, key=lambda row: paths[row[0]]):
            for column, kpi, method in VALUES:
                writer.writerow((paths[record[0]], kpi, record[column], "bound",
                                 f"Gil-Borrás et al. 2020, experiments.xlsx, {method}"))
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
