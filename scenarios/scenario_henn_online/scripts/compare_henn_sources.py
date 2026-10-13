"""Compare the local converted Henn instances with the original OOBP archive."""

from collections import Counter
from pathlib import Path
import re


DATA = Path(__file__).resolve().parents[1] / "data"
ARCHIVE = DATA / "oobp" / "abc1"
INSTANCE = re.compile(r"^(\d+)[sl]-(\d+)-(\d+)-\d+\.txt$")
ARCHIVE_ORDER = re.compile(r"Order (\d+)\s+number of articles (\d+)")
ARCHIVE_ITEM = re.compile(r"\d+\s+Aisle (\d+)\s+Location (\d+)")


def local_instance(path: Path) -> tuple[dict[str, str], list[list[tuple[int, int, str]]]]:
    """Read positions by order from a converted Heßler-Irnich instance."""
    lines = path.read_text(encoding="cp1252").splitlines()
    header: dict[str, str] = {}
    for line in lines[: lines.index("ARTICLE_SECTION")]:
        if ":" in line:
            key, value = line.split(":", 1)
            header[key.strip()] = value.strip()
    skus: dict[int, tuple[int, int, str]] = {}
    start = lines.index("SKU_SECTION") + 1
    end = lines.index("ORDER_SECTION")
    for line in lines[start:end]:
        fields = line.split()
        if fields[:1] == ["ID"]:
            skus[int(fields[1])] = (int(fields[3]), int(fields[5]), fields[-1])
    orders: list[list[tuple[int, int, str]]] = []
    for line in lines[end + 1 :]:
        fields = line.split()
        if fields[:1] == ["NUM_ARTICLES_IN_ORDER"]:
            orders.append([])
        elif fields[:1] == ["ID"]:
            orders[-1].extend([skus[int(fields[1])]] * int(fields[3]))
    return header, orders


def archive_instance(path: Path, cells: int) -> list[list[tuple[int, int, str]]]:
    """Read original positions, mapping doubled aisle IDs and top indexed cells."""
    orders: list[list[tuple[int, int, str]]] = []
    expected = 0
    for line in path.read_text(encoding="cp1252").splitlines():
        order = ARCHIVE_ORDER.fullmatch(line.strip())
        if order:
            if orders and len(orders[-1]) != expected:
                raise ValueError(f"Incorrect item count in {path}")
            if int(order[1]) != len(orders):
                raise ValueError(f"Unexpected order ID in {path}")
            expected = int(order[2])
            orders.append([])
            continue
        item = ARCHIVE_ITEM.fullmatch(line.strip())
        if item:
            aisle, location = map(int, item.groups())
            orders[-1].append((aisle // 2, cells - 1 - location, "right" if aisle % 2 else "left"))
        elif line.strip():
            raise ValueError(f"Unexpected line in {path}: {line}")
    if orders and len(orders[-1]) != expected:
        raise ValueError(f"Incorrect final item count in {path}")
    return orders


def compare(path: Path) -> tuple[str, ...]:
    """Return the semantic comparison columns for one abc1 instance."""
    match = INSTANCE.fullmatch(path.name)
    if match is None:
        raise ValueError(path.name)
    setting, order_count, filename_capacity = map(int, match.groups())
    header, local_orders = local_instance(path)
    settings = {}
    for line in (ARCHIVE / f"sett{setting}.txt").read_text(encoding="cp1252").splitlines():
        if ":" not in line:
            break
        key, value = line.split(":", 1)
        settings[key.strip()] = value.strip()
    cells = int(settings["no_cells__"])
    original_orders = archive_instance(ARCHIVE / path.name, cells)
    if len(local_orders) != order_count:
        raise ValueError(f"Local order count differs from filename: {path}")
    if len(local_orders) != len(original_orders):
        raise ValueError(f"Order count differs: {path}")
    local_positions = Counter(position for order in local_orders for position in order)
    original_positions = Counter(position for order in original_orders for position in order)
    same_orders = sum(Counter(local) == Counter(original) for local, original in zip(local_orders, original_orders))
    local_lines = sum(map(len, local_orders))
    original_lines = sum(map(len, original_orders))
    header_compare = (
        f"aisles {header['NUM_AISLES']}/{settings['no_aisles_']}, "
        f"cells {header['NUM_CELLS']}/{cells}, orders {len(local_orders)}/{len(original_orders)}"
    )
    depot = f"local {header['DEPOT_AISLE']} {header['DEPOT_LOCATION']}; archive files omit depot"
    capacities = (
        f"local {header['PICKER_CAPACITY']}; archive filename {filename_capacity}, "
        f"settings {settings['m_no_a_p_b']}"
    )
    positions = f"{sum((local_positions & original_positions).values())}/{original_lines} matched"
    same_sizes = sum(len(local) == len(original) for local, original in zip(local_orders, original_orders))
    order_lines = (
        f"{same_orders}/{len(local_orders)} orders match positions; "
        f"{same_sizes}/{len(local_orders)} sizes match; {local_lines}/{original_lines} lines"
    )
    return path.name, header_compare, depot, capacities, positions, order_lines


def main() -> None:
    """Print a Markdown comparison table for the 16 local abc1 instances."""
    print("| File | Header (local/archive) | Depot | Capacity | Item positions | Order lines |")
    print("|---|---|---|---|---:|---|")
    for path in sorted(DATA.glob("*.txt")):
        print("| " + " | ".join(compare(path)) + " |")


if __name__ == "__main__":
    main()
