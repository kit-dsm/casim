"""F68: an OOBP Henn location index l is the cell l + 1 of its aisle.

Hand-checked case: one order with two items, ``Aisle 3 Location 0`` and
``Aisle 3 Location 44``. Aisle index 3 is side 1 of physical aisle 3 // 2 = 1,
which the loader numbers x = 2. Location 0 is the first cell, 1 LU behind the
front cross aisle (y = 0), so y = 1; location 44 is the 45th cell, 1 LU before
the rear cross aisle (y = 46), so y = 45. An S-shape tour from the depot (2 x 0.5
LU stub) across to aisle 2 (5 LU each way) entering and leaving the single aisle
from the front to its deepest pick (2 x 45 LU) is 1 + 10 + 90 = 101 LU.
"""

from pathlib import Path

from casim.sources.oobp import OobpSource


def _write_instance(directory: Path) -> str:
    directory.mkdir(parents=True)
    (directory / "sett1.txt").write_text("\n".join((
        "no_aisles_: 10", "no_cells__: 45", "cell_lengt: 1", "cell_width: 1.5",
        "aisle_widt: 2", "dis_ais_wa: 0.5", "m_no_a_p_b: 45")) + "\n", encoding="ascii")
    (directory / "1s-1-45-0.txt").write_text(
        "Order 0 number of articles 2\n0\tAisle 3\tLocation 0\n1\tAisle 3\tLocation 44\n",
        encoding="ascii")
    (directory / "Arrival_Time_1s-1-45-0.csv").write_text("1;0.0\n", encoding="ascii")
    return "1s-1-45-0.txt"


def test_location_index_is_the_cell_behind_the_front_cross_aisle(tmp_path):
    filepath = _write_instance(tmp_path / "data")
    domain = OobpSource(tmp_path / "data", filepath).initial_state()
    assert sorted((location.x, location.y) for location in domain.storage.locations) == [
        (2, 1), (2, 45)]
    graph = domain.layout.layout_network.graph
    assert graph[(2, 0)][(2, 1)]["weight"] == 1.0
    assert graph[(2, 45)][(2, 46)]["weight"] == 1.0
    distance = domain.layout.layout_network.distance_matrix
    assert distance.loc[[(1, -1)], [(2, 45)]].iloc[0, 0] + \
        distance.loc[[(2, 45)], [(1, -1)]].iloc[0, 0] == 101.0
