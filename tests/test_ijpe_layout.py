"""The layout scale reproduces the reported 15 m aisle-change distance."""

from casim.sources.grid_layout import GridLayout


def test_adjacent_aisle_rails_are_15m_apart_across_cross_aisle(tmp_path):
    drawing = tmp_path / "layout.csv"
    drawing.write_text(
        "-90;-90;-90;-90;-90;-90\n"
        "1001;-1;1002;1003;-2;1004\n"
        "-90;-90;-90;-90;-90;-90\n", encoding="utf-8")
    layout, _ = GridLayout(manual_cross_aisle_y=None).build(drawing)
    graph = layout.layout_network.graph
    cross_edges = [data["weight"] for _, _, data in graph.edges(data=True)
                   if data.get("edge_type") == "cross_aisle"]
    rung_edges = [data["weight"] for _, _, data in graph.edges(data=True)
                  if data.get("edge_type") == "rung"]
    assert set(cross_edges) == {15000}
    assert set(rung_edges) == {3850}
