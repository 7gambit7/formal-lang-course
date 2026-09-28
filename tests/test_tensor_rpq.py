import networkx as nx
import pytest

from project.tensor_rpq import tensor_based_rpq


@pytest.fixture
def cycle_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_edges_from(
        [
            (0, 1, {"label": "a"}),
            (1, 2, {"label": "a"}),
            (2, 0, {"label": "b"}),
        ]
    )
    return graph


class TestTensorBasedRPQ:
    def test_single_start(self, cycle_graph: nx.MultiDiGraph) -> None:
        assert tensor_based_rpq("a a", cycle_graph, {0}, {0, 1, 2}) == {(0, 2)}

    def test_star_includes_empty_path(self, cycle_graph: nx.MultiDiGraph) -> None:
        assert tensor_based_rpq("a*", cycle_graph, {0}, {0, 1, 2}) == {
            (0, 0),
            (0, 1),
            (0, 2),
        }

    def test_respects_final_nodes(self, cycle_graph: nx.MultiDiGraph) -> None:
        assert tensor_based_rpq("(a a b)*", cycle_graph, {0, 1}, {0}) == {(0, 0)}

    def test_no_answer(self, cycle_graph: nx.MultiDiGraph) -> None:
        assert tensor_based_rpq("b b", cycle_graph, {0, 1, 2}, {0, 1, 2}) == set()
