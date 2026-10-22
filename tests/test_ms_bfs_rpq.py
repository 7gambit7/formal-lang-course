import networkx as nx
import pytest

from project.ms_bfs_rpq import ms_bfs_based_rpq
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


class TestMsBfsBasedRPQ:
    def test_single_start(self, cycle_graph: nx.MultiDiGraph) -> None:
        assert ms_bfs_based_rpq("a a", cycle_graph, {0}, {0, 1, 2}) == {(0, 2)}

    def test_star_includes_empty_path(self, cycle_graph: nx.MultiDiGraph) -> None:
        assert ms_bfs_based_rpq("a*", cycle_graph, {0}, {0, 1, 2}) == {
            (0, 0),
            (0, 1),
            (0, 2),
        }

    def test_starts_are_independent(self, cycle_graph: nx.MultiDiGraph) -> None:
        assert ms_bfs_based_rpq("a b", cycle_graph, {0, 1}, {0, 1, 2}) == {(1, 0)}

    def test_respects_final_nodes(self, cycle_graph: nx.MultiDiGraph) -> None:
        assert ms_bfs_based_rpq("(a a b)*", cycle_graph, {0, 1}, {0}) == {(0, 0)}

    def test_no_answer(self, cycle_graph: nx.MultiDiGraph) -> None:
        assert ms_bfs_based_rpq("b b", cycle_graph, {0, 1, 2}, {0, 1, 2}) == set()

    def test_same_node_reached_in_different_regex_states(self) -> None:
        graph = nx.MultiDiGraph()
        graph.add_edges_from(
            [
                (0, 1, {"label": "a"}),
                (0, 1, {"label": "b"}),
                (1, 2, {"label": "c"}),
            ]
        )
        assert ms_bfs_based_rpq("b c", graph, {0}, {2}) == {(0, 2)}

    @pytest.mark.parametrize(
        "regex", ["a*", "(a | b)* a", "a b*", "(a a b)* | b", "a (b a)* b"]
    )
    def test_matches_tensor_based_rpq(
        self, cycle_graph: nx.MultiDiGraph, regex: str
    ) -> None:
        nodes = set(cycle_graph.nodes)
        assert ms_bfs_based_rpq(regex, cycle_graph, nodes, nodes) == tensor_based_rpq(
            regex, cycle_graph, nodes, nodes
        )
