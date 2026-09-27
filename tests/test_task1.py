import networkx as nx
import pytest

from project.task1 import (
    GraphInfo,
    build_two_cycles_graph,
    get_graph_info,
    save_graph_to_dot,
)


@pytest.fixture
def sample_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_edges_from(
        [
            (0, 1, {"label": "a"}),
            (1, 2, {"label": "b"}),
            (2, 0, {"label": "a"}),
            (2, 3, {"label": "c"}),
        ]
    )
    return graph


class TestGetGraphInfo:
    def test_counts(self, sample_graph: nx.MultiDiGraph) -> None:
        info = get_graph_info(sample_graph)
        assert isinstance(info, GraphInfo)
        assert info.number_of_nodes == 4
        assert info.number_of_edges == 4

    def test_labels(self, sample_graph: nx.MultiDiGraph) -> None:
        info = get_graph_info(sample_graph)
        assert info.labels == {"a", "b", "c"}

    def test_empty_graph(self) -> None:
        info = get_graph_info(nx.MultiDiGraph())
        assert info.number_of_nodes == 0
        assert info.number_of_edges == 0
        assert info.labels == set()


class TestBuildTwoCyclesGraph:
    def test_structure(self) -> None:
        graph = build_two_cycles_graph(3, 2, ("x", "y"))
        # n + m + 1 shared node
        assert graph.number_of_nodes() == 3 + 2 + 1
        # each cycle of k inner nodes contributes k + 1 edges
        assert graph.number_of_edges() == (3 + 1) + (2 + 1)

    def test_labels(self) -> None:
        graph = build_two_cycles_graph(2, 2, ("x", "y"))
        labels = {label for _, _, label in graph.edges(data="label")}
        assert labels == {"x", "y"}


class TestSaveGraphToDot:
    def test_saved_file_is_valid_dot(self, tmp_path) -> None:
        graph = build_two_cycles_graph(2, 2, ("x", "y"))
        path = tmp_path / "graph.dot"
        save_graph_to_dot(graph, path)

        assert path.exists()

        restored = nx.drawing.nx_pydot.read_dot(str(path))
        assert restored.number_of_nodes() == graph.number_of_nodes()
