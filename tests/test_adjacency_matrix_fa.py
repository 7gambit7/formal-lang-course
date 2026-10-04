import networkx as nx
import pytest

from project.adjacency_matrix_fa import AdjacencyMatrixFA, intersect_automata
from project.task2 import graph_to_nfa, regex_to_dfa


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


class TestAdjacencyMatrixFA:
    def test_accepts_word_from_language(self) -> None:
        fa = AdjacencyMatrixFA(regex_to_dfa("a b* c"))
        assert fa.accepts("abbbc")
        assert fa.accepts("ac")

    def test_rejects_word_not_from_language(self) -> None:
        fa = AdjacencyMatrixFA(regex_to_dfa("a b* c"))
        assert not fa.accepts("ab")
        assert not fa.accepts("abd")
        assert not fa.accepts("")

    def test_accepts_empty_word(self) -> None:
        fa = AdjacencyMatrixFA(regex_to_dfa("a*"))
        assert fa.accepts("")

    def test_from_nfa(self, cycle_graph: nx.MultiDiGraph) -> None:
        fa = AdjacencyMatrixFA(graph_to_nfa(cycle_graph, {0}, {0}))
        assert fa.accepts("aab")
        assert fa.accepts("aabaab")
        assert not fa.accepts("aa")

    def test_is_not_empty(self) -> None:
        assert not AdjacencyMatrixFA(regex_to_dfa("a b")).is_empty()

    def test_is_empty_when_final_unreachable(self) -> None:
        graph = nx.MultiDiGraph()
        graph.add_edge(0, 1, label="a")
        fa = AdjacencyMatrixFA(graph_to_nfa(graph, {1}, {0}))
        assert fa.is_empty()


class TestIntersectAutomata:
    def test_intersection_language(self) -> None:
        fa1 = AdjacencyMatrixFA(regex_to_dfa("a* b"))
        fa2 = AdjacencyMatrixFA(regex_to_dfa("a a (a | b)*"))
        intersection = intersect_automata(fa1, fa2)
        assert intersection.accepts("aab")
        assert intersection.accepts("aaaab")
        assert not intersection.accepts("ab")
        assert not intersection.accepts("aa")

    def test_disjoint_languages(self) -> None:
        fa1 = AdjacencyMatrixFA(regex_to_dfa("a*"))
        fa2 = AdjacencyMatrixFA(regex_to_dfa("b b*"))
        assert intersect_automata(fa1, fa2).is_empty()

    def test_states_count(self) -> None:
        fa1 = AdjacencyMatrixFA(regex_to_dfa("a b"))
        fa2 = AdjacencyMatrixFA(regex_to_dfa("a*"))
        intersection = intersect_automata(fa1, fa2)
        assert intersection.states_count == fa1.states_count * fa2.states_count
