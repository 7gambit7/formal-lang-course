import networkx as nx
import pytest

from project.finite_automata import graph_to_nfa, regex_to_dfa


@pytest.fixture
def linear_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_edges_from(
        [
            (0, 1, {"label": "x"}),
            (1, 2, {"label": "x"}),
        ]
    )
    return graph


class TestRegexToDfa:
    def test_result_is_deterministic(self) -> None:
        dfa = regex_to_dfa("a (b | c)* d")
        assert dfa.is_deterministic()

    def test_result_is_minimal(self) -> None:
        dfa = regex_to_dfa("a (b | c)* d")
        assert len(dfa.states) == len(dfa.minimize().states)

    def test_accepts_word_from_language(self) -> None:
        dfa = regex_to_dfa("a b c")
        assert dfa.accepts(["a", "b", "c"])

    def test_rejects_word_not_from_language(self) -> None:
        dfa = regex_to_dfa("a b c")
        assert not dfa.accepts(["a", "b"])

    def test_kleene_star_accepts_empty_word(self) -> None:
        dfa = regex_to_dfa("a*")
        assert dfa.accepts([])
        assert dfa.accepts(["a", "a", "a"])

    def test_empty_regex_defines_empty_language(self) -> None:
        dfa = regex_to_dfa("")
        assert dfa.is_empty()


class TestGraphToNfa:
    def test_edges_become_transitions(self, linear_graph: nx.MultiDiGraph) -> None:
        nfa = graph_to_nfa(linear_graph, {0}, {2})
        assert nfa.accepts(["x", "x"])
        assert not nfa.accepts(["x"])

    def test_empty_start_and_final_mark_all_nodes(
        self, linear_graph: nx.MultiDiGraph
    ) -> None:
        nfa = graph_to_nfa(linear_graph, set(), set())
        assert nfa.accepts([])
        assert nfa.accepts(["x"])
        assert nfa.accepts(["x", "x"])

    def test_explicit_start_and_final(self, linear_graph: nx.MultiDiGraph) -> None:
        nfa = graph_to_nfa(linear_graph, {0}, {1})
        assert nfa.accepts(["x"])
        assert not nfa.accepts(["x", "x"])

    def test_isolated_nodes(self) -> None:
        graph = nx.MultiDiGraph()
        graph.add_nodes_from([0, 1])
        nfa = graph_to_nfa(graph, {0}, {0})
        assert nfa.accepts([])
        assert not nfa.accepts(["a"])
