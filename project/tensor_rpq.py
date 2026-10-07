from networkx import MultiDiGraph

from project.adjacency_matrix_fa import AdjacencyMatrixFA, intersect_automata
from project.finite_automata import graph_to_nfa, regex_to_dfa


def tensor_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    """Return pairs (start, final) of graph nodes connected by a path whose
    labels form a word from the language of regex."""
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    regex_fa = AdjacencyMatrixFA(regex_to_dfa(regex))
    closure = intersect_automata(graph_fa, regex_fa).transitive_closure()

    n_regex = regex_fa.states_count
    result = set()
    for graph_start in graph_fa.start_states:
        for graph_final in graph_fa.final_states:
            if any(
                closure[
                    graph_start * n_regex + regex_start,
                    graph_final * n_regex + regex_final,
                ]
                for regex_start in regex_fa.start_states
                for regex_final in regex_fa.final_states
            ):
                result.add(
                    (
                        graph_fa.states[graph_start].value,
                        graph_fa.states[graph_final].value,
                    )
                )
    return result
