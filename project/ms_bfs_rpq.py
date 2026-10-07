import numpy as np
from networkx import MultiDiGraph
from scipy.sparse import csr_array, eye_array, kron

from project.adjacency_matrix_fa import AdjacencyMatrixFA
from project.task2 import graph_to_nfa, regex_to_dfa


def ms_bfs_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    regex_fa = AdjacencyMatrixFA(regex_to_dfa(regex))

    starts = sorted(graph_fa.start_states)
    n = graph_fa.states_count
    k = regex_fa.states_count
    if not starts or k == 0:
        return set()

    rows, cols = [], []
    for i, graph_start in enumerate(starts):
        for regex_start in regex_fa.start_states:
            rows.append(i * k + regex_start)
            cols.append(graph_start)
    front = csr_array(
        (np.ones(len(rows), dtype=bool), (rows, cols)),
        shape=(len(starts) * k, n),
        dtype=bool,
    )

    # Applying the regex transition for every start-node block at once:
    # row (i, q) of the front moves to row (i, q') iff q -a-> q'.
    steps = [
        (
            csr_array(
                kron(
                    eye_array(len(starts), dtype=bool),
                    regex_fa.matrices[symbol].T,
                    "csr",
                ),
                dtype=bool,
            ),
            graph_fa.matrices[symbol],
        )
        for symbol in regex_fa.matrices.keys() & graph_fa.matrices.keys()
    ]

    visited = front
    while front.nnz > 0:
        next_front = csr_array(front.shape, dtype=bool)
        for regex_step, graph_step in steps:
            next_front = next_front + regex_step @ (front @ graph_step)
        front = next_front > visited
        visited = visited + front

    result = set()
    visited_rows, visited_cols = visited.nonzero()
    for row, graph_node in zip(visited_rows, visited_cols):
        start_index, regex_state = divmod(int(row), k)
        if regex_state in regex_fa.final_states and graph_node in graph_fa.final_states:
            result.add(
                (
                    graph_fa.states[starts[start_index]].value,
                    graph_fa.states[graph_node].value,
                )
            )
    return result
