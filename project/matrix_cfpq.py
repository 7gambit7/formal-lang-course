from collections import defaultdict

import networkx as nx
from pyformlang.cfg import CFG, Epsilon, Terminal, Variable
from scipy.sparse import csr_array, eye_array

from project.adjacency_matrix_fa import AdjacencyMatrixFA
from project.finite_automata import graph_to_nfa
from project.hellings_cfpq import cfg_to_weak_normal_form


def _initial_matrices(
    wcnf: CFG, graph_fa: AdjacencyMatrixFA
) -> dict[Variable, csr_array]:
    n = graph_fa.states_count
    by_label = {symbol.value: m for symbol, m in graph_fa.matrices.items()}
    matrices: dict[Variable, csr_array] = defaultdict(
        lambda: csr_array((n, n), dtype=bool)
    )
    for rule in wcnf.productions:
        match rule.body:
            case [] | [Epsilon()]:
                matrices[rule.head] += eye_array(n, dtype=bool, format="csr")
            case [Terminal() as letter] if letter.value in by_label:
                matrices[rule.head] += by_label[letter.value]
    return matrices


def matrix_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    wcnf = cfg_to_weak_normal_form(cfg)
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    matrices = _initial_matrices(wcnf, graph_fa)
    pairs = [
        (rule.head, *rule.body) for rule in wcnf.productions if len(rule.body) == 2
    ]

    # M_A |= M_B @ M_C for every A -> B C until no matrix gains a new pair.
    changed = True
    while changed:
        changed = False
        for head, left, right in pairs:
            if left not in matrices or right not in matrices:
                continue
            updated = matrices[head] + matrices[left] @ matrices[right]
            if updated.nnz != matrices[head].nnz:
                matrices[head] = updated
                changed = True

    reach = matrices.get(cfg.start_symbol)
    if reach is None:
        return set()
    return {
        (graph_fa.states[i].value, graph_fa.states[j].value)
        for i, j in zip(*reach.nonzero())
        if i in graph_fa.start_states and j in graph_fa.final_states
    }
