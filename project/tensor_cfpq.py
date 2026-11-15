from collections import defaultdict

import networkx as nx
from pyformlang.cfg import CFG
from pyformlang.finite_automaton import (
    Epsilon,
    EpsilonNFA,
    NondeterministicFiniteAutomaton,
    State,
    Symbol,
)
from pyformlang.rsa import Box, RecursiveAutomaton
from scipy.sparse import csr_array

from project.adjacency_matrix_fa import AdjacencyMatrixFA
from project.finite_automata import graph_to_nfa
from project.ms_bfs_rpq import ms_bfs


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    """Rule `A -> X Y Z` becomes the path X, Y, Z in the box of A."""
    box_nfas: dict[str, EpsilonNFA] = defaultdict(EpsilonNFA)
    for rule_no, rule in enumerate(cfg.productions):
        box_nfa = box_nfas[rule.head.value]
        path = [State("in")]
        path += [State((rule_no, pos)) for pos in range(1, len(rule.body))]
        path.append(State("out"))
        box_nfa.add_start_state(path[0])
        box_nfa.add_final_state(path[-1])
        if not rule.body:
            box_nfa.add_transition(path[0], Epsilon(), path[-1])
        for src, sym, dst in zip(path, rule.body, path[1:]):
            box_nfa.add_transition(src, Symbol(sym.value), dst)

    start = cfg.start_symbol.value if cfg.start_symbol is not None else "S"
    box_nfas.setdefault(start, EpsilonNFA())  # start may have no rules
    return RecursiveAutomaton(
        initial_label=Symbol(start),
        boxes={Box(nfa.minimize(), Symbol(head)) for head, nfa in box_nfas.items()},
    )


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    """Lines `A -> regex`, start symbol S."""
    return RecursiveAutomaton.from_text(ebnf)


class _RsmMatrices:
    """All boxes of the RSM as one automaton, plus start/final states of each box."""

    def __init__(self, rsm: RecursiveAutomaton) -> None:
        nfa = NondeterministicFiniteAutomaton()
        ends: dict[Symbol, tuple[list[State], list[State]]] = {}
        for label, box in rsm.boxes.items():
            dfa = box.dfa

            def lift(state, label=label) -> State:
                return State((label.value, state.value))

            for source, symbol, target in dfa:
                nfa.add_transition(lift(source), symbol, lift(target))
            starts = [lift(s) for s in dfa.start_states]
            finals = [lift(s) for s in dfa.final_states]
            # Also adds states that have no transitions.
            for state in starts:
                nfa.add_start_state(state)
            for state in finals:
                nfa.add_final_state(state)
            ends[label] = (starts, finals)

        self.fa = AdjacencyMatrixFA(nfa)
        index = self.fa.state_to_index
        self.ends = {
            label: ([index[s] for s in starts], [index[f] for f in finals])
            for label, (starts, finals) in ends.items()
        }


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    boxes = _RsmMatrices(rsm)
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    n = graph_fa.states_count
    stride = boxes.fa.states_count
    for label in boxes.ends:
        graph_fa.matrices.setdefault(label, csr_array((n, n), dtype=bool))

    # BFS from every node. If a box goes from start to final between u and v,
    # add edge u -box-> v. Repeat while new edges appear.
    changed = True
    while changed:
        changed = False
        bfs_result = ms_bfs(graph_fa, boxes.fa, list(range(n)))
        for label, (_, finals) in boxes.ends.items():
            known = graph_fa.matrices[label]
            # rows f, f + stride, ...: box final f reached from each node
            grown = sum((bfs_result[f::stride] for f in finals), start=known)
            if grown.nnz > known.nnz:
                graph_fa.matrices[label] = csr_array(grown, dtype=bool)
                changed = True

    reach = graph_fa.matrices.get(rsm.initial_label)
    if reach is None:
        return set()
    return {
        (graph_fa.states[i].value, graph_fa.states[j].value)
        for i, j in zip(*reach.nonzero())
        if i in graph_fa.start_states and j in graph_fa.final_states
    }
