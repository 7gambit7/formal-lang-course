from collections import defaultdict
from itertools import product

import networkx as nx
from pyformlang.cfg import CFG
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State, Symbol
from pyformlang.rsa import Box, RecursiveAutomaton
from scipy.sparse import csr_array

from project.adjacency_matrix_fa import AdjacencyMatrixFA, intersect_automata
from project.finite_automata import graph_to_nfa, regex_to_dfa


def _rules_to_rsm(rules: dict[str, list[str]], start: str) -> RecursiveAutomaton:
    """Make one box per nonterminal: a DFA for "regex1 | regex2 | ..."."""
    boxes = {
        Box(regex_to_dfa(" | ".join(f"({r})" for r in regexes)), Symbol(head))
        for head, regexes in rules.items()
    }
    if start not in rules:
        # No rules for the start symbol, so it derives nothing.
        boxes.add(Box(NondeterministicFiniteAutomaton(), Symbol(start)))
    return RecursiveAutomaton(initial_label=Symbol(start), boxes=boxes)


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    """Each rule `A -> X Y Z` becomes the regex `X Y Z` in the box of A."""
    rules: dict[str, list[str]] = defaultdict(list)
    for rule in cfg.productions:
        body = " ".join(str(symbol.value) for symbol in rule.body)
        rules[str(rule.head.value)].append(body or "$")
    start = cfg.start_symbol.value if cfg.start_symbol is not None else "S"
    return _rules_to_rsm(rules, str(start))


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    """Read lines like `A -> regex`. Start symbol is S (or the first head)."""
    rules: dict[str, list[str]] = defaultdict(list)
    for line in ebnf.splitlines():
        if "->" not in line:
            continue
        head, body = (part.strip() for part in line.split("->", 1))
        rules[head].append(body or "$")
    start = "S" if "S" in rules or not rules else next(iter(rules))
    return _rules_to_rsm(rules, start)


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


def _graph_block(closure: csr_array, rsm_from: int, rsm_to: int, n: int) -> csr_array:
    """Part of the closure for RSM states rsm_from -> rsm_to (graph x graph)."""
    rows = slice(rsm_from * n, (rsm_from + 1) * n)
    cols = slice(rsm_to * n, (rsm_to + 1) * n)
    return closure[rows, cols]


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    boxes = _RsmMatrices(rsm)
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    n = graph_fa.states_count
    for label in boxes.ends:
        graph_fa.matrices.setdefault(label, csr_array((n, n), dtype=bool))

    # If a box goes from start to final between u and v, add edge u -box-> v.
    # Repeat while new edges appear.
    changed = True
    while changed:
        changed = False
        closure = intersect_automata(boxes.fa, graph_fa).transitive_closure()
        for label, (starts, finals) in boxes.ends.items():
            known = graph_fa.matrices[label]
            grown = sum(
                (_graph_block(closure, s, f, n) for s, f in product(starts, finals)),
                start=known,
            )
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
