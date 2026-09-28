from collections.abc import Iterable

import numpy as np
from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton,
    State,
    Symbol,
)
from scipy.sparse import csr_array, eye_array, kron


def _to_symbol(symbol) -> Symbol:
    return symbol if isinstance(symbol, Symbol) else Symbol(symbol)


class AdjacencyMatrixFA:
    """Finite automaton represented as a boolean decomposition of its
    adjacency matrix: one sparse boolean matrix per symbol."""

    def __init__(self, automaton: NondeterministicFiniteAutomaton | None = None):
        self.states: list[State] = []
        self.state_to_index: dict[State, int] = {}
        self.start_states: set[int] = set()
        self.final_states: set[int] = set()
        self.matrices: dict[Symbol, csr_array] = {}

        if automaton is None:
            return

        self.states = list(automaton.states)
        self.state_to_index = {state: i for i, state in enumerate(self.states)}
        self.start_states = {self.state_to_index[s] for s in automaton.start_states}
        self.final_states = {self.state_to_index[s] for s in automaton.final_states}

        transitions: dict[Symbol, tuple[list[int], list[int]]] = {}
        for source, by_symbol in automaton.to_dict().items():
            for symbol, targets in by_symbol.items():
                if not isinstance(targets, set):
                    targets = {targets}
                rows, cols = transitions.setdefault(_to_symbol(symbol), ([], []))
                for target in targets:
                    rows.append(self.state_to_index[source])
                    cols.append(self.state_to_index[target])

        n = self.states_count
        for symbol, (rows, cols) in transitions.items():
            self.matrices[symbol] = csr_array(
                (np.ones(len(rows), dtype=bool), (rows, cols)),
                shape=(n, n),
                dtype=bool,
            )

    @property
    def states_count(self) -> int:
        return len(self.states)

    def accepts(self, word: Iterable[Symbol]) -> bool:
        current = np.zeros(self.states_count, dtype=bool)
        current[list(self.start_states)] = True

        for symbol in word:
            matrix = self.matrices.get(_to_symbol(symbol))
            if matrix is None:
                return False
            current = matrix.T @ current
            if not current.any():
                return False

        return any(current[i] for i in self.final_states)

    def transitive_closure(self) -> csr_array:
        """Reflexive-transitive closure of the adjacency matrix: element
        (i, j) is True iff state j is reachable from state i."""
        closure = eye_array(self.states_count, dtype=bool, format="csr")
        for matrix in self.matrices.values():
            closure = closure + matrix

        while True:
            next_closure = closure + closure @ closure
            if next_closure.nnz == closure.nnz:
                return closure
            closure = next_closure

    def is_empty(self) -> bool:
        closure = self.transitive_closure()
        return not any(
            closure[start, final]
            for start in self.start_states
            for final in self.final_states
        )


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    """Intersect two automata via the tensor (Kronecker) product.

    State (i, j) of the result has index i * n2 + j, where n2 is the
    number of states of automaton2.
    """
    result = AdjacencyMatrixFA()
    n2 = automaton2.states_count

    result.states = [
        State((s1, s2)) for s1 in automaton1.states for s2 in automaton2.states
    ]
    result.state_to_index = {state: i for i, state in enumerate(result.states)}
    result.start_states = {
        i1 * n2 + i2 for i1 in automaton1.start_states for i2 in automaton2.start_states
    }
    result.final_states = {
        i1 * n2 + i2 for i1 in automaton1.final_states for i2 in automaton2.final_states
    }

    for symbol in automaton1.matrices.keys() & automaton2.matrices.keys():
        result.matrices[symbol] = csr_array(
            kron(automaton1.matrices[symbol], automaton2.matrices[symbol], "csr"),
            dtype=bool,
        )

    return result
