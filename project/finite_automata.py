from networkx import MultiDiGraph
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    State,
    Symbol,
)
from pyformlang.regular_expression import Regex


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    """Build a minimal deterministic finite automaton from a regular expression."""
    nfa = Regex(regex).to_epsilon_nfa()
    dfa = nfa.to_deterministic()
    return dfa.minimize()


def graph_to_nfa(
    graph: MultiDiGraph, start_states: set[int], final_states: set[int]
) -> NondeterministicFiniteAutomaton:
    """Build a non-deterministic finite automaton from a graph.

    Each edge becomes a transition labelled with the edge label. When
    ``start_states`` or ``final_states`` is empty, all nodes are treated as
    start or final states respectively.
    """
    nfa = NondeterministicFiniteAutomaton()

    for source, target, label in graph.edges(data="label"):
        if label is not None:
            nfa.add_transition(State(source), Symbol(label), State(target))

    all_nodes = set(graph.nodes)
    if not start_states:
        start_states = all_nodes
    if not final_states:
        final_states = all_nodes

    for state in start_states:
        nfa.add_start_state(State(state))
    for state in final_states:
        nfa.add_final_state(State(state))

    return nfa
