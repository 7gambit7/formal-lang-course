import networkx as nx
import pytest
from pyformlang.cfg import CFG

from project.hellings_cfpq import hellings_based_cfpq
from project.matrix_cfpq import matrix_based_cfpq

# 0 -a-> 1 -a-> 2 -a-> 0 and 0 -b-> 3 -b-> 0
TWO_CYCLES = "0a1 1a2 2a0 0b3 3b0"


def graph_of(spec: str) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for edge in spec.split():
        g.add_edge(int(edge[0]), int(edge[2]), label=edge[1])
    return g


@pytest.mark.parametrize(
    "text, starts, finals, expected",
    [
        # a^n b^2n: only n divisible by 3 returns to 0, an even b-run stays there
        ("S -> a S b b | a b b", {0}, {0, 3}, {(0, 0)}),
        ("S -> $", None, None, {(n, n) for n in range(4)}),
        ("S -> a", {1}, None, {(1, 2)}),
        ("S -> a", {1}, {0}, set()),
        ("S -> b S | a", {0, 3}, {1}, {(0, 1), (3, 1)}),
        ("S -> c", None, None, set()),
    ],
)
def test_matrix_on_two_cycles(text, starts, finals, expected) -> None:
    got = matrix_based_cfpq(CFG.from_text(text), graph_of(TWO_CYCLES), starts, finals)
    assert got == expected


@pytest.mark.parametrize(
    "text",
    [
        "S -> a S b | $",
        "S -> S S | a | b",
        "S -> A B\nA -> a A | $\nB -> b B | b",
        "S -> a S a | b S b | $",
    ],
)
def test_matrix_agrees_with_hellings(text: str) -> None:
    gram = CFG.from_text(text)
    graph = graph_of(TWO_CYCLES + " 1b3 3a2")
    assert matrix_based_cfpq(gram, graph) == hellings_based_cfpq(gram, graph)
