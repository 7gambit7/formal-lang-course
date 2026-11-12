import networkx as nx
import pytest
from pyformlang.cfg import CFG

from project.gll_cfpq import gll_based_cfpq
from project.hellings_cfpq import hellings_based_cfpq
from project.tensor_cfpq import cfg_to_rsm, ebnf_to_rsm, tensor_based_cfpq

# 0 -a-> 1 -a-> 2 -a-> 0 and 0 -b-> 3 -b-> 0
TWO_CYCLES = "0a1 1a2 2a0 0b3 3b0"
# the same plus a shortcut 1 -b-> 3 -a-> 2
TWO_CYCLES_LINKED = TWO_CYCLES + " 1b3 3a2"


def graph_of(spec: str) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for edge in spec.split():
        g.add_edge(int(edge[0]), int(edge[2]), label=edge[1])
    return g


@pytest.mark.parametrize(
    "text, starts, finals, expected",
    [
        ("S -> a S b b | a b b", {0}, {0, 3}, {(0, 0)}),
        ("S -> $", None, None, {(n, n) for n in range(4)}),
        ("S -> a", {1}, None, {(1, 2)}),
        ("S -> a", {1}, {0}, set()),
        ("S -> b S | a", {0, 3}, {1}, {(0, 1), (3, 1)}),
        ("S -> c", None, None, set()),
        ("S -> T b\nT -> a T", None, None, set()),
    ],
)
def test_two_cycles_answers(text, starts, finals, expected) -> None:
    rsm = cfg_to_rsm(CFG.from_text(text))
    assert gll_based_cfpq(rsm, graph_of(TWO_CYCLES), starts, finals) == expected


@pytest.mark.parametrize("text", ["S -> S a | S S | $", "S -> S S a | a"])
def test_left_recursive_grammar_terminates(text: str) -> None:
    rsm = cfg_to_rsm(CFG.from_text(text))
    expected = hellings_based_cfpq(CFG.from_text(text), graph_of(TWO_CYCLES), {0})
    assert gll_based_cfpq(rsm, graph_of(TWO_CYCLES), {0}, None) == expected


def test_box_reused_from_many_call_sites() -> None:
    # X is entered at 0 by S and at 1 by X itself: a^n b^n on a line
    rsm = cfg_to_rsm(CFG.from_text("S -> X\nX -> a X b | a b"))
    graph = graph_of("0a1 1a2 2b3 3b4")
    assert gll_based_cfpq(rsm, graph) == {(0, 4), (1, 3)}


def test_grammar_without_rules() -> None:
    assert gll_based_cfpq(cfg_to_rsm(CFG()), graph_of(TWO_CYCLES)) == set()


def test_initial_label_is_respected() -> None:
    gram = CFG.from_text("P -> a Q | $\nQ -> b P", start_symbol="P")
    graph = graph_of("0a1 1b2 2a3 3b0")
    assert gll_based_cfpq(cfg_to_rsm(gram), graph, {0}, None) == {(0, 0), (0, 2)}


@pytest.mark.parametrize(
    "text",
    [
        "S -> a S b | $",
        "S -> S S | a | b",
        "S -> A B\nA -> a A | $\nB -> b B | b",
        "S -> a S a | b S b | $",
        "S -> S a S b | $",
    ],
)
def test_same_as_hellings_and_tensor(text: str) -> None:
    gram = CFG.from_text(text)
    graph = graph_of(TWO_CYCLES_LINKED)
    rsm = cfg_to_rsm(gram)
    expected = hellings_based_cfpq(gram, graph)
    assert gll_based_cfpq(rsm, graph) == expected
    assert tensor_based_cfpq(rsm, graph) == expected


@pytest.mark.parametrize(
    "ebnf",
    ["S -> (a S b)*", "S -> a* b*", "S -> A b*\nA -> a A a | $"],
)
def test_ebnf_rsm_same_as_tensor(ebnf: str) -> None:
    graph = graph_of(TWO_CYCLES_LINKED)
    rsm = ebnf_to_rsm(ebnf)
    assert gll_based_cfpq(rsm, graph) == tensor_based_cfpq(rsm, graph)
