import networkx as nx
import pytest
from pyformlang.cfg import CFG
from pyformlang.rsa import RecursiveAutomaton

from project.hellings_cfpq import hellings_based_cfpq
from project.tensor_cfpq import cfg_to_rsm, ebnf_to_rsm, tensor_based_cfpq

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
        ("S -> A\nA -> B\nB -> A", None, None, set()),
    ],
)
def test_tensor_on_two_cycles(text, starts, finals, expected) -> None:
    rsm = cfg_to_rsm(CFG.from_text(text))
    assert tensor_based_cfpq(rsm, graph_of(TWO_CYCLES), starts, finals) == expected


def test_cfg_to_rsm_keeps_start_symbol() -> None:
    gram = CFG.from_text("A -> a B | $\nB -> b A", start_symbol="A")
    rsm = cfg_to_rsm(gram)
    assert rsm.initial_label.value == "A"
    assert {label.value for label in rsm.labels} == {"A", "B"}
    graph = graph_of("0a1 1b2 2a3 3b0")
    assert tensor_based_cfpq(rsm, graph, {0}, None) == {(0, 0), (0, 2)}


def test_multichar_variables() -> None:
    gram = CFG.from_text("S -> S1 S2\nS1 -> a S1 | $\nS2 -> b S2 | $")
    assert tensor_based_cfpq(cfg_to_rsm(gram), graph_of(TWO_CYCLES), {0}, None) == {
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
    }


@pytest.mark.parametrize(
    "ebnf, text",
    [
        ("S -> (a S b)*", "S -> a S b S | $"),
        ("S -> a* b*", "S -> A B\nA -> a A | $\nB -> b B | $"),
        ("S -> (a | b) (S | $)\n", "S -> a S | b S | a | b"),
        ("S -> A b*\nA -> a A a | $", "S -> A B\nA -> a A a | $\nB -> b B | $"),
    ],
)
def test_ebnf_matches_cfg(ebnf: str, text: str) -> None:
    graph = graph_of(TWO_CYCLES + " 1b3 3a2")
    expected = hellings_based_cfpq(CFG.from_text(text), graph)
    assert tensor_based_cfpq(ebnf_to_rsm(ebnf), graph) == expected
    assert tensor_based_cfpq(cfg_to_rsm(CFG.from_text(text)), graph) == expected


@pytest.mark.parametrize(
    "ebnf",
    ["S -> (a | b)* c", "S -> a S b | $\n", "S -> A B*\nA -> a\nB -> b A"],
)
def test_ebnf_to_rsm_matches_pyformlang(ebnf: str) -> None:
    assert ebnf_to_rsm(ebnf) == RecursiveAutomaton.from_text(ebnf)


def test_ebnf_rules_split_over_lines() -> None:
    rsm = ebnf_to_rsm("S -> a S\nS -> b\nS ->")
    assert tensor_based_cfpq(rsm, graph_of("0a1 1b2"), {0}, None) == {
        (0, 0),
        (0, 1),
        (0, 2),
    }


def test_ebnf_without_s_starts_from_first_rule() -> None:
    rsm = ebnf_to_rsm("A -> a B\nB -> b")
    assert rsm.initial_label.value == "A"
    assert tensor_based_cfpq(rsm, graph_of("0a1 1b2")) == {(0, 2)}


def test_empty_cfg() -> None:
    assert tensor_based_cfpq(cfg_to_rsm(CFG()), graph_of(TWO_CYCLES)) == set()


@pytest.mark.parametrize(
    "text",
    [
        "S -> a S b | $",
        "S -> S S | a | b",
        "S -> A B\nA -> a A | $\nB -> b B | b",
        "S -> a S a | b S b | $",
    ],
)
def test_tensor_agrees_with_hellings(text: str) -> None:
    gram = CFG.from_text(text)
    graph = graph_of(TWO_CYCLES + " 1b3 3a2")
    assert tensor_based_cfpq(cfg_to_rsm(gram), graph) == hellings_based_cfpq(
        gram, graph
    )
