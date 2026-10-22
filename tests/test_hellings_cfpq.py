import networkx as nx
import pytest
from pyformlang.cfg import CFG, Epsilon, Production, Terminal

from project.hellings_cfpq import cfg_to_weak_normal_form, hellings_based_cfpq

# 0 -a-> 1 -a-> 2 -a-> 0 and 0 -b-> 3 -b-> 0
TWO_CYCLES = "0a1 1a2 2a0 0b3 3b0"


def graph_of(spec: str) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for edge in spec.split():
        g.add_edge(int(edge[0]), int(edge[2]), label=edge[1])
    return g


def shape(rule: Production) -> str | None:
    match rule.body:
        case [] | [Epsilon()]:
            return "eps"
        case [Terminal()]:
            return "letter"
        case [first, second] if not {type(first), type(second)} & {Terminal}:
            return "pair"
    return None


@pytest.mark.parametrize(
    "text, samples",
    [
        ("S -> S p T | T\nT -> n | l S r", ["n", "npn", "lnr", "lnpnr", "np", ""]),
        ("S -> A B C\nA -> a | $\nB -> b B | $\nC -> c", ["c", "abbc", "ac", "ab"]),
        ("S -> k l m n o", ["klmno", "klmn", ""]),
        ("S -> $", ["", "k"]),
    ],
)
def test_weak_form_keeps_language(text: str, samples: list[str]) -> None:
    gram = CFG.from_text(text)
    wcnf = cfg_to_weak_normal_form(gram)
    assert None not in {shape(rule) for rule in wcnf.productions}
    assert {s: wcnf.contains(s) for s in samples} == {
        s: gram.contains(s) for s in samples
    }


@pytest.mark.parametrize(
    "text, starts, finals, expected",
    [
        # a^n b^2n: only n divisible by 3 returns to 0, an even b-run stays there
        ("S -> a S b b | a b b", {0}, {0, 3}, {(0, 0)}),
        ("S -> $", None, None, {(n, n) for n in range(4)}),
        ("S -> a", {1}, None, {(1, 2)}),
        ("S -> a", {1}, {0}, set()),
        ("S -> b S | a", {0, 3}, {1}, {(0, 1), (3, 1)}),
    ],
)
def test_hellings_on_two_cycles(text, starts, finals, expected) -> None:
    got = hellings_based_cfpq(CFG.from_text(text), graph_of(TWO_CYCLES), starts, finals)
    assert got == expected
