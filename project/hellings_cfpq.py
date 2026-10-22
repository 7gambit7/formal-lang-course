import itertools
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator

import networkx as nx
from pyformlang.cfg import CFG, Epsilon, Production, Terminal, Variable
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton

from project.finite_automata import graph_to_nfa


def _as_variable(symbol: Variable | Terminal) -> Variable:
    return Variable(f"[{symbol.value}]") if isinstance(symbol, Terminal) else symbol


def _chain(lhs: Variable, symbols: list[Variable]) -> Iterator[Production]:
    if len(symbols) <= 2:
        yield Production(lhs, symbols)
        return
    suffix = Variable("<" + " ".join(str(s.value) for s in symbols[1:]) + ">")
    yield Production(lhs, [symbols[0], suffix])
    yield from _chain(suffix, symbols[1:])


def _weaken(rule: Production) -> Iterator[Production]:
    symbols = [s for s in rule.body if not isinstance(s, Epsilon)]
    if len(symbols) < 2:
        yield Production(rule.head, symbols)
        return
    terminals = {s for s in symbols if isinstance(s, Terminal)}
    yield from (Production(_as_variable(t), [t]) for t in terminals)
    yield from _chain(rule.head, [_as_variable(s) for s in symbols])


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    without_units = cfg.eliminate_unit_productions()
    return CFG(
        start_symbol=cfg.start_symbol,
        productions=set(
            itertools.chain.from_iterable(map(_weaken, without_units.productions))
        ),
    )


Triple = tuple[Variable, int, int]


class _HellingsSolver:
    def __init__(self, wcnf: CFG) -> None:
        self._eps_vars: set[Variable] = set()
        self._from_symbol: dict[str, set[Variable]] = defaultdict(set)
        self._glue_right = defaultdict(lambda: defaultdict(set))
        self._glue_left = defaultdict(lambda: defaultdict(set))
        for rule in wcnf.productions:
            self._register(rule.head, rule.body)
        self._ends = defaultdict(lambda: defaultdict(set))
        self._begins = defaultdict(lambda: defaultdict(set))
        self._todo: deque[Triple] = deque()

    def _register(self, lhs: Variable, rhs: list) -> None:
        match rhs:
            case [] | [Epsilon()]:
                self._eps_vars.add(lhs)
            case [Terminal() as letter]:
                self._from_symbol[letter.value].add(lhs)
            case [first, second]:
                self._glue_right[first][second].add(lhs)
                self._glue_left[second][first].add(lhs)

    def _absorb(self, candidates: Iterable[Triple]) -> None:
        fresh = {t for t in candidates if t[2] not in self._ends[t[0]][t[1]]}
        for nt, a, b in fresh:
            self._ends[nt][a].add(b)
            self._begins[nt][b].add(a)
        self._todo.extend(fresh)

    def _neighbours(self, nt: Variable, a: int, b: int) -> Iterator[Triple]:
        for other, heads in self._glue_right[nt].items():
            yield from itertools.product(heads, (a,), self._ends[other].get(b, ()))
        for other, heads in self._glue_left[nt].items():
            yield from itertools.product(heads, self._begins[other].get(a, ()), (b,))

    def solve(
        self, nfa: NondeterministicFiniteAutomaton
    ) -> dict[Variable, dict[int, set[int]]]:
        moves = nfa.to_dict()
        self._absorb(
            itertools.chain(
                (
                    (nt, a.value, b.value)
                    for a, by_symbol in moves.items()
                    for sym, targets in by_symbol.items()
                    for b in targets
                    for nt in self._from_symbol.get(sym.value, ())
                ),
                ((nt, x.value, x.value) for x in nfa.states for nt in self._eps_vars),
            )
        )
        while self._todo:
            self._absorb(self._neighbours(*self._todo.popleft()))
        return self._ends


def hellings_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    nfa = graph_to_nfa(graph, start_nodes, final_nodes)
    reach = _HellingsSolver(cfg_to_weak_normal_form(cfg)).solve(nfa)
    sources = {s.value for s in nfa.start_states}
    sinks = {s.value for s in nfa.final_states}
    return {
        (a, b)
        for a, ends in reach[cfg.start_symbol].items()
        if a in sources
        for b in ends & sinks
    }
