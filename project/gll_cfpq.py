from collections import defaultdict
from collections.abc import Hashable

import networkx as nx
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton
from pyformlang.rsa import RecursiveAutomaton

from project.finite_automata import graph_to_nfa

# (box, state), (box, graph node where it was called)
BoxPos = tuple[Hashable, Hashable]
CallSite = tuple[Hashable, int]
Task = tuple[BoxPos, int, CallSite]


class _CallFrame:
    """One call of a box from a graph node, shared by all callers."""

    def __init__(self) -> None:
        self.returns_to: set[tuple[BoxPos, CallSite]] = set()
        self.exits: set[int] = set()


class _BoxTable:
    """Start states, final states and transitions of every box."""

    def __init__(self, rsm: RecursiveAutomaton) -> None:
        self.entry_of: dict[Hashable, list[Hashable]] = {}
        self.accepting: dict[Hashable, set[Hashable]] = {}
        self.arcs: dict[BoxPos, list[tuple[Hashable, Hashable]]] = defaultdict(list)
        for name, box in rsm.boxes.items():
            box_id = name.value
            self.entry_of[box_id] = [q.value for q in box.dfa.start_states]
            self.accepting[box_id] = {q.value for q in box.dfa.final_states}
            for src, sym, dst in box.dfa:
                self.arcs[(box_id, src.value)].append((sym.value, dst.value))

    def is_box(self, symbol: Hashable) -> bool:
        return symbol in self.entry_of


class _GllRunner:
    def __init__(
        self, rsm: RecursiveAutomaton, graph_nfa: NondeterministicFiniteAutomaton
    ) -> None:
        self.table = _BoxTable(rsm)
        self.succ: dict[tuple[int, Hashable], set[int]] = defaultdict(set)
        for tail, label, head in graph_nfa:
            self.succ[(tail.value, label.value)].add(head.value)
        self.frames: dict[CallSite, _CallFrame] = {}
        self.visited: set[Task] = set()
        self.worklist: list[Task] = []

    def push(self, pos: BoxPos, node: int, site: CallSite) -> None:
        task = (pos, node, site)
        if task not in self.visited:
            self.visited.add(task)
            self.worklist.append(task)

    def open_frame(self, site: CallSite) -> _CallFrame:
        frame = self.frames.get(site)
        if frame is None:
            frame = self.frames[site] = _CallFrame()
            box_id, node = site
            for q in self.table.entry_of.get(box_id, ()):
                self.push((box_id, q), node, site)
        return frame

    def enter_box(self, site: CallSite, back_to: BoxPos, caller_site: CallSite) -> None:
        frame = self.open_frame(site)
        link = (back_to, caller_site)
        if link in frame.returns_to:
            return
        frame.returns_to.add(link)
        # The box may have already finished somewhere, go back from there too.
        for exit_node in frame.exits:
            self.push(back_to, exit_node, caller_site)

    def leave_box(self, site: CallSite, node: int) -> None:
        frame = self.frames[site]
        if node in frame.exits:
            return
        frame.exits.add(node)
        for back_to, caller_site in frame.returns_to:
            self.push(back_to, node, caller_site)

    def process(self, pos: BoxPos, node: int, site: CallSite) -> None:
        box_id, q = pos
        for sym, q_next in self.table.arcs[pos]:
            if self.table.is_box(sym):
                self.enter_box((sym, node), (box_id, q_next), site)
                continue
            for w in self.succ.get((node, sym), ()):
                self.push((box_id, q_next), w, site)
        if q in self.table.accepting[box_id]:
            self.leave_box(site, node)

    def solve(self, box_id: Hashable, sources: set[int]) -> dict[int, set[int]]:
        for u in sources:
            self.open_frame((box_id, u))
        # Take a task, move along graph edges, call boxes and return from them.
        # Repeat while new tasks appear.
        while self.worklist:
            self.process(*self.worklist.pop())
        return {u: self.frames[(box_id, u)].exits for u in sources}


def gll_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """Find pairs (u, v) where a path from u to v is accepted by the RSM.

    If start or final nodes are not given, all nodes are used.
    """
    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)
    sources = {q.value for q in graph_nfa.start_states}
    targets = {q.value for q in graph_nfa.final_states}

    reached = _GllRunner(rsm, graph_nfa).solve(rsm.initial_label.value, sources)
    return {(u, v) for u, ends in reached.items() for v in ends & targets}
