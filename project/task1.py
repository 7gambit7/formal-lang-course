from dataclasses import dataclass
from pathlib import Path

import cfpq_data
import networkx as nx
from networkx.drawing.nx_pydot import to_pydot


@dataclass
class GraphInfo:
    number_of_nodes: int
    number_of_edges: int
    labels: set[str]


def load_graph_by_name(name: str) -> nx.MultiDiGraph:
    """Download a graph from the CFPQ_Data dataset by its name"""
    graph_path = cfpq_data.download(name)
    return cfpq_data.graph_from_csv(graph_path)


def get_graph_info(graph: nx.MultiDiGraph) -> GraphInfo:
    """Collect the number of vertices, edges and distinct edge labels of a graph"""
    labels = {label for _, _, label in graph.edges(data="label") if label is not None}
    return GraphInfo(
        number_of_nodes=graph.number_of_nodes(),
        number_of_edges=graph.number_of_edges(),
        labels=labels,
    )


def get_graph_info_by_name(name: str) -> GraphInfo:
    """Return the number of vertices, edges and distinct edge labels for the named graph"""
    return get_graph_info(load_graph_by_name(name))


def build_two_cycles_graph(
    first_cycle_nodes: int,
    second_cycle_nodes: int,
    labels: tuple[str, str],
) -> nx.MultiDiGraph:
    """Build a graph consisting of two cycles connected through a common node"""
    return cfpq_data.labeled_two_cycles_graph(
        first_cycle_nodes, second_cycle_nodes, labels=labels
    )


def save_graph_to_dot(graph: nx.MultiDiGraph, path: str | Path) -> None:
    """Save a graph to the given file in the DOT format using pydot"""
    to_pydot(graph).write_raw(str(path))


def build_and_save_two_cycles_graph(
    first_cycle_nodes: int,
    second_cycle_nodes: int,
    labels: tuple[str, str],
    path: str | Path,
) -> None:
    """Build a two-cycles labeled graph and save it to the file in the DOT format."""
    graph = build_two_cycles_graph(first_cycle_nodes, second_cycle_nodes, labels)
    save_graph_to_dot(graph, path)
