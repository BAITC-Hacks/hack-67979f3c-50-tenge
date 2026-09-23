"""Полный направленный граф, включая узлы без наблюдаемых переводов."""

import networkx as nx


def build_graph(nodes, edges) -> nx.DiGraph:
    """Build after validate_inputs; stable insertion order makes runs repeatable."""
    if nodes["gid"].duplicated().any() or edges.duplicated(["src", "dst"]).any():
        raise ValueError("Граф: gid и пары src, dst должны быть уникальны")
    if not (set(edges["src"]) | set(edges["dst"])) <= set(nodes["gid"]):
        raise ValueError("Граф: концы рёбер отсутствуют в nodes.gid")
    graph = nx.DiGraph()
    for row in nodes.sort_values("gid").itertuples(index=False):
        graph.add_node(int(row.gid), depth=int(row.depth), is_seed=bool(row.is_seed))
    for row in edges.sort_values(["src", "dst"]).itertuples(index=False):
        graph.add_edge(int(row.src), int(row.dst), sum_kzt=float(row.sum_kzt),
                       n_tx=int(row.n_tx), depth=int(row.depth))
    return graph
