"""Optimized shortest-path comparison solvers.

This module keeps the algorithms identical to the baseline versions but applies
safe implementation-level optimizations to reduce Python overhead.
"""

import heapq
from typing import Optional, Tuple, List

from .graph import Graph

INFINITY = float('inf')


def dijkstra_optimized(graph: Graph, source: int, goal: int) -> Optional[Tuple[float, List[int]]]:
    """Classic Dijkstra's algorithm with low-risk Python optimizations.

    The algorithm is unchanged: it still uses a binary heap, lazy deletion of
    stale queue entries, and predecessor-based path reconstruction. The
    improvements are limited to cheaper local lookups and a small fast path for
    trivial source=goal queries.
    """
    if source == goal:
        return 0.0, [source]

    vertices = graph.vertices
    distances = [INFINITY] * vertices
    predecessors = [None] * vertices
    adjacency = graph.adj

    distances[source] = 0.0

    heappush = heapq.heappush
    heappop = heapq.heappop
    pq = [(0.0, source)]

    while pq:
        current_dist, u = heappop(pq)

        # Skip stale entries produced by lazy deletion.
        if current_dist > distances[u]:
            continue

        if u == goal:
            break

        for edge in adjacency[u]:
            new_dist = current_dist + edge.weight
            v = edge.to

            if new_dist < distances[v]:
                distances[v] = new_dist
                predecessors[v] = u
                heappush(pq, (new_dist, v))

    if distances[goal] == INFINITY:
        return None

    path: List[int] = []
    curr: Optional[int] = goal

    while curr is not None:
        path.append(curr)
        if curr == source:
            break
        curr = predecessors[curr]

    if not path or path[-1] != source:
        return None

    return distances[goal], path[::-1]