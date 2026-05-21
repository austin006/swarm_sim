"""Communication graph utilities for swarm controllers."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass
class CommunicationNetwork:
    """Dynamic communication topology backed by NetworkX."""

    num_agents: int
    topology: str = "all-to-all"

    def __post_init__(self) -> None:
        if self.num_agents < 1:
            raise ValueError("num_agents must be at least 1")
        self.graph = self._build_graph(self.topology)

    def set_topology(self, topology: str) -> None:
        """Switch topology and rebuild graph matrices on demand."""

        self.topology = topology
        self.graph = self._build_graph(topology)

    @property
    def adjacency_matrix(self) -> np.ndarray:
        return nx.to_numpy_array(self.graph, nodelist=range(self.num_agents), dtype=float)

    @property
    def degree_matrix(self) -> np.ndarray:
        degrees = self.adjacency_matrix.sum(axis=1)
        return np.diag(degrees)

    @property
    def laplacian_matrix(self) -> np.ndarray:
        return self.degree_matrix - self.adjacency_matrix

    def matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return adjacency, degree, and Laplacian matrices."""

        adjacency = self.adjacency_matrix
        degree = np.diag(adjacency.sum(axis=1))
        laplacian = degree - adjacency
        return adjacency, degree, laplacian

    def _build_graph(self, topology: str) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(self.num_agents))

        if topology == "all-to-all":
            graph.add_edges_from(
                (i, j)
                for i in range(self.num_agents)
                for j in range(i + 1, self.num_agents)
            )
        elif topology == "ring":
            if self.num_agents > 2:
                graph.add_edges_from((i, (i + 1) % self.num_agents) for i in range(self.num_agents))
            elif self.num_agents == 2:
                graph.add_edge(0, 1)
        elif topology == "line":
            graph.add_edges_from((i, i + 1) for i in range(self.num_agents - 1))
        else:
            raise ValueError(
                f"Unsupported topology '{topology}'. Use all-to-all, ring, or line."
            )

        return graph

