import os
from dataclasses import dataclass
from typing import Any, final
import numpy as np
from azure.storage.blob import BlobServiceClient, BlobLeaseClient
from ..az_token_credential import LightRagTokenCredential
from ..types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from ..utils import (
    logger,
)
from ..base import (
    BaseGraphStorage,
)
import networkx as nx
from graspologic import embed


@final
@dataclass
class NetworkXStorage(BaseGraphStorage):
    def __init__(self, global_config: dict[str, Any], namespace: str, **kwargs: Any):
        self._graph = None
        self.namespace = namespace
        self.global_config = global_config

    @staticmethod
    def load_nx_graph(file_name) -> nx.Graph:
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        logger.info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        nx.write_graphml(graph, file_name)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        """
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

        fixed_graph.add_nodes_from(sorted_nodes)
        edges = list(graph.edges(data=True))

        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

    async def initialize(
        self,
        storage_account_url: str,
        storage_container_name: str,
        access_token: LightRagTokenCredential,
    ) -> None:
        lease = None
        blob_lease = None
        try:
            blob_client = BlobServiceClient(
                account_url=storage_account_url, credential=access_token
            )
            container_client = blob_client.get_container_client(storage_container_name)
            container_client.get_container_properties()  # this is to check if the container exists and authentication is valid
            lease: BlobLeaseClient = container_client.acquire_lease()
            blob_list = container_client.list_blob_names()
            blob_name = f"{self.global_config["working_dir"]}/data/graph_{self.namespace}.graphml"
            if not blob_name in blob_list:
                logger.info(f"Creating new graph_{self.namespace}.graphml")
                self._graph = nx.Graph()
                linefeed = chr(10)
                content = linefeed.join(nx.generate_graphml(self._graph))
                blob_client = container_client.get_blob_client(blob_name)
                # reach here means the file does not exist, so with overwrite=False
                # the operation should still succeed.
                blob_client.upload_blob(content, overwrite=False)
                return
            blob_client = container_client.get_blob_client(blob_name)
            blob_lease = blob_client.acquire_lease()
            content = (
                blob_client.download_blob(lease=blob_lease).readall()
            )
            blob_lease.release()
            blob_lease = None
            lease.release() # early release to avoid blocking
            lease = None
            content_str = content.decode("utf-8")
            preloaded_graph = nx.parse_graphml(content_str)
            self._graph = preloaded_graph
        except Exception as e:
            logger.warning(f"Failed to load graph from Azure Blob Storage: {e}")
            raise
        finally:
            if lease:
                lease.release()
            if blob_lease:
                blob_lease.release()
            self._node_embed_algorithms = {
                "node2vec": self._node2vec_embed,
            }

    async def index_done_callback(
        self,
        storage_account_url: str,
        storage_container_name: str,
        access_token: LightRagTokenCredential,
    ) -> None:
        lease = None
        blob_lease = None
        try:
            blob_client = BlobServiceClient(
                account_url=storage_account_url, credential=access_token
            )
            container_client = blob_client.get_container_client(storage_container_name)
            # this is to check if the container exists and authentication is valid
            container_client.get_container_properties()
            # to protect file integrity and ensure complete upload
            # acquire lease on the container to prevent any other ops
            lease: BlobLeaseClient = container_client.acquire_lease()
            blob_name = (
                f"{self.global_config["working_dir"]}/data/graph_{self.namespace}.graphml"
            )
            blob_client = container_client.get_blob_client(blob_name)
            blob_lease = blob_client.acquire_lease()
            linefeed = chr(10)
            content = linefeed.join(nx.generate_graphml(self._graph))
            blob_client.upload_blob(content, lease=blob_lease, overwrite=True)
            blob_lease.release()
            blob_lease = None
            lease.release()
            lease = None
        except Exception as e:
            logger.warning(f"Failed to save graph to Azure Blob Storage: {e}")
            raise
        finally:
            if lease:
                lease.release()
            if blob_lease:
                blob_lease.release()

    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        return self._graph.nodes.get(node_id)

    async def node_degree(self, node_id: str) -> int:
        return self._graph.degree(node_id)

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return self._graph.degree(src_id) + self._graph.degree(tgt_id)

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        return self._graph.edges.get((source_node_id, target_node_id))

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        self._graph.add_node(node_id, **node_data)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def delete_node(self, node_id: str) -> None:
        if self._graph.has_node(node_id):
            self._graph.remove_node(node_id)
            logger.info(f"Node {node_id} deleted from the graph.")
        else:
            logger.warning(f"Node {node_id} not found in the graph for deletion.")

    async def embed_nodes(
        self, algorithm: str
    ) -> tuple[np.ndarray[Any, Any], list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    # @TODO: NOT USED
    async def _node2vec_embed(self):
        embeddings, nodes = embed.node2vec_embed(
            self._graph,
            **self.global_config["node2vec_params"],
        )

        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids

    def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes

        Args:
            nodes: List of node IDs to be deleted
        """
        for node in nodes:
            if self._graph.has_node(node):
                self._graph.remove_node(node)

    def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges

        Args:
            edges: List of edges to be deleted, each edge is a (source, target) tuple
        """
        for source, target in edges:
            if self._graph.has_edge(source, target):
                self._graph.remove_edge(source, target)

    async def get_all_labels(self) -> list[str]:
        """
        Get all node labels in the graph
        Returns:
            [label1, label2, ...]  # Alphabetically sorted label list
        """
        labels = set()
        for node in self._graph.nodes():
            labels.add(str(node))  # Add node id as a label

        # Return sorted list
        return sorted(list(labels))

    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 5
    ) -> KnowledgeGraph:
        """
        Get complete connected subgraph for specified node (including the starting node itself)

        Args:
            node_label: Label of the starting node
            max_depth: Maximum depth of the subgraph

        Returns:
            KnowledgeGraph object containing nodes and edges
        """
        result = KnowledgeGraph()
        seen_nodes = set()
        seen_edges = set()

        # Handle special case for "*" label
        if node_label == "*":
            # For "*", return the entire graph including all nodes and edges
            subgraph = (
                self._graph.copy()
            )  # Create a copy to avoid modifying the original graph
        else:
            # Find nodes with matching node id (partial match)
            nodes_to_explore = []
            for n, attr in self._graph.nodes(data=True):
                if node_label in str(n):  # Use partial matching
                    nodes_to_explore.append(n)

            if not nodes_to_explore:
                logger.warning(f"No nodes found with label {node_label}")
                return result

            # Get subgraph using ego_graph
            subgraph = nx.ego_graph(self._graph, nodes_to_explore[0], radius=max_depth)

        # Check if number of nodes exceeds max_graph_nodes
        max_graph_nodes = 500
        if len(subgraph.nodes()) > max_graph_nodes:
            origin_nodes = len(subgraph.nodes())
            node_degrees = dict(subgraph.degree())
            top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[
                :max_graph_nodes
            ]
            top_node_ids = [node[0] for node in top_nodes]
            # Create new subgraph with only top nodes
            subgraph = subgraph.subgraph(top_node_ids)
            logger.info(
                f"Reduced graph from {origin_nodes} nodes to {max_graph_nodes} nodes (depth={max_depth})"
            )

        # Add nodes to result
        for node in subgraph.nodes():
            if str(node) in seen_nodes:
                continue

            node_data = dict(subgraph.nodes[node])
            # Get entity_type as labels
            labels = []
            if "entity_type" in node_data:
                if isinstance(node_data["entity_type"], list):
                    labels.extend(node_data["entity_type"])
                else:
                    labels.append(node_data["entity_type"])

            # Create node with properties
            node_properties = {k: v for k, v in node_data.items()}

            result.nodes.append(
                KnowledgeGraphNode(
                    id=str(node), labels=[str(node)], properties=node_properties
                )
            )
            seen_nodes.add(str(node))

        # Add edges to result
        for edge in subgraph.edges():
            source, target = edge
            edge_id = f"{source}-{target}"
            if edge_id in seen_edges:
                continue

            edge_data = dict(subgraph.edges[edge])

            # Create edge with complete information
            result.edges.append(
                KnowledgeGraphEdge(
                    id=edge_id,
                    type="DIRECTED",
                    source=str(source),
                    target=str(target),
                    properties=edge_data,
                )
            )
            seen_edges.add(edge_id)

        # logger.info(result.edges)

        logger.info(
            f"Subgraph query successful | Node count: {len(result.nodes)} | Edge count: {len(result.edges)}"
        )
        return result
