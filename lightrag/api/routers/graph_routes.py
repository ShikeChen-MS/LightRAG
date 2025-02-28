"""
This module contains all graph-related routes for the LightRAG API.
"""

from typing import Optional
from lightrag.api.base_request import BaseRequest
from fastapi import APIRouter, Depends, Header
from ..utils_api import get_api_key_dependency, initialize_rag

router = APIRouter(tags=["graph"])


def create_graph_routes(rag_instance_manager, api_key: Optional[str] = None):
    optional_api_key = get_api_key_dependency(api_key)

    @router.get("/graph/label/list", dependencies=[Depends(optional_api_key)])
    async def get_graph_labels(
            base_request: BaseRequest,
            storage_access_token: str = Header(None, alias="Storage_Access_Token"),
            X_Affinity_Token: str = Header(None, alias="X-Affinity-Token")
    ):
        """Get all graph labels"""
        rag = initialize_rag(rag_instance_manager, base_request, X_Affinity_Token, storage_access_token)
        return await rag.get_graph_labels()

    @router.get("/graphs", dependencies=[Depends(optional_api_key)])
    async def get_knowledge_graph(
            base_request: BaseRequest,
            label: str,
            storage_access_token: str = Header(None, alias="Storage_Access_Token"),
            X_Affinity_Token: str = Header(None, alias="X-Affinity-Token"),
            max_depth: int = 3

    ):
        """Get knowledge graph for a specific label"""
        rag = initialize_rag(rag_instance_manager, base_request, X_Affinity_Token, storage_access_token)
        return await rag.get_knowledge_graph(node_label=label, max_depth=max_depth)

    return router
