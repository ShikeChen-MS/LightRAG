"""
This module contains all graph-related routes for the LightRAG API.
"""
import json
from typing import Optional
from fastapi import(
    APIRouter,
    Depends,
    Header,
    HTTPException,
)
from fastapi.responses import JSONResponse
from lightrag.base_requestbody import BaseRequest
from ..utils_api import get_api_key_dependency, prepare_rag_instance
from lightrag.utils import extract_token_value
from lightrag.azure_token_handler import (
    AzureToken,
    AzureTokenHandler,
    TokenScope,
)

router = APIRouter(tags=["graph"])


def create_graph_routes(ragmanager, api_key: Optional[str] = None):
    optional_api_key = get_api_key_dependency(api_key)

    @router.get("/graph/label/list", dependencies=[Depends(optional_api_key)])
    async def get_graph_labels(
            base_request: BaseRequest,
            user_access_token: str = Header(None, alias="Azure_Ad_Token"),
            X_Affinity_Token: str = Header(None, alias="X-Affinity-Token")
    ):
        """Get all graph labels"""
        # In case of networkx implementation, graph db (as a file) has already been loaded
        # into memory, there's no actual authentication needed. Therefore, we'll be trying to acquire
        # token, if succeeded, we assume the user has access to the graph db, but the token is not used
        try:
            token = extract_token_value(user_access_token)
            access_token: AzureToken = AzureTokenHandler.acquire_token_by_user_token(token, TokenScope.CognitiveServices)
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
        try:
            rag = prepare_rag_instance(ragmanager, base_request, X_Affinity_Token)
            res =  await rag.get_graph_labels()
            json.dumps(res, indent=2)
            return JSONResponse(content=res,headers={"X-Affinity-Token": rag.affinity_token})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


    @router.get("/graphs", dependencies=[Depends(optional_api_key)])
    async def get_knowledge_graph(
            base_request: BaseRequest,
            label: str,
            max_depth: int = 3,
            user_access_token: str = Header(None, alias="Azure_Ad_Token"),
            X_Affinity_Token: str = Header(None, alias="X-Affinity-Token")
    ):
        """Get knowledge graph for a specific label"""
        try:
            token = extract_token_value(user_access_token)
            access_token: AzureToken = AzureTokenHandler.acquire_token_by_user_token(token, TokenScope.CognitiveServices)
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
        try:
            rag = prepare_rag_instance(ragmanager, base_request, X_Affinity_Token)
            res = await rag.get_knowledge_graph(node_label=label, max_depth=max_depth)
            json.dumps(res, indent=2)
            return JSONResponse(content=res,headers={"X-Affinity-Token": rag.affinity_token})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router
