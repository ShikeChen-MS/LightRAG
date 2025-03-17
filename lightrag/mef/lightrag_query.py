import json
import logging
import logging.config
import os
import traceback
from typing import Dict, Any
from .. import LightRAG
from ..api.utils_api import extract_token_value
from ..rag_instance_manager import RAGInstanceManager
from ..base import QueryParam


class LightragQuery:

    def __init__(self):
        current_dir = os.path.dirname(__file__)
        logging_config_path = os.path.join(current_dir, "..", "logging_config.json")
        with open(logging_config_path, "r") as f:
            logging_configs = json.load(f)
            logging.config.dictConfig(logging_configs)

    async def query_text(
        self,
        request: Dict[str, Any],
        ai_access_token: str,
        rag_configs: Dict[str, Any],
        db_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Handle a POST request at the /query endpoint to process user queries using RAG capabilities.
        """
        rag: LightRAG | None = None
        res = {}
        try:
            rag_instance_manager = RAGInstanceManager(args=rag_configs)
            ai_access_token = extract_token_value(
                ai_access_token, "Azure-AI-Access-Token"
            )
            storage_access_token = extract_token_value(
                db_params["db_access_token"], "DB_Access_Token"
            )
            rag = await rag_instance_manager.get_rag_instance(
                db_url=db_params["db_url"],
                db_name=db_params["db_name"],
                db_user_name=db_params["db_user_name"],
                db_access_token=storage_access_token,
            )

            param = QueryParam(
                mode=request["mode"],
                only_need_context=request["only_need_context"],
                only_need_prompt=request["only_need_prompt"],
                response_type=request["response_type"],
                stream=False,
                top_k=60 if not ("top_k" in rag_configs) else rag_configs["top_k"],
                max_token_for_text_unit=rag_configs["max_token_text_chunk"],
                max_token_for_global_context=rag_configs["max_token_relation_desc"],
                max_token_for_local_context=rag_configs["max_token_entity_desc"],
                hl_keywords=request["hl_keywords"],
                ll_keywords=request["ll_keywords"],
                conversation_history=request["conversation_history"],
                history_turns=request["history_turns"],
            )
            response = await rag.aquery(
                request["query"],
                ai_access_token,
                param=param,
            )
            # If response is a string (e.g. cache hit), return directly
            if isinstance(response, str):
                res = {"response": response}
            if isinstance(response, dict):
                res = response
            else:
                res = {"response": response}
        except Exception as e:
            res = {
                "error": f"error: {str(e)}\n, traceback: {str(traceback.format_exc())}"
            }
        finally:
            if rag:
                await rag.finalize_storages()
            return res
