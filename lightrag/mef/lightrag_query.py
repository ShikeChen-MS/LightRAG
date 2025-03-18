import logging
import traceback
from typing import Dict
from dotenv import load_dotenv

from lightrag import QueryParam, LightRAG
from lightrag.api.rag_instance_manager import RAGInstanceManager
from lightrag.api.utils_api import parse_args, extract_token_value, get_lightrag_token_credential, \
    initialize_rag_with_header, wait_for_storage_initialization


class LightRAGQuery:

    async def execute(
        self,
        query: str,
        mode: str,
        history_turns: int,
        storage_account_url: str,
        storage_container_name: str,
        storage_token_expiry: str,
        ai_access_token: str,
        storage_access_token: str
    )-> Dict[str, str]:
        """
        Handle a POST request at the /query endpoint to process user queries using RAG capabilities.
        """
        try:
            load_dotenv(override=True)
            args = parse_args()
            rag_instance_manager = RAGInstanceManager(args=args)
            ai_access_token = extract_token_value(
                ai_access_token, "Azure-AI-Access-Token"
            )
            storage_access_token = extract_token_value(
                storage_access_token, "Storage_Access_Token"
            )
            lightrag_token = get_lightrag_token_credential(
                storage_access_token, storage_token_expiry
            )
            if mode not in ["local", "global", "hybrid", "naive", "mix"]:
                raise ValueError("Invalid mode. Choose from 'local', 'global', 'hybrid', 'naive', or 'mix'.")
            param = QueryParam(
                mode= mode,
                only_need_context = True,
                only_need_prompt = False,
                response_type = "Multiple Paragraphs",
                stream = False,
                top_k = 60,
                max_token_for_text_unit = 4000,
                max_token_for_global_context = 4000,
                max_token_for_local_context = 4000,
                hl_keywords = [],
                ll_keywords = [],
                conversation_history = [],
                history_turns = history_turns
            )
            rag: LightRAG = await initialize_rag_with_header(
                rag_instance_manager,
                storage_account_url,
                storage_container_name,
                None,
                storage_access_token,
                storage_token_expiry,
            )
            await wait_for_storage_initialization(
                rag,
                get_lightrag_token_credential(
                    storage_access_token, storage_token_expiry
                ),
            )
            response = await rag.aquery(
                query,
                ai_access_token,
                storage_account_url,
                storage_container_name,
                lightrag_token,
                param=param,
            )

            # If response is a string (e.g. cache hit), return directly
            if isinstance(response, str):
                return {"response": response}

            if isinstance(response, dict):
                return response
            else:
                return {"response": str(response)}
        except Exception as e:
            logging.error(f"Error /query: {str(e)}")
            logging.error(traceback.format_exc())
            raise RuntimeError(f"Error occured during query: detail: {str(e)}")

