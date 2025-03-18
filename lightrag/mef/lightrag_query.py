import logging
import traceback
from typing import Dict
from dotenv import load_dotenv
from ai_models.model import Executable, Initializable
from lightrag import QueryParam, LightRAG
from lightrag.api.rag_instance_manager import RAGInstanceManager
from lightrag.api.utils_api import(
    parse_args,
    extract_token_value,
    get_lightrag_token_credential,
    initialize_rag_with_header,
    wait_for_storage_initialization
)
# TODO: following imports are a temporary workaround for long load time
# TODO: on graph db related module especially networkx and graspologic.
# TODO: This expected to be fix after migrate to Azure Database server for PostgreSQL.
# TODO: the workaround is to import the module here so LightRAG server will
# TODO: take longer to start up but the initialization of the storage will be faster.
import lightrag.kg.json_doc_status_impl
import lightrag.kg.json_kv_impl
import lightrag.kg.nano_vector_db_impl
import lightrag.kg.networkx_impl
import lightrag.llm.azure_openai
import lightrag.lightrag
import lightrag.az_token_credential
import lightrag.base


class LightRAGQuery:

    def __init__(self):
        """Inherited init."""
        Initializable.__init__(self)
        Executable.__init__(self)
        self.rag_instance_manager = None
        self.arguments = None

    def initialize(self, *args, **kwargs):
        """Load env variables and initialize RAGInstanceManager."""

        try:
            logging.info(f"RUNNING initialize with args: {args}")
            load_dotenv(override=True)
            self.arguments = parse_args()
            self.rag_instance_manager = RAGInstanceManager(args=self.arguments)

        except Exception as ex:
            logging.error(f"Error in LightRAGBuilder initialization: {str(ex)}")
            raise ex

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
                self.rag_instance_manager,
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

