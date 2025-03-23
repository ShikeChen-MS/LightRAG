import asyncio
import threading
from typing import Any
from .lightrag import LightRAG
from .llm.azure_openai import azure_openai_complete_if_cache, azure_openai_embed
from .postgresql import PostgreSQLDB
from .types import GPTKeywordExtractionFormat
from .utils import EmbeddingFunc


class RAGInstanceManager:
    instance = None
    _lock = asyncio.Lock()

    # Singleton pattern
    # preserve one instance of RAGInstanceManager under instance variable
    # following __new__ and __init__ will ensure only one instance is created
    # any subsequent call to initialize will return the same instance
    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            # Only when instance is None, we actually create a new instance
            cls.instance = super(RAGInstanceManager, cls).__new__(cls)
        return cls.instance

    def __init__(self, *args, **kwargs):
        # use _initialized to avoid re-initialization
        # so any attempt to create new instance after first time
        # will not only end up get the first instance created
        # but also the class variables will remain unchanged
        if not hasattr(self, "_initialized"):
            self._initialized = True
            # use kwargs to accept named arguments
            # here we take the args from argparser
            if kwargs["args"] is not None:
                self.args = kwargs["args"]
            else:
                raise ValueError("args cannot be None")

    async def get_rag_instance(
        self, db_url: str, db_name: str, db_user_name: str, db_access_token: str
    ) -> Any:
        # This function and following embedding_func will
        # be passed to LightRAG instance to be used for completion and embedding
        async def azure_openai_model_complete(
            # DO NOT MODIFY SEQUENCE OF ARGUMENTS
            # this function will be wrapped in partial function callable
            # with parameters set in advance, modifying the sequence will break the code
            access_token,
            prompt,
            system_prompt=None,
            history_messages=None,
            keyword_extraction=False,
            **kwargs,
        ) -> str:
            keyword_extraction = kwargs.pop("keyword_extraction", None)
            if keyword_extraction:
                kwargs["response_format"] = GPTKeywordExtractionFormat
            if history_messages is None:
                history_messages = []
            return await azure_openai_complete_if_cache(
                self.args["llm_model"],
                prompt,
                access_token,
                system_prompt=system_prompt,
                history_messages=history_messages,
                base_url=self.args["llm_binding_host"],
                api_version=self.args["llm_api_version"],
                **kwargs,
            )

        embedding_func = EmbeddingFunc(
            embedding_dim=self.args["embedding_dim"],
            max_token_size=self.args["max_embed_tokens"],
            # DO NOT MODIFY SEQUENCE OF ARGUMENTS
            # this function will be wrapped in partial function callable
            # with parameters set in advance, modifying the sequence will break the code
            func=lambda aad_token, texts: azure_openai_embed(
                access_token=aad_token,
                texts=texts,
                model=self.args["embedding_model"],
                base_url=self.args["embedding_binding_host"],
                api_version=self.args["embedding_api_version"],
            ),
        )
        config = {
            "host": db_url,
            "port": 6432,
            "user": db_user_name,
            "password": db_access_token,
            "database": db_name,
        }
        db = PostgreSQLDB(config)
        await db.initdb()
        await db.check_tables()
        rag = LightRAG(
            db=db,
            scan_progress={
                "is_scanning": False,
                "current_file": "",
                "indexed_count": 0,
                "total_files": 0,
                "progress": 0,
            },
            progress_lock=threading.Lock(),
            db_url=db_url,
            db_name=db_name,
            llm_model_func=azure_openai_model_complete,
            chunk_token_size=int(self.args["chunk_size"]),
            chunk_overlap_token_size=int(self.args["chunk_overlap_size"]),
            llm_model_kwargs={
                "timeout": self.args["timeout"],
            },
            llm_model_name=self.args["llm_model"],
            llm_model_max_async=self.args["max_async"],
            llm_model_max_token_size=self.args["max_tokens"],
            embedding_func=embedding_func,
            kv_storage=self.args["kv_storage"],
            graph_storage=self.args["graph_storage"],
            vector_storage=self.args["vector_storage"],
            doc_status_storage=self.args["doc_status_storage"],
            vector_db_storage_cls_kwargs={
                "cosine_better_than_threshold": self.args["cosine_threshold"]
            },
            enable_llm_cache_for_entity_extract=False,  # set to True for debugging to reduce llm fee
            embedding_cache_config={
                "enabled": True,
                "similarity_threshold": 0.95,
                "use_llm_check": False,
            },
            log_level=self.args["log_level"],
            namespace_prefix=self.args["namespace_prefix"],
            max_parallel_insert=self.args["max_parallel_insert"],
            cosine_threshold=self.args["cosine_threshold"],
        )
        await rag.create_storages()
        return rag
