import asyncio
import hashlib
from typing import Dict

from lightrag.az_token_credential import LightRagTokenCredential
from lightrag import LightRAG
from lightrag.llm.azure_openai import azure_openai_complete_if_cache, azure_openai_embed
from lightrag.types import GPTKeywordExtractionFormat
from lightrag.utils import EmbeddingFunc, always_get_an_event_loop

class RAGInstanceManager:
    instance = None
    _lock = asyncio.Lock()

    # Singleton pattern
    # preserve one instance of RAGInstanceManager under instance variable
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
            self.rag_instances: Dict[str, LightRAG] = {}
            # use kwargs to accept named arguments
            # here we take the args from argparser
            self.args = kwargs["args"]

    async def get_rag_instance(self,
        storage_account_url: str,
        storage_container_name: str,
        access_token: LightRagTokenCredential
    ) -> LightRAG:
        async with self._lock:
            # calculating the hash of storage account url + container name
            # and take first 13 characters as the id, this also serves as
            # affinity token; given that same storage account url + container name
            # will always point to one LightRAG storage.
            if not storage_account_url.endswith("/"):
                connection_str = storage_account_url + "/"
            connection_str += storage_container_name
            connection_str = connection_str.lower()
            hash_object = hashlib.sha256(connection_str.encode())
            rag_id = hash_object.hexdigest()[:13]
            if rag_id in self.rag_instances:
                return self.rag_instances[rag_id]
            else:
                async def azure_openai_model_complete(
                        prompt,
                        access_token,
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
                        self.args.llm_model,
                        prompt,
                        access_token,
                        system_prompt=system_prompt,
                        history_messages=history_messages,
                        base_url=self.args.llm_binding_host,
                        api_version=self.args.llm_api_version,
                        **kwargs,
                    )

                embedding_func = EmbeddingFunc(
                    embedding_dim=self.args.embedding_dim,
                    max_token_size=self.args.max_embed_tokens,
                    func=lambda texts, access_token: azure_openai_embed(
                        texts,
                        self.args.embedding_model,
                        access_token,
                        self.args.embedding_binding_host,
                        self.args.embedding_api_version,
                    )
                )
                self.rag_instances[rag_id] = LightRAG(
                    affinity_token=rag_id,
                    storage_account_url=storage_account_url,
                    storage_container_name=storage_container_name,
                    llm_model_func=azure_openai_model_complete,
                    chunk_token_size=int(self.args.chunk_size),
                    chunk_overlap_token_size=int(self.args.chunk_overlap_size),
                    llm_model_kwargs={
                        "timeout": self.args.timeout,
                    },
                    llm_model_name=self.args.llm_model,
                    llm_model_max_async=self.args.max_async,
                    llm_model_max_token_size=self.args.max_tokens,
                    embedding_func=embedding_func,
                    kv_storage=self.args.kv_storage,
                    graph_storage=self.args.graph_storage,
                    vector_storage=self.args.vector_storage,
                    doc_status_storage=self.args.doc_status_storage,
                    vector_db_storage_cls_kwargs={
                        "cosine_better_than_threshold": self.args.cosine_threshold
                    },
                    enable_llm_cache_for_entity_extract=False,  # set to True for debuging to reduce llm fee
                    embedding_cache_config={
                        "enabled": True,
                        "similarity_threshold": 0.95,
                        "use_llm_check": False,
                    },
                    log_level=self.args.log_level,
                    namespace_prefix=self.args.namespace_prefix,
                    auto_manage_storages_states=False,
                )
        # The storage initializing is expensive operation (takes time to fetch files from blob)
        # so we delay it after LightRAG instance created. and make it none blocking.
        # In actual API call, we will check storage status before any actual ops on storage.
        # # After LightRAG instance created, storage should be marked as CREATED.
        # # After invoke initialize, storage should be marked as INITIALIZING.
        # # After initializing done, storage should be marked as INITIALIZED.
        # # Only then we can proceed with any ops on storage.
        await self.rag_instances[rag_id].initialize_storages(access_token)
        return self.rag_instances[rag_id]

    async def get_rag_instance_by_affinity_token(self, affinity_token: str) -> LightRAG:
        async with self._lock:
            if affinity_token not in self.rag_instances:
                raise ValueError(f"Affinity token {affinity_token} not found in RAG instances.")
        return self.rag_instances[affinity_token]

    def get_lightrag(self, *args, **kwargs) -> LightRAG:
        if self.instance is None:
            raise ValueError("RAGInstanceManager is not initialized. No LightRAG instances available.")
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.instance.get_rag_instance(*args, **kwargs))

    def get_lightrag_by_affinity_token(self, affinity_token: str) -> LightRAG:
        if self.instance is None:
            raise ValueError("RAGInstanceManager is not initialized. No LightRAG instances available.")
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.instance.get_rag_instance_by_affinity_token(affinity_token))