import asyncio
import hashlib
import threading
from typing import Dict, Any
from lightrag.az_token_credential import LightRagTokenCredential
from ..base import InitializeStatus
from ..document_manager import DocumentManager
from ..lightrag import LightRAG
from ..llm.azure_openai import azure_openai_complete_if_cache, azure_openai_embed
from ..types import GPTKeywordExtractionFormat
from ..utils import EmbeddingFunc, always_get_an_event_loop


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
            self.rag_instances: Dict[str, Any] = {}
            # use kwargs to accept named arguments
            # here we take the args from argparser
            self.args = kwargs["args"]

    async def get_rag_instance(
        self,
        storage_account_url: str,
        storage_container_name: str,
        access_token: LightRagTokenCredential,
    ) -> Any:
        # calculating the hash of storage account url + container name
        # and take the hash(since SHA256 has fixed length of 64 characters as the id,
        # this also serves as affinity token;
        # given that same storage account url + container name
        # will always point to one LightRAG storage.
        connection_str = storage_account_url
        if not storage_account_url.endswith("/"):
            connection_str = storage_account_url + "/"
        connection_str += storage_container_name
        connection_str = connection_str.lower()
        hash_object = hashlib.sha256(connection_str.encode())
        rag_id = hash_object.hexdigest()
        async with self._lock:
            if rag_id in self.rag_instances:
                return self.rag_instances[rag_id]
            else:
                # This function and following embedding_func will
                # be passed to LightRAG instance to be used for completion and embedding
                async def azure_openai_model_complete(
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
                    func=lambda aad_token, texts: azure_openai_embed(
                        aad_token,
                        texts,
                        self.args.embedding_model,
                        self.args.embedding_binding_host,
                        self.args.embedding_api_version,
                    ),
                )
                doc_manager = DocumentManager(f"{self.args.working_dir}/{self.args.input_dir}")
                self.rag_instances[rag_id] = LightRAG(
                    affinity_token=rag_id,
                    working_dir=self.args.working_dir,
                    scan_progress={
                        "is_scanning": False,
                        "current_file": "",
                        "indexed_count": 0,
                        "total_files": 0,
                        "progress": 0,
                    },
                    progress_lock=threading.Lock(),
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
                    document_manager=doc_manager,
                    max_parallel_insert=self.args.max_parallel_insert,
                    cosine_threshold=self.args.cosine_threshold,
                )
                self.rag_instances[rag_id].initialize_status = (
                    InitializeStatus.INITIALIZING
                )
        # The storage initializing is expensive operation (takes time to fetch files from blob)
        # so we delay it after LightRAG instance created. and make it none blocking.
        # In actual API call, we will check storage status before any actual ops on storage.
        try:
            await self.rag_instances[rag_id].initialize_storages(access_token)
        except Exception as e:
            print(f"Failed to initialize storage for {rag_id}: {e}")
        return self.rag_instances[rag_id]

    async def get_rag_instance_by_affinity_token(self, affinity_token: str) -> LightRAG:
        async with self._lock:
            if affinity_token not in self.rag_instances:
                raise ValueError(
                    f"Affinity token {affinity_token} not found in RAG instances."
                )
        return self.rag_instances[affinity_token]

    def get_lightrag(self, *args, **kwargs) -> LightRAG:
        if self.instance is None:
            raise ValueError(
                "RAGInstanceManager is not initialized. No LightRAG instances available."
            )
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.instance.get_rag_instance(*args, **kwargs))

    def get_lightrag_by_affinity_token(self, affinity_token: str) -> LightRAG:
        if self.instance is None:
            raise ValueError(
                "RAGInstanceManager is not initialized. No LightRAG instances available."
            )
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.instance.get_rag_instance_by_affinity_token(affinity_token)
        )
