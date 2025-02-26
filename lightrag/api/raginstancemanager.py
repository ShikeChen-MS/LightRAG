from lightrag import LightRAG
from lightrag.llm.azure_openai import azure_openai_complete_if_cache, azure_openai_embed
from lightrag.types import GPTKeywordExtractionFormat
from lightrag.utils import EmbeddingFunc


class RAGInstanceManager:
    instance = None

    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super(RAGInstanceManager, cls).__new__(cls, *args, **kwargs)
        return cls.instance

    def __init__(self, *args, **kwargs):
        if not hasattr(self, "_initialized"):
            super(RAGInstanceManager, self).__init__(*args, **kwargs)
            self._initialized = True
            self.rag_instances = {}
            self.args = kwargs["args"]

    def get_rag_instance(self,
        rag_id: str,
        storage_connection_string: str,
        embedding_endpoint: str,
        embedding_api_version: str,
        embedding_dimension: int,
        max_embedding_tokens: int,
        embedding_model: str,
        llm_model: str,
        llm_endpoint: str,
        llm_api_version: str,
        affinity_token: str
    ) -> LightRAG:
        if rag_id in self.rag_instances:
            return self.rag_instances[rag_id]
        else:
            async def azure_openai_model_complete(
                    prompt,
                    access_token: str = None,
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
                    llm_model,
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    base_url=llm_endpoint,
                    access_token=access_token,
                    api_version=llm_api_version,
                    **kwargs,
                )

            embedding_func = EmbeddingFunc(
                embedding_dim=embedding_dimension,
                max_token_size=max_embedding_tokens,
                func=lambda texts, access_token: azure_openai_embed(
                    texts,
                    model=embedding_model,
                    base_url=embedding_endpoint,
                    access_token=access_token,
                    api_version=embedding_api_version
                )
            )
            self.rag_instances[rag_id] = LightRAG(
                affinity_token=affinity_token,
                storage_connection_string=storage_connection_string,
                llm_model_func=azure_openai_model_complete,
                chunk_token_size=int(self.args.chunk_size),
                chunk_overlap_token_size=int(self.args.chunk_overlap_size),
                llm_model_kwargs={
                    "timeout": self.args.timeout,
                },
                llm_model_name=llm_model,
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
            self.rag_instances[affinity_token] = self.rag_instances[rag_id]
        return self.rag_instances[rag_id]

    def get_rag_instance_by_affinity_token(self, affinity_token: str) -> LightRAG:
        if affinity_token not in self.rag_instances:
            raise ValueError(f"Affinity token {affinity_token} not found in RAG instances.")
        return self.rag_instances[affinity_token]




