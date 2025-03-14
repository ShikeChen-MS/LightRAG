from __future__ import annotations
import asyncio
import configparser
import os
import threading
import logging
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Any, AsyncIterator, Callable, Iterator, cast, final, Dict
from .kg import (
    STORAGE_ENV_REQUIREMENTS,
    STORAGES,
    verify_storage_implementation,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
    QueryParam,
    StorageNameSpace,
    StoragesStatus,
    InitializeStatus,
)
from .namespace import NameSpace, make_namespace
from .operate import (
    chunking_by_token_size,
    extract_entities,
    extract_keywords_only,
    kg_query,
    kg_query_with_keywords,
    mix_kg_vector_query,
    naive_query,
)
from .postgresql import PostgreSQLDB
from .prompt import GRAPH_FIELD_SEP
from .utils import (
    EmbeddingFunc,
    always_get_an_event_loop,
    compute_mdhash_id,
    convert_response_to_json,
    encode_string_by_tiktoken,
    lazy_external_import,
    limit_async_func_call,
)
from .types import KnowledgeGraph
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# TODO: TO REMOVE @Yannick
config = configparser.ConfigParser()
config.read("config.ini", "utf-8")


@final
@dataclass
class LightRAG:
    """LightRAG: Simple and Fast Retrieval-Augmented Generation."""

    def __init__(
        self,
        db: PostgreSQLDB,
        scan_progress: Dict[str, Any],
        progress_lock: threading.Lock,
        db_url: str,
        db_name: str,
        llm_model_func: Callable[..., object],
        chunk_token_size: int,
        chunk_overlap_token_size: int,
        llm_model_kwargs: Dict[str, Any],
        llm_model_name: str,
        llm_model_max_async: int,
        llm_model_max_token_size: int,
        embedding_func: EmbeddingFunc,
        kv_storage: str,
        vector_storage: str,
        graph_storage: str,
        doc_status_storage: str,
        vector_db_storage_cls_kwargs: Dict[str, Any],
        enable_llm_cache_for_entity_extract: bool,
        embedding_cache_config: Dict[str, Any],
        log_level: int,
        namespace_prefix: str,
        max_parallel_insert: int,
        cosine_threshold: float = 0.2,
    ):
        # which in turn target to specific set of users, therefore, affinity token
        # stays same for all these users given this instance can handle all these requests
        # each instance of LightRAG will be dedicated to specific storage
        # without further initializing more LightRAG instances
        self.db = db
        self.scan_progress: Dict[str, Any] = scan_progress
        self.progress_lock = progress_lock
        self.db_url: str = db_url
        self.db_name: str = db_name
        self.kv_storage: str = kv_storage
        self.vector_storage: str = vector_storage
        self.graph_storage: str = graph_storage
        self.doc_status_storage: str = doc_status_storage
        self.log_level: int = log_level
        self.log_file_path: str = os.path.join(os.getcwd(), "lightrag.log")
        self.entity_extract_max_gleaning: int = 1
        self.entity_summary_to_max_tokens: int = 500
        self.chunk_token_size: int = chunk_token_size
        self.chunk_overlap_token_size: int = chunk_overlap_token_size
        self.tiktoken_model_name: str = "gpt-4o"
        self.chunking_func: Callable[
            [
                str,
                str | None,
                bool,
                int,
                int,
                str,
            ],
            list[dict[str, Any]],
        ] = chunking_by_token_size
        self.node_embedding_algorithm: str = "node2vec"
        self.node2vec_params: dict[str, int] = {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
        self.embedding_func: EmbeddingFunc = embedding_func
        self.embedding_batch_num: int = 32
        self.embedding_func_max_async: int = 16
        self.embedding_cache_config: dict[str, Any] = embedding_cache_config
        self.llm_model_func: Callable[..., object] = llm_model_func
        self.llm_model_name: str = llm_model_name
        self.llm_model_max_token_size: int = llm_model_max_token_size
        self.llm_model_max_async: int = llm_model_max_async
        self.llm_model_kwargs: dict[str, Any] = llm_model_kwargs
        self.vector_db_storage_cls_kwargs: dict[str, Any] = vector_db_storage_cls_kwargs
        self.namespace_prefix: str = namespace_prefix
        self.enable_llm_cache: bool = True
        self.enable_llm_cache_for_entity_extract: bool = (
            enable_llm_cache_for_entity_extract
        )
        self.max_parallel_insert: int = max_parallel_insert
        self.addon_params: dict[str, Any] = {}
        self.auto_manage_storages_states: bool = False
        self.convert_response_to_json_func: Callable[[str], dict[str, Any]] = (
            convert_response_to_json
        )
        self.cosine_better_than_threshold: float = cosine_threshold
        self._storages_status: StoragesStatus = StoragesStatus.NOT_CREATED
        self.initialize_status: InitializeStatus = InitializeStatus.NOT_INITIALIZED
        # Verify storage implementation compatibility and environment variables
        storage_configs = [
            ("KV_STORAGE", self.kv_storage),
            ("VECTOR_STORAGE", self.vector_storage),
            ("GRAPH_STORAGE", self.graph_storage),
            ("DOC_STATUS_STORAGE", self.doc_status_storage),
        ]

        for storage_type, storage_name in storage_configs:
            # Verify storage implementation compatibility
            verify_storage_implementation(storage_type, storage_name)
            # Check environment variables
            self.check_storage_env_vars(storage_name)

        # Ensure vector_db_storage_cls_kwargs has required fields
        self.vector_db_storage_cls_kwargs = {
            "cosine_better_than_threshold": self.cosine_better_than_threshold,
            **self.vector_db_storage_cls_kwargs,
        }

        # Show config
        global_config = self.__dict__
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in global_config.items()])
        logging.debug(f"LightRAG init with param:\n  {_print_config}\n")

        # Init LLM
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(  # type: ignore
            self.embedding_func
        )

    def __del__(self):
        if self.auto_manage_storages_states:
            self._run_async_safely(self.finalize_storages, "Storage Finalization")

    def check_storage_status(self):
        return self._storages_status.value

    def _run_async_safely(self, async_func, action_name=""):
        """Safely execute an async function, avoiding event loop conflicts."""
        try:
            loop = always_get_an_event_loop()
            if loop.is_running():
                task = loop.create_task(async_func())
                task.add_done_callback(
                    lambda t: logging.info(f"{action_name} completed!")
                )
            else:
                loop.run_until_complete(async_func())
        except RuntimeError:
            logging.warning(
                f"No running event loop, creating a new loop for {action_name}."
            )
            loop = asyncio.new_event_loop()
            loop.run_until_complete(async_func())
            loop.close()

    async def create_storages(self):
        # Initialize all storages
        global_config = self.__dict__
        self.key_string_value_json_storage_cls: type[BaseKVStorage] = (
            self._get_storage_class(self.kv_storage)
        )  # type: ignore
        self.vector_db_storage_cls: type[BaseVectorStorage] = self._get_storage_class(
            self.vector_storage
        )  # type: ignore
        self.graph_storage_cls: type[BaseGraphStorage] = self._get_storage_class(
            self.graph_storage
        )  # type: ignore
        self.key_string_value_json_storage_cls = partial(  # type: ignore
            self.key_string_value_json_storage_cls, global_config=global_config
        )
        self.vector_db_storage_cls = partial(  # type: ignore
            self.vector_db_storage_cls, global_config=global_config
        )
        self.graph_storage_cls = partial(  # type: ignore
            self.graph_storage_cls, global_config=global_config
        )

        # Initialize document status storage
        self.doc_status_storage_cls = self._get_storage_class(self.doc_status_storage)

        self.llm_response_cache: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
            ),
            embedding_func=self.embedding_func,
            db = self.db
        )

        self.full_docs: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.KV_STORE_FULL_DOCS
            ),
            embedding_func=self.embedding_func,
            db = self.db
        )
        self.text_chunks: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.KV_STORE_TEXT_CHUNKS
            ),
            embedding_func=self.embedding_func,
            db = self.db
        )
        self.chunk_entity_relation_graph: BaseGraphStorage = self.graph_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION
            ),
            embedding_func=self.embedding_func,
            db = self.db
        )

        self.entities_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.VECTOR_STORE_ENTITIES
            ),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name", "input_source_id"},
            db = self.db
        )
        self.relationships_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.VECTOR_STORE_RELATIONSHIPS
            ),
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id", "input_source_id"},
            db = self.db
        )
        self.chunks_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.VECTOR_STORE_CHUNKS
            ),
            embedding_func=self.embedding_func,
            db = self.db
        )

        # Initialize document status storage
        self.doc_status: DocStatusStorage = self.doc_status_storage_cls(
            namespace=make_namespace(self.namespace_prefix, NameSpace.DOC_STATUS),
            global_config=global_config,
            embedding_func=None,
            db = self.db
        )
        if self.llm_response_cache and hasattr(
            self.llm_response_cache, "global_config"
        ):
            hashing_kv = self.llm_response_cache
        else:
            hashing_kv = self.key_string_value_json_storage_cls(  # type: ignore
                namespace=make_namespace(
                    self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                ),
                embedding_func=self.embedding_func,
                db = self.db
            )

        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
            partial(
                self.llm_model_func,  # type: ignore
                hashing_kv=hashing_kv,
                **self.llm_model_kwargs,
            )
        )
        self._storages_status = StoragesStatus.CREATED

    async def clear_storages(self):
        """Asynchronously clear storages"""
        if self._storages_status == StoragesStatus.INITIALIZED:
            tasks = []

            for storage in (
                self.full_docs,
                self.text_chunks,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.chunk_entity_relation_graph,
                self.llm_response_cache,
                self.doc_status,
            ):
                if storage:
                    tasks.append(storage.drop())

            await asyncio.gather(*tasks)

            self._storages_status = StoragesStatus.FINALIZED
            logging.debug("Finalized Storages")

    async def finalize_storages(self):
        """Asynchronously finalize the storages"""
        await self.db.pool.close()
        logging.debug("Finalized Storages")

    async def get_graph_labels(self):
        text = await self.chunk_entity_relation_graph.get_all_labels()
        return text

    async def get_knowledge_graph(
        self, node_label: str, max_depth: int
    ) -> KnowledgeGraph:
        return await self.chunk_entity_relation_graph.get_knowledge_graph(
            node_label=node_label, max_depth=max_depth
        )

    def _get_storage_class(self, storage_name: str) -> Callable[..., Any]:
        import_path = STORAGES[storage_name]
        storage_class = lazy_external_import(import_path, storage_name)
        return storage_class

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text by removing null bytes (0x00) and whitespace"""
        return text.strip().replace("\x00", "")

    def insert(
        self,
        ai_access_token: str,
        input: str | list[str],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        ids: str | list[str] | None = None,
    ) -> None:
        """Sync Insert documents with checkpoint support
        """
        loop = always_get_an_event_loop()
        loop.run_until_complete(
            self.ainsert(
                ai_access_token,
                input,
                split_by_character,
                split_by_character_only,
                ids,
            )
        )

    async def ainsert(
        self,
        ai_access_token:str,
        input: str | list[str],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        ids: str | list[str] | None = None,
    ) -> None:
        """Async Insert documents with checkpoint support"""
        await self.apipeline_enqueue_documents(
            ids, input
        )
        await self.apipeline_process_enqueue_documents(
            ai_access_token,
            split_by_character,
            split_by_character_only,
        )

    def insert_custom_chunks(
        self,
        full_text: str,
        text_chunks: list[str],
        ai_access_token: str,
        doc_id: str | list[str] | None = None,
    ) -> None:
        loop = always_get_an_event_loop()
        loop.run_until_complete(
            self.ainsert_custom_chunks(
                full_text,
                text_chunks,
                ai_access_token,
                doc_id,
            )
        )

    async def ainsert_custom_chunks(
        self,
        full_text: str,
        text_chunks: list[str],
        ai_access_token: str,
        doc_id: str | None = None,
    ) -> None:
        update_storage = False
        try:
            # Clean input texts
            full_text = self.clean_text(full_text)
            text_chunks = [self.clean_text(chunk) for chunk in text_chunks]

            # Process cleaned texts
            if doc_id is None:
                doc_key = compute_mdhash_id(full_text, prefix="doc-")
            else:
                doc_key = doc_id
            new_docs = {doc_key: {"content": full_text}}

            _add_doc_keys = await self.full_docs.filter_keys({doc_key})
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logging.warning("This document is already in the storage.")
                return

            update_storage = True
            logging.info(f"Inserting {len(new_docs)} docs")

            inserting_chunks: dict[str, Any] = {}
            for chunk_text in text_chunks:
                chunk_key = compute_mdhash_id(chunk_text, prefix="chunk-")

                inserting_chunks[chunk_key] = {
                    "content": chunk_text,
                    "full_doc_id": doc_key,
                }

            doc_ids = set(inserting_chunks.keys())
            add_chunk_keys = await self.text_chunks.filter_keys(doc_ids)
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in add_chunk_keys
            }
            if not len(inserting_chunks):
                logging.warning("All chunks are already in the storage.")
                return

            tasks = [
                self.chunks_vdb.upsert(inserting_chunks, ai_access_token),
                self._process_entity_relation_graph(ai_access_token, inserting_chunks),
                self.full_docs.upsert(new_docs),
                self.text_chunks.upsert(inserting_chunks),
            ]
            await asyncio.gather(*tasks)

        finally:
            if update_storage:
                await self._insert_done()

    async def apipeline_enqueue_documents(
        self,
        source_ids: str | list[str],
        input: str | list[str]
    ) -> None:
        """
        Pipeline for Processing Documents

        1. Validate ids if provided or generate MD5 hash IDs
        2. Remove duplicate contents
        3. Generate document initial status
        4. Filter out already processed documents
        5. Enqueue document in status
        """
        if isinstance(input, str):
            input = [input]

        if len(source_ids) != len(input):
            raise ValueError("Number of source_ids must match the number of documents")

        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source_ids must be unique")


        # Generate contents dict of IDs provided by user and documents
        contents = {id_: doc for id_, doc in zip(source_ids, input)}

        # 2. Remove duplicate contents
        unique_contents = {
            id_: content
            for content, id_ in {
                content: id_ for id_, content in contents.items()
            }.items()
        }

        # 3. Generate document initial status
        new_docs: dict[str, Any] = {
            id_: {
                "content": content,
                "source_id": id_,
                "content_summary": self._get_content_summary(content),
                "content_length": len(content),
                "status": DocStatus.PENDING,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            for id_, content in unique_contents.items()
        }

        # 4. Filter out already processed documents
        # Get docs ids
        all_new_doc_ids = set(new_docs.keys())
        # Exclude IDs of documents that are already in progress
        unique_new_doc_ids = await self.doc_status.filter_keys(all_new_doc_ids)
        excluded_doc = all_new_doc_ids - unique_new_doc_ids
        if len(excluded_doc) > 0:
            raise ValueError(f"Following source Id already exists: {excluded_doc}")
        # Filter new_docs to only include documents with unique IDs
        new_docs = {doc_id: new_docs[doc_id] for doc_id in unique_new_doc_ids}

        if not new_docs:
            logging.info("No new unique documents were found.")
            return

        # 5. Store status document
        await self.doc_status.upsert(
            new_docs
        )
        logging.info(f"Stored {len(new_docs)} new unique documents")

    async def apipeline_process_enqueue_documents(
        self,
        ai_access_token: str,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
    ) -> None:
        """
        Process pending documents by splitting them into chunks, processing
        each chunk for entity and relation extraction, and updating the
        document status.

        1. Get all pending, failed, and abnormally terminated processing documents.
        2. Split document content into chunks
        3. Process each chunk for entity and relation extraction
        4. Update the document status
        """
        # 1. Get all pending, failed, and abnormally terminated processing documents.
        # Run the asynchronous status retrievals in parallel using asyncio.gather
        processing_docs, failed_docs, pending_docs = await asyncio.gather(
            self.doc_status.get_docs_by_status(DocStatus.PROCESSING),
            self.doc_status.get_docs_by_status(DocStatus.FAILED),
            self.doc_status.get_docs_by_status(DocStatus.PENDING),
        )

        to_process_docs: dict[str, DocProcessingStatus] = {}
        to_process_docs.update(processing_docs)
        to_process_docs.update(failed_docs)
        to_process_docs.update(pending_docs)

        if not to_process_docs:
            logging.info("All documents have been processed or are duplicates")
            return

        # 2. split docs into chunks, insert chunks, update doc status
        docs_batches = [
            list(to_process_docs.items())[i : i + self.max_parallel_insert]
            for i in range(0, len(to_process_docs), self.max_parallel_insert)
        ]

        logging.info(f"Number of batches to process: {len(docs_batches)}.")

        batches: list[Any] = []
        # 3. iterate over batches
        for batch_idx, docs_batch in enumerate(docs_batches):

            async def batch(
                batch_idx: int,
                docs_batch: list[tuple[str, DocProcessingStatus]],
                size_batch: int,
            ) -> None:
                logging.info(f"Start processing batch {batch_idx + 1} of {size_batch}.")
                # 4. iterate over batch
                for doc_id_processing_status in docs_batch:
                    doc_id, status_doc = doc_id_processing_status
                    # Generate chunks from document
                    chunks: dict[str, Any] = {
                        compute_mdhash_id(dp["content"], prefix="chunk-"): {
                            **dp,
                            "full_doc_id": doc_id,
                        }
                        for dp in self.chunking_func(
                            status_doc.content,
                            status_doc.source_id,
                            split_by_character,
                            split_by_character_only,
                            self.chunk_overlap_token_size,
                            self.chunk_token_size,
                            self.tiktoken_model_name,
                        )
                    }
                    # Process document (text chunks and full docs) in parallel
                    tasks = [
                        self.doc_status.upsert(
                            {
                                doc_id: {
                                    "status": DocStatus.PROCESSING,
                                    "source_id": doc_id,
                                    "updated_at": datetime.now().isoformat(),
                                    "content": status_doc.content,
                                    "content_summary": status_doc.content_summary,
                                    "content_length": status_doc.content_length,
                                    "created_at": status_doc.created_at,
                                }
                            }
                        ),
                        self.chunks_vdb.upsert(chunks, ai_access_token),
                        self._process_entity_relation_graph(ai_access_token, chunks),
                        self.full_docs.upsert(
                            {doc_id: {"content": status_doc.content}}
                        ),
                        self.text_chunks.upsert(chunks),
                    ]
                    try:
                        await asyncio.gather(*tasks)
                        await self.doc_status.upsert(
                            {
                                doc_id: {
                                    "status": DocStatus.PROCESSED,
                                    "source_id": doc_id,
                                    "chunks_count": len(chunks),
                                    "content": status_doc.content,
                                    "content_summary": status_doc.content_summary,
                                    "content_length": status_doc.content_length,
                                    "created_at": status_doc.created_at,
                                    "updated_at": datetime.now().isoformat(),
                                }
                            }
                        )
                    except Exception as e:
                        logging.error(f"Failed to process document {doc_id}: {str(e)}")
                        await self.doc_status.upsert(
                            {
                                doc_id: {
                                    "status": DocStatus.FAILED,
                                    "error": str(e),
                                    "source_id": doc_id,
                                    "content": status_doc.content,
                                    "content_summary": status_doc.content_summary,
                                    "content_length": status_doc.content_length,
                                    "created_at": status_doc.created_at,
                                    "updated_at": datetime.now().isoformat(),
                                }
                            }
                        )
                        continue
                logging.info(f"Completed batch {batch_idx + 1} of {len(docs_batches)}.")

            batches.append(batch(batch_idx, docs_batch, len(docs_batches)))

        await asyncio.gather(*batches)
        await self._insert_done()

    async def _process_entity_relation_graph(
        self, ai_access_token: str, chunk: dict[str, Any]
    ) -> None:
        try:
            await extract_entities(
                ai_access_token,
                chunk,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                relationships_vdb=self.relationships_vdb,
                llm_response_cache=self.llm_response_cache,
                global_config=self.__dict__,
            )
        except Exception as e:
            logging.error("Failed to extract entities and relationships")
            raise e

    async def _insert_done(self) -> None:
        tasks = [
            cast(StorageNameSpace, storage_inst).index_done_callback()
            for storage_inst in [  # type: ignore
                self.full_docs,
                self.text_chunks,
                self.llm_response_cache,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.chunk_entity_relation_graph,
            ]
            if storage_inst is not None
        ]
        await asyncio.gather(*tasks)
        logging.info("All Insert done")

    def insert_custom_kg(
        self,
        ai_access_token: str,
        custom_kg: dict[str, Any],
    ) -> None:
        loop = always_get_an_event_loop()
        loop.run_until_complete(
            self.ainsert_custom_kg(
                ai_access_token,
                custom_kg,
            )
        )

    async def ainsert_custom_kg(
        self,
        ai_access_token: str,
        custom_kg: dict[str, Any],
    ) -> None:
        update_storage = False
        try:
            # Insert chunks into vector storage
            all_chunks_data: dict[str, dict[str, str]] = {}
            chunk_to_source_map: dict[str, str] = {}
            for chunk_data in custom_kg.get("chunks", {}):
                chunk_content = self.clean_text(chunk_data["content"])
                source_id = chunk_data["source_id"]
                tokens = len(
                    encode_string_by_tiktoken(
                        chunk_content, model_name=self.tiktoken_model_name
                    )
                )
                chunk_order_index = (
                    0
                    if "chunk_order_index" not in chunk_data.keys()
                    else chunk_data["chunk_order_index"]
                )
                chunk_id = compute_mdhash_id(chunk_content, prefix="chunk-")

                chunk_entry = {
                    "content": chunk_content,
                    "source_id": source_id,
                    "tokens": tokens,
                    "chunk_order_index": chunk_order_index,
                    "full_doc_id": source_id,
                    "status": DocStatus.PROCESSED,
                }
                all_chunks_data[chunk_id] = chunk_entry
                chunk_to_source_map[source_id] = chunk_id
                update_storage = True

            if all_chunks_data:
                await self.chunks_vdb.upsert(all_chunks_data, ai_access_token)
            if all_chunks_data:
                await self.text_chunks.upsert(all_chunks_data)

            # Insert entities into knowledge graph
            all_entities_data: list[dict[str, str]] = []
            for entity_data in custom_kg.get("entities", []):
                entity_name = f'"{entity_data["entity_name"].upper()}"'
                entity_type = entity_data.get("entity_type", "UNKNOWN")
                description = entity_data.get("description", "No description provided")
                # source_id = entity_data["source_id"]
                source_chunk_id = entity_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")

                # Log if source_id is UNKNOWN
                if source_id == "UNKNOWN":
                    logging.warning(
                        f"Entity '{entity_name}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                # Prepare node data
                node_data: dict[str, str] = {
                    "entity_type": entity_type,
                    "description": description,
                    "source_id": source_id,
                }
                # Insert node data into the knowledge graph
                await self.chunk_entity_relation_graph.upsert_node(
                    entity_name, node_data=node_data
                )
                node_data["entity_name"] = entity_name
                all_entities_data.append(node_data)
                update_storage = True

            # Insert relationships into knowledge graph
            all_relationships_data: list[dict[str, str]] = []
            for relationship_data in custom_kg.get("relationships", []):
                src_id = f'"{relationship_data["src_id"].upper()}"'
                tgt_id = f'"{relationship_data["tgt_id"].upper()}"'
                description = relationship_data["description"]
                keywords = relationship_data["keywords"]
                weight = relationship_data.get("weight", 1.0)
                # source_id = relationship_data["source_id"]
                source_chunk_id = relationship_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")

                # Log if source_id is UNKNOWN
                if source_id == "UNKNOWN":
                    logging.warning(
                        f"Relationship from '{src_id}' to '{tgt_id}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                # Check if nodes exist in the knowledge graph
                for need_insert_id in [src_id, tgt_id]:
                    if not (
                        await self.chunk_entity_relation_graph.has_node(need_insert_id)
                    ):
                        await self.chunk_entity_relation_graph.upsert_node(
                            need_insert_id,
                            node_data={
                                "source_id": source_id,
                                "description": "UNKNOWN",
                                "entity_type": "UNKNOWN",
                            },
                        )

                # Insert edge into the knowledge graph
                await self.chunk_entity_relation_graph.upsert_edge(
                    src_id,
                    tgt_id,
                    edge_data={
                        "weight": weight,
                        "description": description,
                        "keywords": keywords,
                        "source_id": source_id,
                    },
                )
                edge_data: dict[str, str] = {
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "description": description,
                    "keywords": keywords,
                }
                all_relationships_data.append(edge_data)
                update_storage = True

            # Insert entities into vector storage if needed
            data_for_vdb = {
                compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                    "content": dp["entity_name"] + dp["description"],
                    "entity_name": dp["entity_name"],
                }
                for dp in all_entities_data
            }
            await self.entities_vdb.upsert(data_for_vdb, ai_access_token)

            # Insert relationships into vector storage if needed
            data_for_vdb = {
                compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                    "src_id": dp["src_id"],
                    "tgt_id": dp["tgt_id"],
                    "content": dp["keywords"]
                    + dp["src_id"]
                    + dp["tgt_id"]
                    + dp["description"],
                }
                for dp in all_relationships_data
            }
            await self.relationships_vdb.upsert(data_for_vdb, ai_access_token)

        finally:
            if update_storage:
                await self._insert_done()

    def query(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> str | Iterator[str]:
        """
        Perform a sync query.
        """
        loop = always_get_an_event_loop()

        return loop.run_until_complete(self.aquery(query, param, system_prompt))  # type: ignore

    async def aquery(
        self,
        query: str,
        ai_access_token: str,
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> str | AsyncIterator[str]:
        """
        Perform a async query.
        """
        if param.mode in ["local", "global", "hybrid"]:
            response = await kg_query(
                query,
                ai_access_token,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                self.__dict__,
                hashing_kv=(
                    self.llm_response_cache
                    if self.llm_response_cache
                    and hasattr(self.llm_response_cache, "global_config")
                    else self.key_string_value_json_storage_cls(
                        namespace=make_namespace(
                            self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                        ),
                        global_config=self.__dict__,
                        embedding_func=self.embedding_func,
                    )
                ),
                system_prompt=system_prompt,
            )
        elif param.mode == "naive":
            response = await naive_query(
                query,
                ai_access_token,
                self.chunks_vdb,
                self.text_chunks,
                param,
                self.__dict__,
                hashing_kv=(
                    self.llm_response_cache
                    if self.llm_response_cache
                    and hasattr(self.llm_response_cache, "global_config")
                    else self.key_string_value_json_storage_cls(
                        namespace=make_namespace(
                            self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                        ),
                        global_config=self.__dict__,
                        embedding_func=self.embedding_func,
                    )
                ),
                system_prompt=system_prompt,
            )
        elif param.mode == "mix":
            response = await mix_kg_vector_query(
                query,
                ai_access_token,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.text_chunks,
                param,
                self.__dict__,
                hashing_kv=(
                    self.llm_response_cache
                    if self.llm_response_cache
                    and hasattr(self.llm_response_cache, "global_config")
                    else self.key_string_value_json_storage_cls(
                        namespace=make_namespace(
                            self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                        ),
                        global_config=self.__dict__,
                        embedding_func=self.embedding_func,
                    )
                ),
                system_prompt=system_prompt,
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response

    def query_with_separate_keyword_extraction(
        self,
        ai_access_token: str,
        query: str,
        prompt: str,
        param: QueryParam = QueryParam(),
    ):
        """
        1. Extract keywords from the 'query' using new function in operate.py.
        2. Then run the standard aquery() flow with the final prompt (formatted_question).
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aquery_with_separate_keyword_extraction(
                ai_access_token,
                query,
                prompt,
                param,
            )
        )

    async def aquery_with_separate_keyword_extraction(
        self,
        ai_access_token: str,
        query: str,
        prompt: str,
        param: QueryParam = QueryParam(),
    ) -> str | AsyncIterator[str]:
        """
        1. Calls extract_keywords_only to get HL/LL keywords from 'query'.
        2. Then calls kg_query(...) or naive_query(...), etc. as the main query, while also injecting the newly extracted keywords if needed.
        """
        # ---------------------
        # STEP 1: Keyword Extraction
        # ---------------------
        hl_keywords, ll_keywords = await extract_keywords_only(
            text=query,
            ai_access_token=ai_access_token,
            param=param,
            global_config=self.__dict__,
            hashing_kv=self.llm_response_cache
            or self.key_string_value_json_storage_cls(
                namespace=make_namespace(
                    self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                ),
                global_config=self.__dict__,
                embedding_func=self.embedding_func,
            ),
        )

        param.hl_keywords = hl_keywords
        param.ll_keywords = ll_keywords

        # ---------------------
        # STEP 2: Final Query Logic
        # ---------------------

        # Create a new string with the prompt and the keywords
        ll_keywords_str = ", ".join(ll_keywords)
        hl_keywords_str = ", ".join(hl_keywords)
        formatted_question = f"{prompt}\n\n### Keywords:\nHigh-level: {hl_keywords_str}\nLow-level: {ll_keywords_str}\n\n### Query:\n{query}"

        if param.mode in ["local", "global", "hybrid"]:
            response = await kg_query_with_keywords(
                formatted_question,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                self.__dict__,
                hashing_kv=(
                    self.llm_response_cache
                    if self.llm_response_cache
                    and hasattr(self.llm_response_cache, "global_config")
                    else self.key_string_value_json_storage_cls(
                        namespace=make_namespace(
                            self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                        ),
                        global_config=self.__dict__,
                        embedding_func=self.embedding_func,
                    )
                ),
            )
        elif param.mode == "naive":
            response = await naive_query(
                formatted_question,
                ai_access_token,
                self.chunks_vdb,
                self.text_chunks,
                param,
                self.__dict__,
                hashing_kv=(
                    self.llm_response_cache
                    if self.llm_response_cache
                    and hasattr(self.llm_response_cache, "global_config")
                    else self.key_string_value_json_storage_cls(
                        namespace=make_namespace(
                            self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                        ),
                        global_config=self.__dict__,
                        embedding_func=self.embedding_func,
                    )
                ),
            )
        elif param.mode == "mix":
            response = await mix_kg_vector_query(
                formatted_question,
                ai_access_token,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.text_chunks,
                param,
                self.__dict__,
                hashing_kv=(
                    self.llm_response_cache
                    if self.llm_response_cache
                    and hasattr(self.llm_response_cache, "global_config")
                    else self.key_string_value_json_storage_cls(
                        namespace=make_namespace(
                            self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                        ),
                        global_config=self.__dict__,
                        embedding_func=self.embedding_func,
                    )
                ),
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")

        await self._query_done()
        return response

    async def _query_done(self):
        await self.llm_response_cache.index_done_callback()

    def delete_by_entity(self, entity_name: str) -> None:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name))

    async def adelete_by_entity(self, entity_name: str) -> None:
        entity_name = f'"{entity_name.upper()}"'

        try:
            await self.entities_vdb.delete_entity(entity_name)
            await self.relationships_vdb.delete_entity_relation(entity_name)
            await self.chunk_entity_relation_graph.delete_node(entity_name)

            logging.info(
                f"Entity '{entity_name}' and its relationships have been deleted."
            )
            await self._delete_by_entity_done()
        except Exception as e:
            logging.error(f"Error while deleting entity '{entity_name}': {e}")

    async def _delete_by_entity_done(self) -> None:
        await asyncio.gather(
            *[
                cast(StorageNameSpace, storage_inst).index_done_callback()
                for storage_inst in [  # type: ignore
                    self.entities_vdb,
                    self.relationships_vdb,
                    self.chunk_entity_relation_graph,
                ]
            ]
        )

    def _get_content_summary(self, content: str, max_length: int = 100) -> str:
        """Get summary of document content

        Args:
            content: Original document content
            max_length: Maximum length of summary

        Returns:
            Truncated content with ellipsis if needed
        """
        content = content.strip()
        if len(content) <= max_length:
            return content
        return content[:max_length] + "..."

    async def get_processing_status(self) -> dict[str, int]:
        """Get current document processing status counts

        Returns:
            Dict with counts for each status
        """
        return await self.doc_status.get_status_counts()

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        """Get documents by status

        Returns:
            Dict with document id is keys and document status is values
        """
        return await self.doc_status.get_docs_by_status(status)

    async def adelete_by_doc_id(
        self,
        doc_id: str,
    ) -> None:
        """Delete a document and all its related data

        Args:
            doc_id: Document ID to delete
        """
        try:
            # 1. Get the document status and related data
            doc_status = await self.doc_status.get_by_id(doc_id)
            if not doc_status:
                logging.warning(f"Document {doc_id} not found")
                return

            logging.debug(f"Starting deletion for document {doc_id}")

            # 2. Get all related chunks
            chunks = await self.text_chunks.get_by_id(doc_id)
            if not chunks:
                return

            chunk_ids = list(chunks.keys())
            logging.debug(f"Found {len(chunk_ids)} chunks to delete")

            # 3. Before deleting, check the related entities and relationships for these chunks
            for chunk_id in chunk_ids:
                # Check entities
                entities = [
                    dp
                    for dp in self.entities_vdb.client_storage["data"]
                    if dp.get("source_id") == chunk_id
                ]
                logging.debug(f"Chunk {chunk_id} has {len(entities)} related entities")

                # Check relationships
                relations = [
                    dp
                    for dp in self.relationships_vdb.client_storage["data"]
                    if dp.get("source_id") == chunk_id
                ]
                logging.debug(
                    f"Chunk {chunk_id} has {len(relations)} related relations"
                )

            # Continue with the original deletion process...

            # 4. Delete chunks from vector database
            if chunk_ids:
                await self.chunks_vdb.delete(chunk_ids)
                await self.text_chunks.delete(chunk_ids)

            # 5. Find and process entities and relationships that have these chunks as source
            # Get all nodes in the graph
            nodes = self.chunk_entity_relation_graph._graph.nodes(data=True)
            edges = self.chunk_entity_relation_graph._graph.edges(data=True)

            # Track which entities and relationships need to be deleted or updated
            entities_to_delete = set()
            entities_to_update = {}  # entity_name -> new_source_id
            relationships_to_delete = set()
            relationships_to_update = {}  # (src, tgt) -> new_source_id

            # Process entities
            for node, data in nodes:
                if "source_id" in data:
                    # Split source_id using GRAPH_FIELD_SEP
                    sources = set(data["source_id"].split(GRAPH_FIELD_SEP))
                    sources.difference_update(chunk_ids)
                    if not sources:
                        entities_to_delete.add(node)
                        logging.debug(
                            f"Entity {node} marked for deletion - no remaining sources"
                        )
                    else:
                        new_source_id = GRAPH_FIELD_SEP.join(sources)
                        entities_to_update[node] = new_source_id
                        logging.debug(
                            f"Entity {node} will be updated with new source_id: {new_source_id}"
                        )

            # Process relationships
            for src, tgt, data in edges:
                if "source_id" in data:
                    # Split source_id using GRAPH_FIELD_SEP
                    sources = set(data["source_id"].split(GRAPH_FIELD_SEP))
                    sources.difference_update(chunk_ids)
                    if not sources:
                        relationships_to_delete.add((src, tgt))
                        logging.debug(
                            f"Relationship {src}-{tgt} marked for deletion - no remaining sources"
                        )
                    else:
                        new_source_id = GRAPH_FIELD_SEP.join(sources)
                        relationships_to_update[(src, tgt)] = new_source_id
                        logging.debug(
                            f"Relationship {src}-{tgt} will be updated with new source_id: {new_source_id}"
                        )

            # Delete entities
            if entities_to_delete:
                for entity in entities_to_delete:
                    await self.entities_vdb.delete_entity(entity)
                    logging.debug(f"Deleted entity {entity} from vector DB")
                self.chunk_entity_relation_graph.remove_nodes(list(entities_to_delete))
                logging.debug(f"Deleted {len(entities_to_delete)} entities from graph")

            # Update entities
            for entity, new_source_id in entities_to_update.items():
                node_data = self.chunk_entity_relation_graph._graph.nodes[entity]
                node_data["source_id"] = new_source_id
                await self.chunk_entity_relation_graph.upsert_node(entity, node_data)
                logging.debug(
                    f"Updated entity {entity} with new source_id: {new_source_id}"
                )

            # Delete relationships
            if relationships_to_delete:
                for src, tgt in relationships_to_delete:
                    rel_id_0 = compute_mdhash_id(src + tgt, prefix="rel-")
                    rel_id_1 = compute_mdhash_id(tgt + src, prefix="rel-")
                    await self.relationships_vdb.delete([rel_id_0, rel_id_1])
                    logging.debug(f"Deleted relationship {src}-{tgt} from vector DB")
                self.chunk_entity_relation_graph.remove_edges(
                    list(relationships_to_delete)
                )
                logging.debug(
                    f"Deleted {len(relationships_to_delete)} relationships from graph"
                )

            # Update relationships
            for (src, tgt), new_source_id in relationships_to_update.items():
                edge_data = self.chunk_entity_relation_graph._graph.edges[src, tgt]
                edge_data["source_id"] = new_source_id
                await self.chunk_entity_relation_graph.upsert_edge(src, tgt, edge_data)
                logging.debug(
                    f"Updated relationship {src}-{tgt} with new source_id: {new_source_id}"
                )

            # 6. Delete original document and status
            await self.full_docs.delete([doc_id])
            await self.doc_status.delete([doc_id])

            # 7. Ensure all indexes are updated
            await self._insert_done()

            logging.info(
                f"Successfully deleted document {doc_id} and related data. "
                f"Deleted {len(entities_to_delete)} entities and {len(relationships_to_delete)} relationships. "
                f"Updated {len(entities_to_update)} entities and {len(relationships_to_update)} relationships."
            )

            # Add verification step
            async def verify_deletion():
                # Verify if the document has been deleted
                if await self.full_docs.get_by_id(doc_id):
                    logging.error(f"Document {doc_id} still exists in full_docs")

                # Verify if chunks have been deleted
                remaining_chunks = await self.text_chunks.get_by_id(doc_id)
                if remaining_chunks:
                    logging.error(f"Found {len(remaining_chunks)} remaining chunks")

                # Verify entities and relationships
                for chunk_id in chunk_ids:
                    # Check entities
                    entities_with_chunk = [
                        dp
                        for dp in self.entities_vdb.client_storage["data"]
                        if chunk_id
                        in (dp.get("source_id") or "").split(GRAPH_FIELD_SEP)
                    ]
                    if entities_with_chunk:
                        logging.error(
                            f"Found {len(entities_with_chunk)} entities still referencing chunk {chunk_id}"
                        )

                    # Check relationships
                    relations_with_chunk = [
                        dp
                        for dp in self.relationships_vdb.client_storage["data"]
                        if chunk_id
                        in (dp.get("source_id") or "").split(GRAPH_FIELD_SEP)
                    ]
                    if relations_with_chunk:
                        logging.error(
                            f"Found {len(relations_with_chunk)} relations still referencing chunk {chunk_id}"
                        )

            await verify_deletion()

        except Exception as e:
            logging.error(f"Error while deleting document {doc_id}: {e}")

    async def get_entity_info(
        self, entity_name: str, include_vector_data: bool = False
    ) -> dict[str, str | None | dict[str, str]]:
        """Get detailed information of an entity

        Args:
            entity_name: Entity name (no need for quotes)
            include_vector_data: Whether to include data from the vector database

        Returns:
            dict: A dictionary containing entity information, including:
                - entity_name: Entity name
                - source_id: Source document ID
                - graph_data: Complete node data from the graph database
                - vector_data: (optional) Data from the vector database
        """
        entity_name = f'"{entity_name.upper()}"'

        # Get information from the graph
        node_data = await self.chunk_entity_relation_graph.get_node(entity_name)
        source_id = node_data.get("source_id") if node_data else None

        result: dict[str, str | None | dict[str, str]] = {
            "entity_name": entity_name,
            "source_id": source_id,
            "graph_data": node_data,
        }

        # Optional: Get vector database information
        if include_vector_data:
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            vector_data = self.entities_vdb.client.get([entity_id])
            result["vector_data"] = vector_data[0] if vector_data else None

        return result

    async def get_relation_info(
        self, src_entity: str, tgt_entity: str, include_vector_data: bool = False
    ) -> dict[str, str | None | dict[str, str]]:
        """Get detailed information of a relationship

        Args:
            src_entity: Source entity name (no need for quotes)
            tgt_entity: Target entity name (no need for quotes)
            include_vector_data: Whether to include data from the vector database

        Returns:
            dict: A dictionary containing relationship information, including:
                - src_entity: Source entity name
                - tgt_entity: Target entity name
                - source_id: Source document ID
                - graph_data: Complete edge data from the graph database
                - vector_data: (optional) Data from the vector database
        """
        src_entity = f'"{src_entity.upper()}"'
        tgt_entity = f'"{tgt_entity.upper()}"'

        # Get information from the graph
        edge_data = await self.chunk_entity_relation_graph.get_edge(
            src_entity, tgt_entity
        )
        source_id = edge_data.get("source_id") if edge_data else None

        result: dict[str, str | None | dict[str, str]] = {
            "src_entity": src_entity,
            "tgt_entity": tgt_entity,
            "source_id": source_id,
            "graph_data": edge_data,
        }

        # Optional: Get vector database information
        if include_vector_data:
            rel_id = compute_mdhash_id(src_entity + tgt_entity, prefix="rel-")
            vector_data = self.relationships_vdb._client.get([rel_id])
            result["vector_data"] = vector_data[0] if vector_data else None

        return result

    def check_storage_env_vars(self, storage_name: str) -> None:
        """Check if all required environment variables for storage implementation exist

        Args:
            storage_name: Storage implementation name

        Raises:
            ValueError: If required environment variables are missing
        """
        required_vars = STORAGE_ENV_REQUIREMENTS.get(storage_name, [])
        missing_vars = [var for var in required_vars if var not in os.environ]

        if missing_vars:
            raise ValueError(
                f"Storage implementation '{storage_name}' requires the following "
                f"environment variables: {', '.join(missing_vars)}"
            )
