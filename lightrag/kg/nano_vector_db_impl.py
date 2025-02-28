import asyncio
import os
from typing import Any, final
from dataclasses import dataclass
import numpy as np
from azure.storage.blob import BlobServiceClient, BlobLeaseClient
from lightrag.az_token_credential import LighRagTokenCredential
import time

from lightrag.utils import (
    logger,
    compute_mdhash_id,
)
import pipmaster as pm
from lightrag.base import (
    BaseVectorStorage,
)
from lightrag.kg.nanovectordbs import NanoVectorDB


@final
@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    def __init__(self):
        self._client = None
        self._save_lock = asyncio.Lock()

    def __post_init__(self):
        # Use global config value if specified, otherwise use default
        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )

    async def initialize(
            self,
            storage_account_url: str,
            storage_container_name: str,
            access_token: LighRagTokenCredential) -> None:
        try:
            blob_client = BlobServiceClient(
                account_url=storage_account_url, credential=access_token
            )
            container_client = blob_client.get_container_client(storage_container_name)
            container_client.get_container_properties() # this is to check if the container exists and authentication is valid
            # to prevent the file from being modified while trying to read
            # we acquire a lease to make sure no ops is performing on the file
            lease: BlobLeaseClient = container_client.acquire_lease()
            blob_list = container_client.list_blob_names()
            blob_name = f"{self.global_config["working_dir"]}/data/vdb_{self.namespace}.json"
            if not blob_name in blob_list:
                logger.info(f"Creating new vdb_{self.namespace}.json")
                self._client = NanoVectorDB(self.embedding_func.embedding_dim)
                return
            content = container_client.get_blob_client(blob_name).download_blob().readall()
            lease.release()
            content_str = content.decode('utf-8')
            self._client = NanoVectorDB(
                self.embedding_func.embedding_dim, storage_data=content_str
            )
        except Exception as e:
            logger.warning(f"Failed to load graph from Azure Blob Storage: {e}")
            raise

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.info(f"Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        current_time = time.time()
        list_data = [
            {
                "__id__": k,
                "__created_at__": current_time,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        embedding_tasks = [self.embedding_func(batch) for batch in batches]
        embeddings_list = await asyncio.gather(*embedding_tasks)

        embeddings = np.concatenate(embeddings_list)
        if len(embeddings) == len(list_data):
            for i, d in enumerate(list_data):
                d["__vector__"] = embeddings[i]
            results = self._client.upsert(datas=list_data)
            return results
        else:
            # sometimes the embedding is not returned correctly. just log it.
            logger.error(
                f"embedding is not 1-1 with data, {len(embeddings)} != {len(list_data)}"
            )

    async def query(self, query: str, top_k: int) -> list[dict[str, Any]]:
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        results = [
            {
                **dp,
                "id": dp["__id__"],
                "distance": dp["__metrics__"],
                "created_at": dp.get("__created_at__"),
            }
            for dp in results
        ]
        return results

    @property
    def client_storage(self):
        return getattr(self._client, "_NanoVectorDB__storage")

    async def delete(self, ids: list[str]):
        """Delete vectors with specified IDs

        Args:
            ids: List of vector IDs to be deleted
        """
        try:
            self._client.delete(ids)
            logger.info(
                f"Successfully deleted {len(ids)} vectors from {self.namespace}"
            )
        except Exception as e:
            logger.error(f"Error while deleting vectors from {self.namespace}: {e}")

    async def delete_entity(self, entity_name: str) -> None:
        try:
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            logger.debug(
                f"Attempting to delete entity {entity_name} with ID {entity_id}"
            )
            # Check if the entity exists
            if self._client.get([entity_id]):
                await self.delete([entity_id])
                logger.debug(f"Successfully deleted entity {entity_name}")
            else:
                logger.debug(f"Entity {entity_name} not found in storage")
        except Exception as e:
            logger.error(f"Error deleting entity {entity_name}: {e}")

    async def delete_entity_relation(self, entity_name: str) -> None:
        try:
            relations = [
                dp
                for dp in self.client_storage["data"]
                if dp["src_id"] == entity_name or dp["tgt_id"] == entity_name
            ]
            logger.debug(f"Found {len(relations)} relations for entity {entity_name}")
            ids_to_delete = [relation["__id__"] for relation in relations]

            if ids_to_delete:
                await self.delete(ids_to_delete)
                logger.debug(
                    f"Deleted {len(ids_to_delete)} relations for {entity_name}"
                )
            else:
                logger.debug(f"No relations found for entity {entity_name}")
        except Exception as e:
            logger.error(f"Error deleting relations for {entity_name}: {e}")

    async def index_done_callback(
            self,
            storage_account_url: str,
            storage_container_name: str,
            access_token: LighRagTokenCredential
    ) -> None:
        async with self._save_lock:
            json_data = self._client.save()
        blob_client = BlobServiceClient(
            account_url=storage_account_url, credential=access_token
        )
        container_client = blob_client.get_container_client(storage_container_name)
        # this is to check if the container exists and authentication is valid
        container_client.get_container_properties()
        # to protect file integrity and ensure complete upload
        # acquire lease on the container to prevent any other ops
        lease: BlobLeaseClient = container_client.acquire_lease()
        blob_name = f"{self.global_config["working_dir"]}/data/vdb_{self.namespace}.json"
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(json_data, lease=lease, overwrite=True)
        lease.release()
