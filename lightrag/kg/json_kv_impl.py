import asyncio
import os
from dataclasses import dataclass
from typing import Any, final
from azure.storage.blob import BlobServiceClient, BlobLeaseClient
from lightrag.az_token_credential import LighRagTokenCredential
from lightrag.base import (
    BaseKVStorage,
)
from lightrag.utils import (
    load_json,
    logger,
    write_json,
)


@final
@dataclass
class JsonKVStorage(BaseKVStorage):
    def __init__(self):
        self._data = None
        self._lock = asyncio.Lock()

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
            lease: BlobLeaseClient = container_client.acquire_lease()
            blob_list = container_client.list_blob_names()
            blob_name = f"{self.global_config["working_dir"]}/data/kv_store_{self.namespace}.json"
            if not blob_name in blob_list:
                logger.info(f"Creating new kv_store_{self.namespace}.json")
                self._data: dict[str, Any] = {}
                return
            content = container_client.get_blob_client(blob_name).download_blob().readall()
            lease.release()
            content_str = content.decode('utf-8')
            self._data = load_json(content_str)
        except Exception as e:
            logger.warning(f"Failed to load graph from Azure Blob Storage: {e}")
            raise

    async def index_done_callback(
            self,
            storage_account_url: str,
            storage_container_name: str,
            access_token: LighRagTokenCredential
    ) -> None:
        blob_client = BlobServiceClient(
            account_url=storage_account_url, credential=access_token
        )
        container_client = blob_client.get_container_client(storage_container_name)
        # this is to check if the container exists and authentication is valid
        container_client.get_container_properties()
        # to protect file integrity and ensure complete upload
        # acquire lease on the container to prevent any other ops
        lease: BlobLeaseClient = container_client.acquire_lease()
        blob_name = f"{self.global_config["working_dir"]}/data/kv_store_{self.namespace}.json"
        blob_client = container_client.get_blob_client(blob_name)
        with self._lock:
            blob_client.upload_blob(self._data, lease=lease, overwrite=True)
        lease.release()

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        return self._data.get(id)

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        return [
            (
                {k: v for k, v in self._data[id].items()}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, keys: set[str]) -> set[str]:
        return set(keys) - set(self._data.keys())

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.info(f"Inserting {len(data)} to {self.namespace}")
        if not data:
            return
        left_data = {k: v for k, v in data.items() if k not in self._data}
        self._data.update(left_data)

    async def delete(self, ids: list[str]) -> None:
        for doc_id in ids:
            self._data.pop(doc_id, None)
        await self.index_done_callback()
