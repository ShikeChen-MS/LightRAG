import asyncio
import json
from ..lightrag import EmbeddingFunc
from io import BytesIO
from typing import Any, final
from azure.storage.blob import BlobServiceClient, BlobLeaseClient
from ..az_token_credential import LightRagTokenCredential
from ..base import (
    BaseKVStorage,
)
from ..utils import (
    load_json,
    logger,
)


@final
class JsonKVStorage(BaseKVStorage):
    def __init__(
        self,
        global_config: dict[str, Any],
        namespace: str,
        embedding_func: EmbeddingFunc,
    ):
        self._data = None
        self._lock = asyncio.Lock()
        self.global_config = global_config
        self.namespace = namespace
        self.embedding_func = embedding_func

    async def initialize(
        self,
        storage_account_url: str,
        storage_container_name: str,
        access_token: LightRagTokenCredential,
    ) -> None:
        lease = None
        blob_lease = None
        try:
            blob_client = BlobServiceClient(
                account_url=storage_account_url, credential=access_token
            )
            container_client = blob_client.get_container_client(storage_container_name)
            container_client.get_container_properties()  # this is to check if the container exists and authentication is valid
            lease: BlobLeaseClient = container_client.acquire_lease()
            blob_list = container_client.list_blob_names()
            blob_name = f"{self.global_config["working_dir"]}/data/kv_store_{self.namespace}.json"
            if not blob_name in blob_list:
                logger.info(f"Creating new kv_store_{self.namespace}.json")
                self._data: dict[str, Any] = {}
                json_data = json.dumps(self._data)
                json_bytes = BytesIO(json_data.encode("utf-8"))
                blob_client = container_client.get_blob_client(blob_name)
                # reach here means the file does not exist, so with overwrite=False
                # the operation should still succeed.
                blob_client.upload_blob(json_bytes, overwrite=False)
                return
            # to prevent the file from being modified while trying to read
            # we acquire a lease to make sure no ops is performing on the file
            # also we acquire a lease on the container to prevent the container from being deleted
            blob_client = container_client.get_blob_client(blob_name)
            blob_lease = blob_client.acquire_lease()
            content = blob_client.download_blob(lease=blob_lease).readall()
            blob_lease.release()
            blob_lease = None
            lease.release()
            lease = None
            content_str = content.decode("utf-8")
            self._data = json.loads(content_str)
        except Exception as e:
            logger.warning(f"Failed to load graph from Azure Blob Storage: {e}")
            raise
        finally:
            if lease:
                lease.release()
            if blob_lease:
                blob_lease.release()

    async def index_done_callback(
        self,
        storage_account_url: str,
        storage_container_name: str,
        access_token: LightRagTokenCredential,
    ) -> None:
        lease = None
        blob_lease = None
        try:
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
            blob_lease = blob_client.acquire_lease()
            async with self._lock:
                json_data = json.dumps(self._data)
            json_bytes = BytesIO(json_data.encode("utf-8"))
            blob_client.upload_blob(json_bytes, lease=blob_lease, overwrite=True)
            blob_lease.release()
            blob_lease = None
            lease.release()
            lease = None
        except Exception as e:
            logger.warning(f"Failed to upload graph to Azure Blob Storage: {e}")
            raise
        finally:
            if lease:
                lease.release()
            if blob_lease:
                blob_lease.release()

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

    async def clear(self):
        self._data = {}

    async def delete(
            self,
            storage_account_url: str,
            storage_container_name: str,
            access_token: LightRagTokenCredential,
            ids: list[str]
    ) -> None:
        for doc_id in ids:
            self._data.pop(doc_id, None)
        await self.index_done_callback(
            storage_account_url, storage_container_name, access_token
        )
