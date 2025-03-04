import asyncio
import json
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Union, final
from azure.storage.blob import BlobServiceClient, BlobLeaseClient
from ..az_token_credential import LightRagTokenCredential
from ..base import (
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
)
from ..utils import (
    logger,
)


@final
@dataclass
class JsonDocStatusStorage(DocStatusStorage):
    """JSON implementation of document status storage"""

    def __init__(self, global_config: dict[str, Any], namespace: str, **kwargs: Any):
        self._data = None
        self._lock = asyncio.Lock()
        self.namespace = namespace
        self.global_config = global_config

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

    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Return keys that should be processed (not in storage or not successfully processed)"""
        return set(keys) - set(self._data.keys())

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for id in ids:
            data = self._data.get(id, None)
            if data:
                result.append(data)
        return result

    async def get_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status"""
        counts = {status.value: 0 for status in DocStatus}
        for doc in self._data.values():
            counts[doc["status"]] += 1
        return counts

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents with a specific status"""
        result = {}
        for k, v in self._data.items():
            if v["status"] == status.value:
                try:
                    # Make a copy of the data to avoid modifying the original
                    data = v.copy()
                    # If content is missing, use content_summary as content
                    if "content" not in data and "content_summary" in data:
                        data["content"] = data["content_summary"]
                    result[k] = DocProcessingStatus(**data)
                except KeyError as e:
                    logger.error(f"Missing required field for document {k}: {e}")
                    continue
        return result

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
            with self._lock:
                blob_client.upload_blob(self._data, lease=lease, overwrite=True)
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

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.info(f"Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        self._data.update(data)
        await self.index_done_callback()

    async def get_by_id(self, id: str) -> Union[dict[str, Any], None]:
        return self._data.get(id)

    async def delete(self, doc_ids: list[str]):
        for doc_id in doc_ids:
            self._data.pop(doc_id, None)
        await self.index_done_callback()

    async def drop(self) -> None:
        """Drop the storage"""
        self._data.clear()
