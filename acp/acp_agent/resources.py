# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from contextlib import suppress
from datetime import timedelta

from acp_sdk import ResourceLoader, ResourceStore, ResourceUrl
from acp_sdk.models import Message
from acp_sdk.models.types import ResourceId
from granite_core.logging import get_logger
from obstore.store import ObjectStore, S3Store
from pydantic import ValidationError

from acp_agent.cache import AsyncLRUCache
from acp_agent.config import settings

logger = get_logger(__name__)


class AsyncCachingResourceLoader(ResourceLoader):
    cache: AsyncLRUCache[str, bytes] = AsyncLRUCache(max_size=200)

    async def load(self, url: ResourceUrl) -> bytes:  # type: ignore[override]
        key = str(url)

        value = await self.cache.get(key)
        if value is not None:
            return value

        response = await self._client.get(key)
        response.raise_for_status()
        value = await response.aread()

        await self.cache.set(key, value)
        return value


class ResourceStoreFactory:
    @staticmethod
    def create() -> ResourceStore | None:
        if (
            settings.RESOURCE_STORE_PROVIDER == "S3"
            and settings.S3_BUCKET is not None
            and settings.S3_ENDPOINT is not None
            and settings.S3_ACCESS_KEY_ID is not None
            and settings.S3_SECRET_ACCESS_KEY
        ):
            logger.info("Found a valid S3 RESOURCE_STORE_PROVIDER")
            return CompressingResourceStore(
                store=S3Store(
                    bucket=settings.S3_BUCKET,
                    endpoint=settings.S3_ENDPOINT,
                    access_key_id=settings.S3_ACCESS_KEY_ID.get_secret_value(),
                    secret_access_key=settings.S3_SECRET_ACCESS_KEY.get_secret_value(),
                )
            )
        return None


class CompressingResourceStore(ResourceStore):
    def __init__(self, *, store: ObjectStore, presigned_url_expiration: timedelta = timedelta(days=7)) -> None:
        self._store = store
        self._presigned_url_expiration = presigned_url_expiration

    async def store(
        self,
        id: ResourceId,
        data: bytes,
    ) -> None:
        # Compress outgoing messages
        with suppress(ValidationError):
            message = Message.model_validate_json(data)
            data = message.model_dump_json().encode()

        await self._store.put_async(str(id), data)
