from acp_sdk import ResourceLoader, ResourceStore, ResourceUrl
from cachetools import LFUCache
from obstore.store import S3Store

from granite_chat import get_logger
from granite_chat.config import settings

logger = get_logger(__name__)


class AsyncCachingResourceLoader(ResourceLoader):
    cache: LFUCache = LFUCache(maxsize=2000)

    async def load(self, url: ResourceUrl) -> bytes:  # type: ignore[override]
        if url in self.cache:
            return self.cache[url]

        response = await self._client.get(str(url))
        response.raise_for_status()
        data = await response.aread()

        self.cache[url] = data
        return data


class ResourceStoreFactory:
    @staticmethod
    def create() -> ResourceStore | None:
        if settings.RESOURCE_STORE_PROVIDER == "S3":
            logger.info("Found S3 RESOURCE_STORE_PROVIDER")
            return ResourceStore(
                store=S3Store(
                    bucket=settings.S3_BUCKET,
                    endpoint=settings.S3_ENDPOINT,
                    access_key_id=settings.S3_ACCESS_KEY_ID,
                    secret_access_key=settings.S3_SECRET_ACCESS_KEY,
                )
            )
        return None
