from acp_sdk import ResourceLoader, ResourceStore, ResourceUrl
from cachetools import LFUCache
from obstore.store import S3Store

from granite_chat.config import settings

cache: LFUCache = LFUCache(maxsize=1000)


class AsyncCachingResourceLoader(ResourceLoader):
    async def load(self, url: ResourceUrl) -> bytes:  # type: ignore[override]
        if url in cache:
            return cache[url]

        response = await self._client.get(str(url))
        response.raise_for_status()
        data = await response.aread()

        cache[url] = data
        return data


class ResourceStoreFactory:
    @staticmethod
    def create() -> ResourceStore | None:
        if settings.RESOURCE_STORE_PROVIDER == "S3":
            return ResourceStore(
                store=S3Store(
                    bucket=settings.S3_BUCKET,
                    endpoint=settings.S3_ENDPOINT,
                    access_key_id=settings.S3_ACCESS_KEY_ID,
                    secret_access_key=settings.S3_SECRET_ACCESS_KEY,
                )
            )
        return None
