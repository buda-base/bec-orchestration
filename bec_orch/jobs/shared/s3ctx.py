import asyncio
import contextlib
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import boto3
from botocore.config import Config as BotoConfig

logger = logging.getLogger(__name__)


class S3Context:
    """Holds global S3 settings + thread pool for async S3 operations.

    Uses boto3 (standard AWS SDK) with a thread pool for async operations.
    This is simpler and more reliable than aiobotocore.
    
    Credentials are resolved via the default credential chain:
    - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    - Shared credentials file (~/.aws/credentials)
    - IAM role (for EC2 instances via instance metadata service)
    """

    def __init__(self, cfg, global_sem: asyncio.Semaphore):
        self.cfg = cfg
        self.global_sem = global_sem
        self._client = None
        self._lock: Optional[asyncio.Lock] = None
        
        # Thread pool for S3 operations (boto3 is synchronous)
        max_workers = getattr(cfg, "bulk_prefetch_concurrency", 128)
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="s3")

    def _get_lock(self) -> asyncio.Lock:
        """Get or create the lock (must be in event loop context)."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def _ensure_client(self):
        """Lazily create the shared S3 client on first use (thread-safe)."""
        if self._client is not None:
            return
        
        async with self._get_lock():
            if self._client is not None:
                return
            
            # Configure boto3 for high concurrency
            boto_config = BotoConfig(
                max_pool_connections=200,  # High connection pool for parallel fetches
                retries={"max_attempts": 3, "mode": "adaptive"},
            )
            
            logger.info(f"[S3Context] Creating boto3 S3 client (region={self.cfg.s3_region})")
            
            # Create session with optional profile
            aws_profile = getattr(self.cfg, "aws_profile", None)
            if aws_profile and aws_profile != "default":
                session = boto3.Session(profile_name=aws_profile)
            else:
                session = boto3.Session()
            
            self._client = session.client(
                "s3",
                region_name=self.cfg.s3_region,
                config=boto_config,
            )

    async def close(self):
        """Close the S3 client and thread pool."""
        async with self._get_lock():
            if self._client is not None:
                self._client.close()
                self._client = None
                logger.info("[S3Context] Closed boto3 S3 client")
        
        self._executor.shutdown(wait=False)

    @contextlib.asynccontextmanager
    async def client(self):
        """Yield the shared S3 client for use in async context."""
        await self._ensure_client()
        yield self._client
    
    async def get_object(self, bucket: str, key: str) -> tuple[str, bytes]:
        """Async wrapper for S3 get_object using thread pool."""
        await self._ensure_client()
        
        loop = asyncio.get_event_loop()
        
        def _fetch():
            response = self._client.get_object(Bucket=bucket, Key=key)
            etag = response.get("ETag", "").strip('"')
            body = response["Body"].read()
            return etag, body
        
        return await loop.run_in_executor(self._executor, _fetch)