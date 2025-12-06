"""
CineMatch V2.1.6 - External Model State Manager

Provides external state storage for trained models to support horizontal scaling.
Replaces st.session_state singleton pattern with distributed-friendly storage.

Backends supported:
- InMemory: Development/testing (not for multi-instance)
- Redis: Production multi-instance deployments
- S3: Long-term model persistence

Author: CineMatch Development Team
Date: November 2025
"""

import os
import io
import gc
import json
import time
import pickle
import hashlib
import logging
import threading
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from src.algorithms.base_recommender import BaseRecommender

logger = logging.getLogger(__name__)

T = TypeVar('T')


def _get_available_memory_mb() -> float:
    """
    Get available system memory in MB.
    
    Returns:
        Available memory in MB, or -1 if cannot determine.
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.available / (1024 * 1024)
    except ImportError:
        # psutil not available - try platform-specific methods
        try:
            if sys.platform == 'linux':
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemAvailable:'):
                            return int(line.split()[1]) / 1024  # Convert KB to MB
            elif sys.platform == 'win32':
                import ctypes
                kernel32 = ctypes.windll.kernel32
                c_ulong = ctypes.c_ulong
                class MEMORYSTATUS(ctypes.Structure):
                    _fields_ = [
                        ('dwLength', c_ulong),
                        ('dwMemoryLoad', c_ulong),
                        ('dwTotalPhys', c_ulong),
                        ('dwAvailPhys', c_ulong),
                        ('dwTotalPageFile', c_ulong),
                        ('dwAvailPageFile', c_ulong),
                        ('dwTotalVirtual', c_ulong),
                        ('dwAvailVirtual', c_ulong),
                    ]
                memStatus = MEMORYSTATUS()
                memStatus.dwLength = ctypes.sizeof(MEMORYSTATUS)
                kernel32.GlobalMemoryStatus(ctypes.byref(memStatus))
                return memStatus.dwAvailPhys / (1024 * 1024)
        except Exception:
            pass
    return -1  # Unknown


def _log_memory_usage(operation: str, phase: str = "") -> None:
    """
    Log current memory usage for debugging.
    
    Args:
        operation: Name of the operation being performed
        phase: Phase of the operation (e.g., 'before', 'after')
    """
    available_mb = _get_available_memory_mb()
    if available_mb > 0:
        phase_str = f" ({phase})" if phase else ""
        logger.info(f"Memory{phase_str} - {operation}: {available_mb:.0f}MB available")
    

class StorageBackendType(Enum):
    """Available storage backend types."""
    MEMORY = "memory"
    REDIS = "redis"
    S3 = "s3"
    FILE = "file"


@dataclass
class ModelStateConfig:
    """Configuration for model state storage."""
    backend: StorageBackendType = StorageBackendType.MEMORY
    
    # General settings
    prefix: str = "cinematch:model:"
    ttl_seconds: int = 3600 * 24  # 24 hours default for models
    serializer: str = "pickle"  # "pickle" or "joblib"
    compression: bool = True
    
    # Redis settings
    redis_url: str = "redis://localhost:6379/0"
    redis_max_connections: int = 5
    redis_socket_timeout: float = 30.0  # Longer timeout for large models
    
    # S3 settings
    s3_bucket: str = "cinematch-models"
    s3_region: str = "us-east-1"
    s3_prefix: str = "trained-models/"
    
    # File settings
    file_path: str = "models/state/"
    
    @classmethod
    def from_env(cls) -> 'ModelStateConfig':
        """Create configuration from environment variables."""
        backend_str = os.getenv('CINEMATCH_MODEL_BACKEND', 'memory').lower()
        backend = StorageBackendType(backend_str) if backend_str in [e.value for e in StorageBackendType] else StorageBackendType.MEMORY
        
        return cls(
            backend=backend,
            prefix=os.getenv('CINEMATCH_MODEL_PREFIX', 'cinematch:model:'),
            ttl_seconds=int(os.getenv('CINEMATCH_MODEL_TTL', str(3600 * 24))),
            redis_url=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
            s3_bucket=os.getenv('CINEMATCH_MODEL_S3_BUCKET', 'cinematch-models'),
            s3_region=os.getenv('AWS_REGION', 'us-east-1'),
            file_path=os.getenv('CINEMATCH_MODEL_FILE_PATH', 'models/state/'),
        )


@dataclass
class ModelMetadata:
    """Metadata for a stored model."""
    model_type: str
    version: str
    trained_at: datetime
    training_time_seconds: float
    metrics: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""
    size_bytes: int = 0
    node_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_type': self.model_type,
            'version': self.version,
            'trained_at': self.trained_at.isoformat(),
            'training_time_seconds': self.training_time_seconds,
            'metrics': self.metrics,
            'parameters': self.parameters,
            'checksum': self.checksum,
            'size_bytes': self.size_bytes,
            'node_id': self.node_id,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ModelMetadata':
        return cls(
            model_type=d['model_type'],
            version=d['version'],
            trained_at=datetime.fromisoformat(d['trained_at']),
            training_time_seconds=d['training_time_seconds'],
            metrics=d.get('metrics', {}),
            parameters=d.get('parameters', {}),
            checksum=d.get('checksum', ''),
            size_bytes=d.get('size_bytes', 0),
            node_id=d.get('node_id'),
        )


class ModelStateBackend(ABC):
    """Abstract base class for model state storage backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[bytes]:
        """Retrieve serialized model data by key."""
        pass
    
    @abstractmethod
    def set(self, key: str, data: bytes, metadata: ModelMetadata, ttl: Optional[int] = None) -> bool:
        """Store serialized model data."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a model."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a model exists."""
        pass
    
    @abstractmethod
    def get_metadata(self, key: str) -> Optional[ModelMetadata]:
        """Get metadata for a model without loading the full model."""
        pass
    
    @abstractmethod
    def list_models(self, pattern: str = "*") -> List[str]:
        """List all stored models matching pattern."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics."""
        pass


class InMemoryModelBackend(ModelStateBackend):
    """
    In-memory model storage for development and single-instance deployments.
    
    WARNING: Models stored here are NOT shared across instances.
    Use Redis or S3 for production multi-instance deployments.
    """
    
    def __init__(self, config: Optional[ModelStateConfig] = None):
        self.config = config or ModelStateConfig()
        self._models: Dict[str, bytes] = {}
        self._metadata: Dict[str, ModelMetadata] = {}
        self._expiry: Dict[str, float] = {}
        self._lock = threading.RLock()
        logger.info("Initialized in-memory model state backend (single-instance only)")
    
    def _make_key(self, key: str) -> str:
        return f"{self.config.prefix}{key}"
    
    def _is_expired(self, key: str) -> bool:
        full_key = self._make_key(key)
        if full_key not in self._expiry:
            return False
        return time.time() > self._expiry[full_key]
    
    def get(self, key: str) -> Optional[bytes]:
        with self._lock:
            if self._is_expired(key):
                self.delete(key)
                return None
            full_key = self._make_key(key)
            return self._models.get(full_key)
    
    def set(self, key: str, data: bytes, metadata: ModelMetadata, ttl: Optional[int] = None) -> bool:
        with self._lock:
            full_key = self._make_key(key)
            self._models[full_key] = data
            self._metadata[full_key] = metadata
            
            effective_ttl = ttl or self.config.ttl_seconds
            self._expiry[full_key] = time.time() + effective_ttl
            
            logger.debug(f"Stored model '{key}' ({len(data)} bytes) in memory")
            return True
    
    def delete(self, key: str) -> bool:
        with self._lock:
            full_key = self._make_key(key)
            deleted = False
            if full_key in self._models:
                del self._models[full_key]
                deleted = True
            if full_key in self._metadata:
                del self._metadata[full_key]
            if full_key in self._expiry:
                del self._expiry[full_key]
            return deleted
    
    def exists(self, key: str) -> bool:
        if self._is_expired(key):
            self.delete(key)
            return False
        return self._make_key(key) in self._models
    
    def get_metadata(self, key: str) -> Optional[ModelMetadata]:
        with self._lock:
            if self._is_expired(key):
                self.delete(key)
                return None
            return self._metadata.get(self._make_key(key))
    
    def list_models(self, pattern: str = "*") -> List[str]:
        with self._lock:
            import fnmatch
            prefix_len = len(self.config.prefix)
            keys = []
            for full_key in self._models.keys():
                key = full_key[prefix_len:] if full_key.startswith(self.config.prefix) else full_key
                if pattern == "*" or fnmatch.fnmatch(key, pattern):
                    keys.append(key)
            return keys
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total_size = sum(len(d) for d in self._models.values())
            return {
                'backend': 'in_memory',
                'model_count': len(self._models),
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
            }


class RedisModelBackend(ModelStateBackend):
    """
    Redis-based model storage for production multi-instance deployments.
    
    Stores serialized models in Redis with metadata in a separate key.
    Supports TTL for automatic cleanup of unused models.
    
    Requires redis-py package: pip install redis
    """
    
    def __init__(self, config: Optional[ModelStateConfig] = None):
        self.config = config or ModelStateConfig.from_env()
        self._client = None
        self._pool = None
        self._connected = False
    
    def _get_client(self):
        """Get or create Redis client."""
        if self._client is not None:
            return self._client
        
        try:
            import redis  # type: ignore[import-not-found]
            
            self._pool = redis.ConnectionPool.from_url(
                self.config.redis_url,
                max_connections=self.config.redis_max_connections,
                socket_timeout=self.config.redis_socket_timeout,
            )
            self._client = redis.Redis(connection_pool=self._pool)
            
            # Test connection
            self._client.ping()
            self._connected = True
            logger.info(f"Connected to Redis for model storage at {self.config.redis_url}")
            
            return self._client
        except ImportError:
            logger.error("redis-py package not installed. Run: pip install redis")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _make_key(self, key: str) -> str:
        return f"{self.config.prefix}{key}"
    
    def _metadata_key(self, key: str) -> str:
        return f"{self.config.prefix}{key}:metadata"
    
    def get(self, key: str) -> Optional[bytes]:
        try:
            client = self._get_client()
            data = client.get(self._make_key(key))
            if data:
                logger.debug(f"Retrieved model '{key}' from Redis ({len(data)} bytes)")
            return data
        except Exception as e:
            logger.error(f"Failed to get model '{key}' from Redis: {e}")
            return None
    
    def set(self, key: str, data: bytes, metadata: ModelMetadata, ttl: Optional[int] = None) -> bool:
        try:
            client = self._get_client()
            effective_ttl = ttl or self.config.ttl_seconds
            
            # Store model data
            model_key = self._make_key(key)
            client.setex(model_key, effective_ttl, data)
            
            # Store metadata separately (smaller, faster to read)
            meta_key = self._metadata_key(key)
            client.setex(meta_key, effective_ttl, json.dumps(metadata.to_dict()))
            
            logger.info(f"Stored model '{key}' in Redis ({len(data)} bytes, TTL={effective_ttl}s)")
            return True
        except Exception as e:
            logger.error(f"Failed to store model '{key}' in Redis: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        try:
            client = self._get_client()
            result = client.delete(self._make_key(key), self._metadata_key(key))
            return result > 0
        except Exception as e:
            logger.error(f"Failed to delete model '{key}' from Redis: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        try:
            client = self._get_client()
            return client.exists(self._make_key(key)) > 0
        except Exception as e:
            logger.error(f"Failed to check model '{key}' existence in Redis: {e}")
            return False
    
    def get_metadata(self, key: str) -> Optional[ModelMetadata]:
        try:
            client = self._get_client()
            meta_data = client.get(self._metadata_key(key))
            if meta_data:
                return ModelMetadata.from_dict(json.loads(meta_data))
            return None
        except Exception as e:
            logger.error(f"Failed to get metadata for model '{key}' from Redis: {e}")
            return None
    
    def list_models(self, pattern: str = "*") -> List[str]:
        try:
            client = self._get_client()
            prefix_len = len(self.config.prefix)
            search_pattern = f"{self.config.prefix}{pattern}"
            
            # Exclude metadata keys
            keys = []
            for key in client.scan_iter(match=search_pattern):
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                if not key_str.endswith(':metadata'):
                    keys.append(key_str[prefix_len:])
            return keys
        except Exception as e:
            logger.error(f"Failed to list models from Redis: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        try:
            client = self._get_client()
            keys = self.list_models()
            total_size = 0
            for key in keys:
                size = client.strlen(self._make_key(key))
                total_size += size
            
            return {
                'backend': 'redis',
                'connected': self._connected,
                'model_count': len(keys),
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
            }
        except Exception as e:
            logger.error(f"Failed to get Redis stats: {e}")
            return {'backend': 'redis', 'connected': False, 'error': str(e)}


class S3ModelBackend(ModelStateBackend):
    """
    S3-based model storage for long-term persistence and large models.
    
    Ideal for:
    - Persisting models across deployments
    - Storing very large models (>100MB)
    - Cross-region model sharing
    
    Requires boto3 package: pip install boto3
    """
    
    def __init__(self, config: Optional[ModelStateConfig] = None):
        self.config = config or ModelStateConfig.from_env()
        self._client = None
    
    def _get_client(self):
        """Get or create S3 client."""
        if self._client is not None:
            return self._client
        
        try:
            import boto3  # type: ignore[import-not-found]
            
            self._client = boto3.client('s3', region_name=self.config.s3_region)
            logger.info(f"Connected to S3 bucket '{self.config.s3_bucket}'")
            return self._client
        except ImportError:
            logger.error("boto3 package not installed. Run: pip install boto3")
            raise
        except Exception as e:
            logger.error(f"Failed to create S3 client: {e}")
            raise
    
    def _make_key(self, key: str) -> str:
        return f"{self.config.s3_prefix}{key}.pkl"
    
    def _metadata_key(self, key: str) -> str:
        return f"{self.config.s3_prefix}{key}.metadata.json"
    
    def get(self, key: str) -> Optional[bytes]:
        try:
            client = self._get_client()
            response = client.get_object(Bucket=self.config.s3_bucket, Key=self._make_key(key))
            data = response['Body'].read()
            logger.debug(f"Retrieved model '{key}' from S3 ({len(data)} bytes)")
            return data
        except client.exceptions.NoSuchKey:
            return None
        except Exception as e:
            logger.error(f"Failed to get model '{key}' from S3: {e}")
            return None
    
    def set(self, key: str, data: bytes, metadata: ModelMetadata, ttl: Optional[int] = None) -> bool:
        try:
            client = self._get_client()
            
            # Store model data
            client.put_object(
                Bucket=self.config.s3_bucket,
                Key=self._make_key(key),
                Body=data,
                Metadata={
                    'model_type': metadata.model_type,
                    'version': metadata.version,
                    'checksum': metadata.checksum,
                }
            )
            
            # Store metadata separately
            client.put_object(
                Bucket=self.config.s3_bucket,
                Key=self._metadata_key(key),
                Body=json.dumps(metadata.to_dict()),
                ContentType='application/json'
            )
            
            logger.info(f"Stored model '{key}' in S3 ({len(data)} bytes)")
            return True
        except Exception as e:
            logger.error(f"Failed to store model '{key}' in S3: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        try:
            client = self._get_client()
            client.delete_objects(
                Bucket=self.config.s3_bucket,
                Delete={
                    'Objects': [
                        {'Key': self._make_key(key)},
                        {'Key': self._metadata_key(key)},
                    ]
                }
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete model '{key}' from S3: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        try:
            client = self._get_client()
            client.head_object(Bucket=self.config.s3_bucket, Key=self._make_key(key))
            return True
        except client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            raise
        except Exception as e:
            logger.error(f"Failed to check model '{key}' existence in S3: {e}")
            return False
    
    def get_metadata(self, key: str) -> Optional[ModelMetadata]:
        try:
            client = self._get_client()
            response = client.get_object(Bucket=self.config.s3_bucket, Key=self._metadata_key(key))
            meta_data = response['Body'].read()
            return ModelMetadata.from_dict(json.loads(meta_data))
        except Exception as e:
            logger.error(f"Failed to get metadata for model '{key}' from S3: {e}")
            return None
    
    def list_models(self, pattern: str = "*") -> List[str]:
        try:
            client = self._get_client()
            response = client.list_objects_v2(
                Bucket=self.config.s3_bucket,
                Prefix=self.config.s3_prefix
            )
            
            keys = []
            for obj in response.get('Contents', []):
                key = obj['Key']
                if key.endswith('.pkl') and not key.endswith('.metadata.json'):
                    # Extract model name from path
                    model_key = key[len(self.config.s3_prefix):-4]  # Remove prefix and .pkl
                    keys.append(model_key)
            return keys
        except Exception as e:
            logger.error(f"Failed to list models from S3: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        try:
            client = self._get_client()
            response = client.list_objects_v2(
                Bucket=self.config.s3_bucket,
                Prefix=self.config.s3_prefix
            )
            
            total_size = 0
            model_count = 0
            for obj in response.get('Contents', []):
                if obj['Key'].endswith('.pkl'):
                    total_size += obj['Size']
                    model_count += 1
            
            return {
                'backend': 's3',
                'bucket': self.config.s3_bucket,
                'model_count': model_count,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
            }
        except Exception as e:
            logger.error(f"Failed to get S3 stats: {e}")
            return {'backend': 's3', 'error': str(e)}


class FileModelBackend(ModelStateBackend):
    """
    File-based model storage for local persistence.
    
    Suitable for:
    - Single-node deployments with persistent storage
    - Development with model persistence
    - Docker volumes
    """
    
    def __init__(self, config: Optional[ModelStateConfig] = None):
        self.config = config or ModelStateConfig()
        self._base_path = Path(self.config.file_path)
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        logger.info(f"Initialized file model backend at {self._base_path}")
    
    def _model_path(self, key: str) -> Path:
        return self._base_path / f"{key}.pkl"
    
    def _metadata_path(self, key: str) -> Path:
        return self._base_path / f"{key}.metadata.json"
    
    def get(self, key: str) -> Optional[bytes]:
        with self._lock:
            path = self._model_path(key)
            if not path.exists():
                return None
            return path.read_bytes()
    
    def set(self, key: str, data: bytes, metadata: ModelMetadata, ttl: Optional[int] = None) -> bool:
        with self._lock:
            try:
                self._model_path(key).write_bytes(data)
                self._metadata_path(key).write_text(json.dumps(metadata.to_dict()))
                logger.info(f"Stored model '{key}' to file ({len(data)} bytes)")
                return True
            except Exception as e:
                logger.error(f"Failed to store model '{key}' to file: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        with self._lock:
            deleted = False
            model_path = self._model_path(key)
            meta_path = self._metadata_path(key)
            if model_path.exists():
                model_path.unlink()
                deleted = True
            if meta_path.exists():
                meta_path.unlink()
            return deleted
    
    def exists(self, key: str) -> bool:
        return self._model_path(key).exists()
    
    def get_metadata(self, key: str) -> Optional[ModelMetadata]:
        meta_path = self._metadata_path(key)
        if not meta_path.exists():
            return None
        try:
            data = json.loads(meta_path.read_text())
            return ModelMetadata.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to read metadata for '{key}': {e}")
            return None
    
    def list_models(self, pattern: str = "*") -> List[str]:
        import fnmatch
        keys = []
        for path in self._base_path.glob("*.pkl"):
            key = path.stem
            if pattern == "*" or fnmatch.fnmatch(key, pattern):
                keys.append(key)
        return keys
    
    def get_stats(self) -> Dict[str, Any]:
        total_size = 0
        model_count = 0
        for path in self._base_path.glob("*.pkl"):
            total_size += path.stat().st_size
            model_count += 1
        
        return {
            'backend': 'file',
            'path': str(self._base_path),
            'model_count': model_count,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
        }


class ModelStateManager:
    """
    High-level manager for model state storage.
    
    Features:
    - Automatic backend selection based on configuration
    - Model serialization/deserialization
    - Compression support
    - Checksum verification
    - Thread-safe operations
    
    Usage:
        # Initialize (auto-detects backend from environment)
        state = ModelStateManager.get_instance()
        
        # Store a trained model
        state.save_model('svd', svd_model, training_time=45.2)
        
        # Load a model
        model = state.load_model('svd')
        
        # Check if model exists
        if state.has_model('svd'):
            metadata = state.get_model_info('svd')
    """
    
    _instance: Optional['ModelStateManager'] = None
    _lock = threading.Lock()
    
    def __init__(self, config: Optional[ModelStateConfig] = None):
        self.config = config or ModelStateConfig.from_env()
        self._backend = self._create_backend()
        self._local_cache: Dict[str, Any] = {}  # L1 cache for frequent access
        self._cache_lock = threading.RLock()
        self._node_id = os.getenv('CINEMATCH_INSTANCE_ID', f"node_{int(time.time())}")
        logger.info(f"ModelStateManager initialized with {self.config.backend.value} backend")
    
    @classmethod
    def get_instance(cls, config: Optional[ModelStateConfig] = None) -> 'ModelStateManager':
        """Get singleton instance of ModelStateManager."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config)
            return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None
    
    def _create_backend(self) -> ModelStateBackend:
        """Create the appropriate backend based on configuration."""
        backend_map = {
            StorageBackendType.MEMORY: InMemoryModelBackend,
            StorageBackendType.REDIS: RedisModelBackend,
            StorageBackendType.S3: S3ModelBackend,
            StorageBackendType.FILE: FileModelBackend,
        }
        
        backend_class = backend_map.get(self.config.backend, InMemoryModelBackend)
        return backend_class(self.config)
    
    def _estimate_model_size(self, model: Any) -> int:
        """
        Estimate the serialized size of a model without full serialization.
        
        Uses sys.getsizeof recursively for a rough estimate.
        This is faster than full serialization but may underestimate.
        
        Args:
            model: The model to estimate size for
            
        Returns:
            Estimated size in bytes
        """
        import sys
        
        def get_size(obj, seen=None):
            """Recursively calculate object size."""
            size = sys.getsizeof(obj)
            if seen is None:
                seen = set()
            obj_id = id(obj)
            if obj_id in seen:
                return 0
            seen.add(obj_id)
            
            if isinstance(obj, dict):
                size += sum([get_size(v, seen) for v in obj.values()])
                size += sum([get_size(k, seen) for k in obj.keys()])
            elif hasattr(obj, '__dict__'):
                size += get_size(obj.__dict__, seen)
            elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
                try:
                    size += sum([get_size(i, seen) for i in obj])
                except TypeError:
                    pass
            return size
        
        try:
            # Quick estimate - actual serialized size is usually 1.5-3x this
            estimated = get_size(model)
            # Apply compression ratio estimate (zlib typically achieves 2-3x compression)
            if self.config.compression:
                estimated = int(estimated * 0.4)  # Compressed estimate
            return estimated
        except Exception:
            return -1  # Unknown
    
    def _serialize_model(self, model: Any, chunk_size_mb: float = 100.0) -> bytes:
        """
        Serialize a model to bytes with chunked processing for large models.
        
        Args:
            model: The model to serialize
            chunk_size_mb: Chunk size for streaming serialization (default 100MB)
            
        Returns:
            Serialized bytes (potentially compressed)
        """
        import zlib
        
        if self.config.serializer == 'joblib':
            try:
                import joblib  # type: ignore[import-not-found]
                buffer = io.BytesIO()
                joblib.dump(model, buffer)
                data = buffer.getvalue()
            except ImportError:
                # Fallback to pickle
                data = pickle.dumps(model)
        else:
            data = pickle.dumps(model)
        
        if self.config.compression:
            # For large data, use streaming compression to reduce memory spike
            chunk_size = int(chunk_size_mb * 1024 * 1024)
            if len(data) > chunk_size:
                # Streaming compression for large models
                compressor = zlib.compressobj(level=6)
                compressed_chunks = []
                for i in range(0, len(data), chunk_size):
                    chunk = data[i:i + chunk_size]
                    compressed_chunks.append(compressor.compress(chunk))
                compressed_chunks.append(compressor.flush())
                data = b''.join(compressed_chunks)
                logger.debug(f"Used streaming compression for large model ({len(data)/(1024*1024):.1f}MB)")
            else:
                data = zlib.compress(data, level=6)
        
        return data
    
    def _deserialize_model(self, data: bytes) -> Any:
        """Deserialize a model from bytes."""
        import zlib
        
        if self.config.compression:
            try:
                data = zlib.decompress(data)
            except zlib.error:
                pass  # Data might not be compressed
        
        if self.config.serializer == 'joblib':
            try:
                import joblib  # type: ignore[import-not-found]
                buffer = io.BytesIO(data)
                return joblib.load(buffer)
            except ImportError:
                pass
        
        return pickle.loads(data)
    
    def _compute_checksum(self, data: bytes) -> str:
        """Compute SHA256 checksum of data."""
        return hashlib.sha256(data).hexdigest()[:16]
    
    def save_model(
        self,
        key: str,
        model: Any,
        model_type: str = "unknown",
        version: str = "1.0.0",
        training_time: float = 0.0,
        metrics: Optional[Dict[str, float]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
        cache_locally: bool = True,
        max_size_mb: float = 500.0
    ) -> bool:
        """
        Save a trained model to external storage.
        
        Args:
            key: Unique identifier for the model
            model: The model object to store
            model_type: Type of model (e.g., 'SVD', 'UserKNN')
            version: Model version
            training_time: Training duration in seconds
            metrics: Performance metrics dict
            parameters: Model hyperparameters
            ttl: Time-to-live in seconds (None = use default)
            cache_locally: Whether to keep in L1 cache
            max_size_mb: Maximum model size in MB (default 500MB to prevent memory exhaustion)
            
        Returns:
            True if saved successfully
        """
        try:
            # Log memory before save operation
            _log_memory_usage(f"save_model({key})", "before")
            
            # Check available memory before serialization (estimate 2x model size needed)
            available_mb = _get_available_memory_mb()
            if available_mb > 0 and available_mb < 500:
                logger.warning(
                    f"Low memory warning: {available_mb:.0f}MB available. "
                    f"Skipping serialization for model '{key}' to prevent crash."
                )
                # Cache locally only (already in memory)
                if cache_locally:
                    with self._cache_lock:
                        self._local_cache[key] = model
                return False
            
            # Pre-check: Estimate model size before serialization
            estimated_size = self._estimate_model_size(model)
            max_size_bytes = int(max_size_mb * 1024 * 1024)
            if estimated_size > 0 and estimated_size > max_size_bytes * 2:
                logger.warning(
                    f"Model '{key}' estimated size ({estimated_size / (1024*1024):.1f}MB) "
                    f"likely exceeds limit ({max_size_mb}MB). Skipping serialization attempt."
                )
                if cache_locally:
                    with self._cache_lock:
                        self._local_cache[key] = model
                return False
            
            # Serialize model
            data = self._serialize_model(model)
            
            # Check model size limit to prevent memory exhaustion
            if len(data) > max_size_bytes:
                logger.warning(
                    f"Model '{key}' size ({len(data) / (1024*1024):.1f}MB) exceeds limit "
                    f"({max_size_mb}MB). Skipping external storage save."
                )
                # Still cache locally since it's already in memory
                if cache_locally:
                    with self._cache_lock:
                        self._local_cache[key] = model
                return False
            
            checksum = self._compute_checksum(data)
            
            # Create metadata
            metadata = ModelMetadata(
                model_type=model_type,
                version=version,
                trained_at=datetime.now(),
                training_time_seconds=training_time,
                metrics=metrics or {},
                parameters=parameters or {},
                checksum=checksum,
                size_bytes=len(data),
                node_id=self._node_id,
            )
            
            # Store in backend
            success = self._backend.set(key, data, metadata, ttl)
            
            if success and cache_locally:
                with self._cache_lock:
                    self._local_cache[key] = model
            
            logger.info(f"Saved model '{key}' ({len(data)} bytes, checksum={checksum})")
            
            # Force garbage collection to free serialization buffers
            del data
            gc.collect()
            
            # Log memory after save operation
            _log_memory_usage(f"save_model({key})", "after")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to save model '{key}': {e}")
            gc.collect()  # Clean up any partial allocations
            return False
    
    def load_model(
        self,
        key: str,
        verify_checksum: bool = True,
        use_cache: bool = True
    ) -> Optional[Any]:
        """
        Load a model from external storage.
        
        Args:
            key: Model identifier
            verify_checksum: Whether to verify data integrity
            use_cache: Whether to check L1 cache first
            
        Returns:
            Deserialized model object or None if not found
        """
        # Log memory before load operation
        _log_memory_usage(f"load_model({key})", "before")
        
        # Check L1 cache first
        if use_cache:
            with self._cache_lock:
                if key in self._local_cache:
                    logger.debug(f"Model '{key}' loaded from L1 cache")
                    return self._local_cache[key]
        
        try:
            # Get from backend
            data = self._backend.get(key)
            if data is None:
                return None
            
            # Verify checksum if enabled
            if verify_checksum:
                metadata = self._backend.get_metadata(key)
                if metadata and metadata.checksum:
                    actual_checksum = self._compute_checksum(data)
                    if actual_checksum != metadata.checksum:
                        logger.error(f"Checksum mismatch for model '{key}'! "
                                   f"Expected {metadata.checksum}, got {actual_checksum}")
                        return None
            
            # Deserialize
            model = self._deserialize_model(data)
            
            # Cache locally
            if use_cache:
                with self._cache_lock:
                    self._local_cache[key] = model
            
            logger.info(f"Loaded model '{key}' ({len(data)} bytes)")
            
            # Log memory after load operation
            _log_memory_usage(f"load_model({key})", "after")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model '{key}': {e}")
            return None
    
    def has_model(self, key: str) -> bool:
        """Check if a model exists in storage."""
        # Check L1 cache first
        with self._cache_lock:
            if key in self._local_cache:
                return True
        
        return self._backend.exists(key)
    
    def delete_model(self, key: str) -> bool:
        """Delete a model from storage."""
        # Remove from L1 cache
        with self._cache_lock:
            self._local_cache.pop(key, None)
        
        return self._backend.delete(key)
    
    def get_model_info(self, key: str) -> Optional[ModelMetadata]:
        """Get metadata for a model without loading it."""
        return self._backend.get_metadata(key)
    
    def list_models(self, pattern: str = "*") -> List[str]:
        """List all stored models."""
        return self._backend.list_models(pattern)
    
    def clear_local_cache(self) -> None:
        """Clear the L1 local cache."""
        with self._cache_lock:
            self._local_cache.clear()
        logger.info("Cleared local model cache")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        backend_stats = self._backend.get_stats()
        with self._cache_lock:
            backend_stats['local_cache_size'] = len(self._local_cache)
        return backend_stats


# Convenience function for getting the singleton instance
def get_model_state() -> ModelStateManager:
    """Get the global ModelStateManager instance."""
    return ModelStateManager.get_instance()
