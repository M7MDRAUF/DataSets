"""
Object Pool Module for CineMatch V2.1.6

Implements object pooling for memory efficiency:
- Reusable object pools
- Pre-allocation
- Pool statistics
- Automatic cleanup

Phase 6 - Task 6.3: Object Pool
"""

import logging
import time
import threading
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class PoolConfig:
    """Object pool configuration."""
    initial_size: int = 10
    max_size: int = 100
    min_size: int = 5
    growth_factor: float = 2.0
    idle_timeout: float = 300.0  # seconds
    validation_interval: float = 60.0


@dataclass
class PooledObject(Generic[T]):
    """Wrapper for pooled objects."""
    obj: T
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    
    def touch(self) -> None:
        self.last_used = time.time()
        self.use_count += 1
    
    @property
    def idle_time(self) -> float:
        return time.time() - self.last_used


class ObjectPool(Generic[T]):
    """
    Generic object pool for reusable objects.
    
    Features:
    - Object reuse
    - Pre-allocation
    - Automatic growth
    - Idle cleanup
    """
    
    def __init__(
        self,
        factory: Callable[[], T],
        config: Optional[PoolConfig] = None,
        validator: Optional[Callable[[T], bool]] = None,
        reset_func: Optional[Callable[[T], None]] = None,
        name: str = "pool"
    ):
        """
        Initialize object pool.
        
        Args:
            factory: Function to create new objects
            config: Pool configuration
            validator: Function to validate objects before reuse
            reset_func: Function to reset object state
            name: Pool name for logging
        """
        self.factory = factory
        self.config = config or PoolConfig()
        self.validator = validator
        self.reset_func = reset_func
        self.name = name
        
        self._available: deque[PooledObject[T]] = deque()
        self._in_use: Dict[int, PooledObject[T]] = {}
        self._lock = threading.RLock()
        self._total_created = 0
        self._total_destroyed = 0
        
        # Pre-allocate
        self._pre_allocate()
    
    def _pre_allocate(self) -> None:
        """Pre-allocate initial objects."""
        for _ in range(self.config.initial_size):
            obj = self._create_object()
            if obj:
                self._available.append(obj)
    
    def _create_object(self) -> Optional[PooledObject[T]]:
        """Create a new pooled object."""
        try:
            obj = self.factory()
            self._total_created += 1
            return PooledObject(obj=obj)
        except Exception as e:
            logger.error(f"Failed to create object in pool '{self.name}': {e}")
            return None
    
    def _destroy_object(self, pooled: PooledObject[T]) -> None:
        """Destroy a pooled object."""
        self._total_destroyed += 1
        # Allow garbage collection
        del pooled
    
    @property
    def size(self) -> int:
        """Total pool size (available + in use)."""
        return len(self._available) + len(self._in_use)
    
    @property
    def available_count(self) -> int:
        """Number of available objects."""
        return len(self._available)
    
    @property
    def in_use_count(self) -> int:
        """Number of objects in use."""
        return len(self._in_use)
    
    def acquire(self, timeout: Optional[float] = None) -> Optional[T]:
        """
        Acquire an object from the pool.
        
        Args:
            timeout: Max time to wait for object
            
        Returns:
            Object from pool, or None if unavailable
        """
        deadline = time.time() + timeout if timeout else None
        
        while True:
            with self._lock:
                # Try to get from available
                while self._available:
                    pooled = self._available.popleft()
                    
                    # Validate if needed
                    if self.validator and not self.validator(pooled.obj):
                        self._destroy_object(pooled)
                        continue
                    
                    # Reset if needed
                    if self.reset_func:
                        try:
                            self.reset_func(pooled.obj)
                        except Exception as e:
                            logger.error(f"Reset failed: {e}")
                            self._destroy_object(pooled)
                            continue
                    
                    pooled.touch()
                    self._in_use[id(pooled.obj)] = pooled
                    return pooled.obj
                
                # Need to create new object
                if self.size < self.config.max_size:
                    pooled = self._create_object()
                    if pooled:
                        pooled.touch()
                        self._in_use[id(pooled.obj)] = pooled
                        return pooled.obj
            
            # Check timeout
            if deadline and time.time() >= deadline:
                return None
            
            # Wait and retry
            time.sleep(0.01)
    
    def release(self, obj: T) -> bool:
        """
        Release an object back to the pool.
        
        Args:
            obj: Object to release
            
        Returns:
            True if released, False if not from this pool
        """
        with self._lock:
            obj_id = id(obj)
            
            if obj_id not in self._in_use:
                logger.warning(f"Attempted to release object not from pool '{self.name}'")
                return False
            
            pooled = self._in_use.pop(obj_id)
            
            # Check if we should keep it
            if self.size <= self.config.max_size:
                self._available.append(pooled)
            else:
                self._destroy_object(pooled)
            
            return True
    
    def clear(self) -> int:
        """Clear all available objects."""
        with self._lock:
            count = len(self._available)
            for pooled in self._available:
                self._destroy_object(pooled)
            self._available.clear()
            return count
    
    def cleanup_idle(self) -> int:
        """Remove idle objects beyond minimum size."""
        with self._lock:
            cleaned = 0
            
            while (len(self._available) > self.config.min_size and
                   self._available):
                pooled = self._available[0]
                
                if pooled.idle_time > self.config.idle_timeout:
                    self._available.popleft()
                    self._destroy_object(pooled)
                    cleaned += 1
                else:
                    break
            
            return cleaned
    
    def grow(self, count: int = 1) -> int:
        """Pre-allocate additional objects."""
        with self._lock:
            created = 0
            for _ in range(count):
                if self.size >= self.config.max_size:
                    break
                
                pooled = self._create_object()
                if pooled:
                    self._available.append(pooled)
                    created += 1
            
            return created
    
    def shrink(self, target_size: int) -> int:
        """Shrink pool to target size."""
        with self._lock:
            removed = 0
            
            while self.size > target_size and self._available:
                pooled = self._available.pop()
                self._destroy_object(pooled)
                removed += 1
            
            return removed
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            avg_use_count = 0
            if self._in_use:
                avg_use_count = sum(p.use_count for p in self._in_use.values()) / len(self._in_use)
            
            return {
                'name': self.name,
                'size': self.size,
                'available': self.available_count,
                'in_use': self.in_use_count,
                'total_created': self._total_created,
                'total_destroyed': self._total_destroyed,
                'avg_use_count': round(avg_use_count, 2),
                'max_size': self.config.max_size
            }


class PoolManager:
    """
    Manages multiple object pools.
    
    Features:
    - Central pool management
    - Pool discovery
    - Aggregate statistics
    """
    
    def __init__(self):
        self._pools: Dict[str, ObjectPool] = {}
        self._lock = threading.RLock()
    
    def register_pool(
        self,
        name: str,
        pool: ObjectPool
    ) -> None:
        """Register a pool."""
        with self._lock:
            self._pools[name] = pool
    
    def get_pool(self, name: str) -> Optional[ObjectPool]:
        """Get pool by name."""
        return self._pools.get(name)
    
    def create_pool(
        self,
        name: str,
        factory: Callable[[], Any],
        config: Optional[PoolConfig] = None,
        **kwargs: Any
    ) -> ObjectPool:
        """Create and register a new pool."""
        pool = ObjectPool(factory, config, name=name, **kwargs)
        self.register_pool(name, pool)
        return pool
    
    def remove_pool(self, name: str) -> bool:
        """Remove and clear a pool."""
        with self._lock:
            pool = self._pools.pop(name, None)
            if pool:
                pool.clear()
                return True
            return False
    
    def cleanup_all(self) -> Dict[str, int]:
        """Cleanup idle objects in all pools."""
        results = {}
        with self._lock:
            for name, pool in self._pools.items():
                results[name] = pool.cleanup_idle()
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics."""
        with self._lock:
            return {
                'pool_count': len(self._pools),
                'pools': {name: pool.get_statistics() for name, pool in self._pools.items()}
            }


class PoolContext(Generic[T]):
    """Context manager for automatic pool release."""
    
    def __init__(self, pool: ObjectPool[T], timeout: Optional[float] = None):
        self.pool = pool
        self.timeout = timeout
        self.obj: Optional[T] = None
    
    def __enter__(self) -> T:
        self.obj = self.pool.acquire(self.timeout)
        if self.obj is None:
            raise RuntimeError("Failed to acquire object from pool")
        return self.obj
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.obj is not None:
            self.pool.release(self.obj)


# Global pool manager
_manager: Optional[PoolManager] = None


def get_pool_manager() -> PoolManager:
    """Get global pool manager."""
    global _manager
    if _manager is None:
        _manager = PoolManager()
    return _manager


def create_pool(
    name: str,
    factory: Callable[[], Any],
    config: Optional[PoolConfig] = None
) -> ObjectPool:
    """Create and register a pool."""
    return get_pool_manager().create_pool(name, factory, config)


def get_pool(name: str) -> Optional[ObjectPool]:
    """Get pool by name."""
    return get_pool_manager().get_pool(name)


if __name__ == "__main__":
    print("Object Pool Module Demo")
    print("=" * 50)
    
    # Example: String buffer pool
    class StringBuffer:
        def __init__(self, size: int = 1024):
            self.buffer = bytearray(size)
            self.position = 0
        
        def reset(self) -> None:
            self.position = 0
        
        def write(self, data: bytes) -> None:
            end = self.position + len(data)
            self.buffer[self.position:end] = data
            self.position = end
    
    print("\n1. Create Buffer Pool")
    print("-" * 30)
    
    pool = ObjectPool(
        factory=lambda: StringBuffer(1024),
        config=PoolConfig(initial_size=5, max_size=20),
        reset_func=lambda b: b.reset(),
        name="string_buffers"
    )
    
    stats = pool.get_statistics()
    print(f"Pool size: {stats['size']}")
    print(f"Available: {stats['available']}")
    
    print("\n\n2. Acquire Objects")
    print("-" * 30)
    
    buffers = []
    for i in range(3):
        buf = pool.acquire()
        buf.write(f"Data {i}".encode())
        buffers.append(buf)
        print(f"Acquired buffer {i}, position: {buf.position}")
    
    stats = pool.get_statistics()
    print(f"\nAvailable: {stats['available']}, In use: {stats['in_use']}")
    
    print("\n\n3. Release Objects")
    print("-" * 30)
    
    for buf in buffers:
        pool.release(buf)
    
    stats = pool.get_statistics()
    print(f"Available: {stats['available']}, In use: {stats['in_use']}")
    
    print("\n\n4. Pool Context Manager")
    print("-" * 30)
    
    with PoolContext(pool) as buf:
        buf.write(b"Context data")
        print(f"Using buffer, position: {buf.position}")
    
    stats = pool.get_statistics()
    print(f"After context: Available={stats['available']}, In use={stats['in_use']}")
    
    print("\n\n5. Pool Growth")
    print("-" * 30)
    
    print(f"Size before: {pool.size}")
    created = pool.grow(5)
    print(f"Created: {created}")
    print(f"Size after: {pool.size}")
    
    print("\n\n6. Pool Shrink")
    print("-" * 30)
    
    removed = pool.shrink(5)
    print(f"Removed: {removed}")
    print(f"Size after: {pool.size}")
    
    print("\n\n7. Pool Manager")
    print("-" * 30)
    
    manager = PoolManager()
    
    # Create different pools
    manager.create_pool("buffers", lambda: StringBuffer(512))
    manager.create_pool("lists", lambda: [])
    manager.create_pool("dicts", lambda: {})
    
    stats = manager.get_statistics()
    print(f"Total pools: {stats['pool_count']}")
    for name, pool_stats in stats['pools'].items():
        print(f"  {name}: size={pool_stats['size']}")
    
    print("\n\n8. Cleanup Idle")
    print("-" * 30)
    
    # Set short idle timeout for demo
    pool.config.idle_timeout = 0.1
    time.sleep(0.2)
    
    cleaned = pool.cleanup_idle()
    print(f"Cleaned: {cleaned} idle objects")
    
    print("\n\n9. Pool Statistics")
    print("-" * 30)
    
    stats = pool.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n\n✓ Object pool demo complete")
