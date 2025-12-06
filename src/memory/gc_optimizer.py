"""
Garbage Collection Optimizer Module for CineMatch V2.1.6

Implements GC optimization strategies:
- GC tuning
- Generation management
- Collection scheduling
- Memory pressure handling

Phase 6 - Task 6.2: GC Optimizer
"""

import logging
import time
import threading
import gc
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class GCMode(Enum):
    """Garbage collection mode."""
    NORMAL = "normal"
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    DISABLED = "disabled"


@dataclass
class GCConfig:
    """GC optimizer configuration."""
    # Thresholds for each generation
    gen0_threshold: int = 700
    gen1_threshold: int = 10
    gen2_threshold: int = 10
    
    # Collection intervals
    full_collection_interval: float = 300.0  # 5 minutes
    
    # Pressure thresholds
    pressure_threshold: float = 80.0  # Memory percent
    aggressive_threshold: float = 90.0


@dataclass
class GCStats:
    """GC statistics."""
    collections: Tuple[int, int, int] = (0, 0, 0)  # Per generation
    collected: int = 0
    uncollectable: int = 0
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'collections': list(self.collections),
            'collected': self.collected,
            'uncollectable': self.uncollectable,
            'duration_ms': round(self.duration_ms, 2),
            'timestamp': datetime.fromtimestamp(self.timestamp).isoformat()
        }


class GCOptimizer:
    """
    Optimizes garbage collection.
    
    Features:
    - GC tuning
    - Scheduled collections
    - Memory pressure handling
    - Collection statistics
    """
    
    def __init__(self, config: Optional[GCConfig] = None):
        self.config = config or GCConfig()
        self._mode = GCMode.NORMAL
        self._original_thresholds = gc.get_threshold()
        self._stats_history: List[GCStats] = []
        self._lock = threading.RLock()
        self._scheduler_thread: Optional[threading.Thread] = None
        self._stop_scheduler = threading.Event()
        self._callbacks: List[Callable[[GCStats], None]] = []
        
        # Track collections
        self._collection_counts = [0, 0, 0]
        self._total_collected = 0
    
    @property
    def mode(self) -> GCMode:
        return self._mode
    
    def set_mode(self, mode: GCMode) -> None:
        """Set GC mode."""
        with self._lock:
            self._mode = mode
            
            if mode == GCMode.DISABLED:
                gc.disable()
                logger.info("GC disabled")
            elif mode == GCMode.AGGRESSIVE:
                gc.enable()
                gc.set_threshold(100, 5, 5)
                logger.info("GC set to aggressive mode")
            elif mode == GCMode.CONSERVATIVE:
                gc.enable()
                gc.set_threshold(1000, 20, 20)
                logger.info("GC set to conservative mode")
            else:  # NORMAL
                gc.enable()
                gc.set_threshold(
                    self.config.gen0_threshold,
                    self.config.gen1_threshold,
                    self.config.gen2_threshold
                )
                logger.info("GC set to normal mode")
    
    def tune_thresholds(
        self,
        gen0: Optional[int] = None,
        gen1: Optional[int] = None,
        gen2: Optional[int] = None
    ) -> Tuple[int, int, int]:
        """Tune GC thresholds."""
        current = gc.get_threshold()
        
        new_thresholds = (
            gen0 if gen0 is not None else current[0],
            gen1 if gen1 is not None else current[1],
            gen2 if gen2 is not None else current[2]
        )
        
        gc.set_threshold(*new_thresholds)
        
        logger.info(f"GC thresholds: {current} -> {new_thresholds}")
        
        return new_thresholds
    
    def collect(
        self,
        generation: int = 2,
        force: bool = False
    ) -> GCStats:
        """
        Perform garbage collection.
        
        Args:
            generation: 0, 1, or 2 (higher = more thorough)
            force: Collect even in disabled mode
        """
        if self._mode == GCMode.DISABLED and not force:
            return GCStats()
        
        start_time = time.time()
        counts_before = gc.get_count()
        
        collected = gc.collect(generation)
        
        counts_after = gc.get_count()
        duration_ms = (time.time() - start_time) * 1000
        
        stats = GCStats(
            collections=(
                counts_before[0] - counts_after[0],
                counts_before[1] - counts_after[1],
                counts_before[2] - counts_after[2]
            ),
            collected=collected,
            uncollectable=len(gc.garbage),
            duration_ms=duration_ms
        )
        
        with self._lock:
            self._stats_history.append(stats)
            if len(self._stats_history) > 1000:
                self._stats_history = self._stats_history[-1000:]
            
            self._total_collected += collected
            self._collection_counts[generation] += 1
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(stats)
            except Exception as e:
                logger.error(f"GC callback error: {e}")
        
        logger.debug(f"GC gen{generation}: collected {collected}, took {duration_ms:.1f}ms")
        
        return stats
    
    def collect_if_needed(self, memory_percent: float) -> Optional[GCStats]:
        """Collect if memory pressure is high."""
        if memory_percent >= self.config.aggressive_threshold:
            self.set_mode(GCMode.AGGRESSIVE)
            return self.collect(generation=2)
        elif memory_percent >= self.config.pressure_threshold:
            return self.collect(generation=1)
        return None
    
    def on_collection(self, callback: Callable[[GCStats], None]) -> None:
        """Register collection callback."""
        self._callbacks.append(callback)
    
    def start_scheduler(self) -> None:
        """Start scheduled collections."""
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            return
        
        self._stop_scheduler.clear()
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True
        )
        self._scheduler_thread.start()
        logger.info("GC scheduler started")
    
    def stop_scheduler(self) -> None:
        """Stop scheduled collections."""
        self._stop_scheduler.set()
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)
        logger.info("GC scheduler stopped")
    
    def _scheduler_loop(self) -> None:
        """Scheduled collection loop."""
        while not self._stop_scheduler.wait(self.config.full_collection_interval):
            if self._mode != GCMode.DISABLED:
                self.collect(generation=2)
    
    def freeze(self) -> None:
        """Freeze GC for performance-critical sections."""
        gc.disable()
        logger.debug("GC frozen")
    
    def unfreeze(self) -> None:
        """Unfreeze GC after performance-critical sections."""
        gc.enable()
        logger.debug("GC unfrozen")
    
    def get_status(self) -> Dict[str, Any]:
        """Get GC status."""
        return {
            'enabled': gc.isenabled(),
            'mode': self._mode.value,
            'thresholds': gc.get_threshold(),
            'counts': gc.get_count(),
            'garbage_objects': len(gc.garbage),
            'total_collected': self._total_collected,
            'collection_counts': self._collection_counts
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get GC statistics."""
        with self._lock:
            if not self._stats_history:
                return {}
            
            recent = self._stats_history[-100:]
            
            total_collected = sum(s.collected for s in recent)
            total_duration = sum(s.duration_ms for s in recent)
            
            return {
                'total_collections': len(self._stats_history),
                'recent_collected': total_collected,
                'recent_duration_ms': round(total_duration, 2),
                'avg_duration_ms': round(total_duration / len(recent), 2),
                'uncollectable': sum(s.uncollectable for s in recent),
                'mode': self._mode.value
            }
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        with self._lock:
            self._stats_history.clear()
            self._collection_counts = [0, 0, 0]
            self._total_collected = 0
    
    def restore_defaults(self) -> None:
        """Restore original GC settings."""
        gc.enable()
        gc.set_threshold(*self._original_thresholds)
        self._mode = GCMode.NORMAL
        logger.info("GC settings restored to defaults")


class GCContext:
    """Context manager for GC control."""
    
    def __init__(
        self,
        optimizer: GCOptimizer,
        disable: bool = True,
        collect_after: bool = True
    ):
        self.optimizer = optimizer
        self.disable = disable
        self.collect_after = collect_after
        self._was_enabled = gc.isenabled()
    
    def __enter__(self) -> 'GCContext':
        if self.disable:
            self.optimizer.freeze()
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.disable:
            self.optimizer.unfreeze()
        if self.collect_after:
            self.optimizer.collect(generation=0)


# Global optimizer
_optimizer: Optional[GCOptimizer] = None


def get_gc_optimizer() -> GCOptimizer:
    """Get global GC optimizer."""
    global _optimizer
    if _optimizer is None:
        _optimizer = GCOptimizer()
    return _optimizer


def gc_collect(generation: int = 2) -> GCStats:
    """Perform GC collection."""
    return get_gc_optimizer().collect(generation)


def set_gc_mode(mode: GCMode) -> None:
    """Set GC mode."""
    get_gc_optimizer().set_mode(mode)


def gc_disabled() -> GCContext:
    """Context manager to disable GC temporarily."""
    return GCContext(get_gc_optimizer())


if __name__ == "__main__":
    print("GC Optimizer Module Demo")
    print("=" * 50)
    
    # Create optimizer
    optimizer = GCOptimizer()
    
    print("\n1. Current GC Status")
    print("-" * 30)
    
    status = optimizer.get_status()
    print(f"Enabled: {status['enabled']}")
    print(f"Mode: {status['mode']}")
    print(f"Thresholds: {status['thresholds']}")
    print(f"Counts: {status['counts']}")
    
    print("\n\n2. Tune Thresholds")
    print("-" * 30)
    
    new_thresh = optimizer.tune_thresholds(gen0=500, gen1=8, gen2=8)
    print(f"New thresholds: {new_thresh}")
    
    print("\n\n3. GC Modes")
    print("-" * 30)
    
    optimizer.set_mode(GCMode.AGGRESSIVE)
    print(f"Mode: {optimizer.mode.value}")
    print(f"Thresholds: {gc.get_threshold()}")
    
    optimizer.set_mode(GCMode.CONSERVATIVE)
    print(f"Mode: {optimizer.mode.value}")
    print(f"Thresholds: {gc.get_threshold()}")
    
    optimizer.set_mode(GCMode.NORMAL)
    
    print("\n\n4. Manual Collection")
    print("-" * 30)
    
    # Create garbage
    garbage = [{"key": list(range(100))} for _ in range(1000)]
    del garbage
    
    stats = optimizer.collect(generation=2)
    print(f"Collected: {stats.collected} objects")
    print(f"Duration: {stats.duration_ms:.2f}ms")
    print(f"Uncollectable: {stats.uncollectable}")
    
    print("\n\n5. GC Context Manager")
    print("-" * 30)
    
    print(f"GC enabled before: {gc.isenabled()}")
    
    with GCContext(optimizer, disable=True, collect_after=True):
        print(f"GC enabled during: {gc.isenabled()}")
        # Performance-critical code here
        _ = [i ** 2 for i in range(10000)]
    
    print(f"GC enabled after: {gc.isenabled()}")
    
    print("\n\n6. Collection Callback")
    print("-" * 30)
    
    def on_gc(stats: GCStats):
        print(f"  GC completed: {stats.collected} objects in {stats.duration_ms:.1f}ms")
    
    optimizer.on_collection(on_gc)
    
    garbage = [list(range(100)) for _ in range(500)]
    del garbage
    optimizer.collect(generation=1)
    
    print("\n\n7. Statistics")
    print("-" * 30)
    
    statistics = optimizer.get_statistics()
    for key, value in statistics.items():
        print(f"  {key}: {value}")
    
    print("\n\n8. Restore Defaults")
    print("-" * 30)
    
    optimizer.restore_defaults()
    status = optimizer.get_status()
    print(f"Thresholds restored: {status['thresholds']}")
    
    print("\n\n✓ GC optimizer demo complete")
