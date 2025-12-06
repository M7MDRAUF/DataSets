"""
Memory Monitor Module for CineMatch V2.1.6

Implements memory monitoring and tracking:
- Memory usage monitoring
- Memory alerts
- Usage history
- Memory statistics

Phase 6 - Task 6.1: Memory Monitor
"""

import logging
import time
import threading
import sys
import gc
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import psutil for detailed memory info
try:
    import psutil  # type: ignore
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - memory monitoring limited")


class MemoryLevel(Enum):
    """Memory usage level."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    timestamp: float
    rss_bytes: int  # Resident Set Size
    vms_bytes: int  # Virtual Memory Size
    percent: float
    available_bytes: int
    total_bytes: int
    heap_bytes: int = 0
    gc_counts: Tuple[int, int, int] = (0, 0, 0)
    
    @property
    def rss_mb(self) -> float:
        return self.rss_bytes / (1024 * 1024)
    
    @property
    def vms_mb(self) -> float:
        return self.vms_bytes / (1024 * 1024)
    
    @property
    def available_mb(self) -> float:
        return self.available_bytes / (1024 * 1024)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': datetime.fromtimestamp(self.timestamp).isoformat(),
            'rss_mb': round(self.rss_mb, 2),
            'vms_mb': round(self.vms_mb, 2),
            'percent': round(self.percent, 2),
            'available_mb': round(self.available_mb, 2),
            'gc_counts': list(self.gc_counts)
        }


@dataclass
class MemoryConfig:
    """Memory monitor configuration."""
    warning_threshold: float = 70.0  # percent
    critical_threshold: float = 85.0
    emergency_threshold: float = 95.0
    check_interval: float = 10.0
    history_size: int = 1000
    auto_gc_threshold: float = 80.0


class MemoryAlert:
    """Memory alert record."""
    
    def __init__(
        self,
        level: MemoryLevel,
        message: str,
        snapshot: MemorySnapshot
    ):
        self.level = level
        self.message = message
        self.snapshot = snapshot
        self.timestamp = time.time()
        self.resolved = False
        self.resolved_at: Optional[float] = None
    
    def resolve(self) -> None:
        self.resolved = True
        self.resolved_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'level': self.level.value,
            'message': self.message,
            'memory_percent': round(self.snapshot.percent, 2),
            'timestamp': datetime.fromtimestamp(self.timestamp).isoformat(),
            'resolved': self.resolved
        }


class MemoryMonitor:
    """
    Monitors system memory usage.
    
    Features:
    - Real-time memory tracking
    - Threshold-based alerts
    - Historical data
    - Automatic GC triggers
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self._history: List[MemorySnapshot] = []
        self._alerts: List[MemoryAlert] = []
        self._lock = threading.RLock()
        self._current_level = MemoryLevel.NORMAL
        self._callbacks: Dict[MemoryLevel, List[Callable[[MemoryAlert], None]]] = {
            level: [] for level in MemoryLevel
        }
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._process = psutil.Process() if PSUTIL_AVAILABLE else None
    
    def get_current_usage(self) -> MemorySnapshot:
        """Get current memory usage."""
        gc_counts = gc.get_count()
        
        if PSUTIL_AVAILABLE and self._process:
            mem_info = self._process.memory_info()
            sys_mem = psutil.virtual_memory()
            
            return MemorySnapshot(
                timestamp=time.time(),
                rss_bytes=mem_info.rss,
                vms_bytes=mem_info.vms,
                percent=self._process.memory_percent(),
                available_bytes=sys_mem.available,
                total_bytes=sys_mem.total,
                heap_bytes=self._get_heap_size(),
                gc_counts=gc_counts
            )
        else:
            # Fallback without psutil
            return MemorySnapshot(
                timestamp=time.time(),
                rss_bytes=0,
                vms_bytes=0,
                percent=0,
                available_bytes=0,
                total_bytes=0,
                heap_bytes=self._get_heap_size(),
                gc_counts=gc_counts
            )
    
    def _get_heap_size(self) -> int:
        """Estimate heap size from tracked objects."""
        try:
            return sum(sys.getsizeof(obj) for obj in gc.get_objects()[:1000])
        except Exception:
            return 0
    
    def _get_memory_level(self, percent: float) -> MemoryLevel:
        """Determine memory level from percentage."""
        if percent >= self.config.emergency_threshold:
            return MemoryLevel.EMERGENCY
        elif percent >= self.config.critical_threshold:
            return MemoryLevel.CRITICAL
        elif percent >= self.config.warning_threshold:
            return MemoryLevel.WARNING
        return MemoryLevel.NORMAL
    
    def check_memory(self) -> MemorySnapshot:
        """Check memory and trigger alerts if needed."""
        snapshot = self.get_current_usage()
        
        with self._lock:
            # Add to history
            self._history.append(snapshot)
            if len(self._history) > self.config.history_size:
                self._history = self._history[-self.config.history_size:]
            
            # Check level
            new_level = self._get_memory_level(snapshot.percent)
            
            if new_level != self._current_level:
                # Level changed
                if new_level.value != MemoryLevel.NORMAL.value:
                    self._raise_alert(new_level, snapshot)
                else:
                    # Resolved
                    self._resolve_alerts()
                
                self._current_level = new_level
            
            # Auto GC
            if snapshot.percent >= self.config.auto_gc_threshold:
                gc.collect()
        
        return snapshot
    
    def _raise_alert(self, level: MemoryLevel, snapshot: MemorySnapshot) -> None:
        """Raise a memory alert."""
        message = f"Memory usage at {snapshot.percent:.1f}% ({level.value})"
        alert = MemoryAlert(level, message, snapshot)
        
        self._alerts.append(alert)
        
        logger.warning(f"Memory alert: {message}")
        
        # Notify callbacks
        for callback in self._callbacks.get(level, []):
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def _resolve_alerts(self) -> None:
        """Resolve active alerts."""
        for alert in self._alerts:
            if not alert.resolved:
                alert.resolve()
                logger.info(f"Memory alert resolved: {alert.message}")
    
    def on_alert(
        self,
        level: MemoryLevel,
        callback: Callable[[MemoryAlert], None]
    ) -> None:
        """Register alert callback."""
        self._callbacks[level].append(callback)
    
    def start_monitoring(self) -> None:
        """Start background monitoring."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._stop_monitoring.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Memory monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_monitoring.wait(self.config.check_interval):
            try:
                self.check_memory()
            except Exception as e:
                logger.error(f"Memory check error: {e}")
    
    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get memory history."""
        with self._lock:
            return [s.to_dict() for s in self._history[-limit:]]
    
    def get_alerts(self, active_only: bool = False) -> List[Dict[str, Any]]:
        """Get memory alerts."""
        with self._lock:
            alerts = self._alerts
            if active_only:
                alerts = [a for a in alerts if not a.resolved]
            return [a.to_dict() for a in alerts[-50:]]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        with self._lock:
            if not self._history:
                return {}
            
            recent = self._history[-100:]
            
            return {
                'current': self._history[-1].to_dict() if self._history else None,
                'level': self._current_level.value,
                'avg_percent': round(sum(s.percent for s in recent) / len(recent), 2),
                'max_percent': round(max(s.percent for s in recent), 2),
                'min_percent': round(min(s.percent for s in recent), 2),
                'samples': len(self._history),
                'active_alerts': len([a for a in self._alerts if not a.resolved])
            }
    
    def force_gc(self, generation: int = 2) -> Dict[str, Any]:
        """Force garbage collection."""
        before = self.get_current_usage()
        
        collected = gc.collect(generation)
        
        after = self.get_current_usage()
        
        return {
            'collected_objects': collected,
            'before_mb': round(before.rss_mb, 2),
            'after_mb': round(after.rss_mb, 2),
            'freed_mb': round(before.rss_mb - after.rss_mb, 2)
        }


# Global monitor
_monitor: Optional[MemoryMonitor] = None


def get_memory_monitor() -> MemoryMonitor:
    """Get global memory monitor."""
    global _monitor
    if _monitor is None:
        _monitor = MemoryMonitor()
    return _monitor


def get_memory_usage() -> MemorySnapshot:
    """Get current memory usage."""
    return get_memory_monitor().get_current_usage()


def check_memory() -> MemorySnapshot:
    """Check memory and trigger alerts."""
    return get_memory_monitor().check_memory()


def start_memory_monitoring() -> None:
    """Start memory monitoring."""
    get_memory_monitor().start_monitoring()


def stop_memory_monitoring() -> None:
    """Stop memory monitoring."""
    get_memory_monitor().stop_monitoring()


if __name__ == "__main__":
    print("Memory Monitor Module Demo")
    print("=" * 50)
    
    # Create monitor
    monitor = MemoryMonitor(MemoryConfig(
        warning_threshold=30.0,  # Low for demo
        check_interval=1.0
    ))
    
    print("\n1. Current Memory Usage")
    print("-" * 30)
    
    snapshot = monitor.get_current_usage()
    print(f"RSS: {snapshot.rss_mb:.2f} MB")
    print(f"VMS: {snapshot.vms_mb:.2f} MB")
    print(f"Percent: {snapshot.percent:.2f}%")
    print(f"Available: {snapshot.available_mb:.2f} MB")
    print(f"GC counts: {snapshot.gc_counts}")
    
    print("\n\n2. Register Alert Callback")
    print("-" * 30)
    
    def on_warning(alert: MemoryAlert):
        print(f"  ⚠️ WARNING: {alert.message}")
    
    def on_critical(alert: MemoryAlert):
        print(f"  🔴 CRITICAL: {alert.message}")
    
    monitor.on_alert(MemoryLevel.WARNING, on_warning)
    monitor.on_alert(MemoryLevel.CRITICAL, on_critical)
    
    print("Callbacks registered")
    
    print("\n\n3. Memory Check (with alerts)")
    print("-" * 30)
    
    for i in range(3):
        snapshot = monitor.check_memory()
        print(f"Check {i+1}: {snapshot.percent:.2f}%")
        time.sleep(0.5)
    
    print("\n\n4. Force Garbage Collection")
    print("-" * 30)
    
    # Create some garbage
    garbage = [list(range(1000)) for _ in range(100)]
    del garbage
    
    result = monitor.force_gc()
    print(f"Collected: {result['collected_objects']} objects")
    print(f"Before: {result['before_mb']} MB")
    print(f"After: {result['after_mb']} MB")
    print(f"Freed: {result['freed_mb']} MB")
    
    print("\n\n5. Statistics")
    print("-" * 30)
    
    stats = monitor.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n\n6. History")
    print("-" * 30)
    
    history = monitor.get_history(limit=5)
    for entry in history:
        print(f"  {entry['timestamp']}: {entry['percent']}%")
    
    print("\n\n✓ Memory monitor demo complete")
