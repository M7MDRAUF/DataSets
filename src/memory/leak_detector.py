"""
Memory Leak Detector Module for CineMatch V2.1.6

Implements memory leak detection:
- Reference tracking
- Growth monitoring
- Leak identification
- Memory profiling

Phase 6 - Task 6.5: Leak Detector
"""

import logging
import time
import threading
import gc
import weakref
import sys
from typing import Any, Callable, Dict, List, Optional, Set, Type
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import tracemalloc for detailed tracking
try:
    import tracemalloc
    TRACEMALLOC_AVAILABLE = True
except ImportError:
    TRACEMALLOC_AVAILABLE = False


@dataclass
class ObjectSnapshot:
    """Snapshot of object counts by type."""
    timestamp: float
    type_counts: Dict[str, int]
    total_objects: int
    total_bytes: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': datetime.fromtimestamp(self.timestamp).isoformat(),
            'total_objects': self.total_objects,
            'total_bytes': self.total_bytes,
            'top_types': dict(sorted(
                self.type_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20])
        }


@dataclass
class LeakReport:
    """Memory leak detection report."""
    timestamp: float
    growing_types: Dict[str, int]  # Type name -> growth count
    suspicious_objects: List[Dict[str, Any]]
    total_growth: int
    severity: str  # low, medium, high
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': datetime.fromtimestamp(self.timestamp).isoformat(),
            'severity': self.severity,
            'total_growth': self.total_growth,
            'growing_types': self.growing_types,
            'suspicious_count': len(self.suspicious_objects)
        }


@dataclass
class LeakConfig:
    """Leak detector configuration."""
    snapshot_interval: float = 60.0
    growth_threshold: int = 100
    min_snapshots: int = 3
    track_allocations: bool = True
    max_snapshots: int = 100


class ObjectTracker:
    """Tracks object instances using weak references."""
    
    def __init__(self):
        self._tracked: Dict[str, Set[int]] = defaultdict(set)
        self._weak_refs: Dict[int, weakref.ref] = {}
        self._lock = threading.RLock()
    
    def track(self, obj: Any, label: str = "") -> None:
        """Track an object."""
        with self._lock:
            obj_id = id(obj)
            type_name = type(obj).__name__
            
            if label:
                type_name = f"{label}:{type_name}"
            
            self._tracked[type_name].add(obj_id)
            
            # Create weak reference
            try:
                ref = weakref.ref(obj, lambda r: self._on_finalize(obj_id, type_name))
                self._weak_refs[obj_id] = ref
            except TypeError:
                # Object doesn't support weak references
                pass
    
    def _on_finalize(self, obj_id: int, type_name: str) -> None:
        """Called when tracked object is garbage collected."""
        with self._lock:
            self._tracked[type_name].discard(obj_id)
            self._weak_refs.pop(obj_id, None)
    
    def get_counts(self) -> Dict[str, int]:
        """Get counts of tracked objects."""
        with self._lock:
            return {k: len(v) for k, v in self._tracked.items()}
    
    def get_alive_count(self) -> int:
        """Get count of alive tracked objects."""
        with self._lock:
            return sum(
                1 for ref in self._weak_refs.values()
                if ref() is not None
            )
    
    def clear(self) -> None:
        """Clear all tracking."""
        with self._lock:
            self._tracked.clear()
            self._weak_refs.clear()


class MemoryLeakDetector:
    """
    Detects potential memory leaks.
    
    Features:
    - Object growth tracking
    - Type-based analysis
    - Allocation tracking
    - Leak reports
    """
    
    def __init__(self, config: Optional[LeakConfig] = None):
        self.config = config or LeakConfig()
        self._snapshots: List[ObjectSnapshot] = []
        self._reports: List[LeakReport] = []
        self._lock = threading.RLock()
        self._tracker = ObjectTracker()
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        # Start tracemalloc if available
        if TRACEMALLOC_AVAILABLE and self.config.track_allocations:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
    
    def take_snapshot(self) -> ObjectSnapshot:
        """Take a snapshot of current object counts."""
        gc.collect()  # Force collection first
        
        type_counts: Dict[str, int] = defaultdict(int)
        total_bytes = 0
        
        for obj in gc.get_objects():
            type_name = type(obj).__name__
            type_counts[type_name] += 1
            try:
                total_bytes += sys.getsizeof(obj)
            except (TypeError, RuntimeError):
                pass
        
        snapshot = ObjectSnapshot(
            timestamp=time.time(),
            type_counts=dict(type_counts),
            total_objects=sum(type_counts.values()),
            total_bytes=total_bytes
        )
        
        with self._lock:
            self._snapshots.append(snapshot)
            if len(self._snapshots) > self.config.max_snapshots:
                self._snapshots = self._snapshots[-self.config.max_snapshots:]
        
        return snapshot
    
    def detect_leaks(self) -> Optional[LeakReport]:
        """Analyze snapshots for potential leaks."""
        with self._lock:
            if len(self._snapshots) < self.config.min_snapshots:
                return None
            
            # Compare oldest vs newest
            oldest = self._snapshots[0]
            newest = self._snapshots[-1]
            
            # Find growing types
            growing_types: Dict[str, int] = {}
            
            for type_name, new_count in newest.type_counts.items():
                old_count = oldest.type_counts.get(type_name, 0)
                growth = new_count - old_count
                
                if growth > self.config.growth_threshold:
                    growing_types[type_name] = growth
            
            if not growing_types:
                return None
            
            # Determine severity
            total_growth = sum(growing_types.values())
            
            if total_growth > 10000:
                severity = "high"
            elif total_growth > 1000:
                severity = "medium"
            else:
                severity = "low"
            
            # Find suspicious objects
            suspicious = self._find_suspicious_objects(growing_types)
            
            report = LeakReport(
                timestamp=time.time(),
                growing_types=growing_types,
                suspicious_objects=suspicious,
                total_growth=total_growth,
                severity=severity
            )
            
            self._reports.append(report)
            
            if severity in ("medium", "high"):
                logger.warning(
                    f"Potential memory leak detected: {total_growth} objects growing"
                )
            
            return report
    
    def _find_suspicious_objects(
        self,
        growing_types: Dict[str, int]
    ) -> List[Dict[str, Any]]:
        """Find suspicious objects of growing types."""
        suspicious = []
        
        for type_name in list(growing_types.keys())[:5]:  # Top 5
            count = 0
            for obj in gc.get_objects():
                if type(obj).__name__ == type_name:
                    count += 1
                    if count <= 3:  # Sample a few
                        try:
                            referrers = len(gc.get_referrers(obj))
                            suspicious.append({
                                'type': type_name,
                                'referrer_count': referrers,
                                'size_bytes': sys.getsizeof(obj)
                            })
                        except Exception:
                            pass
        
        return suspicious
    
    def track_object(self, obj: Any, label: str = "") -> None:
        """Track an object for leak detection."""
        self._tracker.track(obj, label)
    
    def get_tracked_counts(self) -> Dict[str, int]:
        """Get counts of tracked objects."""
        return self._tracker.get_counts()
    
    def start_monitoring(self) -> None:
        """Start background leak monitoring."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Leak detector monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._stop_monitoring.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Leak detector monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_monitoring.wait(self.config.snapshot_interval):
            try:
                self.take_snapshot()
                self.detect_leaks()
            except Exception as e:
                logger.error(f"Leak detection error: {e}")
    
    def get_memory_diff(self) -> Optional[Dict[str, Any]]:
        """Get memory difference between first and last snapshot."""
        with self._lock:
            if len(self._snapshots) < 2:
                return None
            
            first = self._snapshots[0]
            last = self._snapshots[-1]
            
            return {
                'time_span_seconds': last.timestamp - first.timestamp,
                'object_growth': last.total_objects - first.total_objects,
                'byte_growth': last.total_bytes - first.total_bytes,
                'snapshots': len(self._snapshots)
            }
    
    def get_allocation_stats(self) -> Optional[Dict[str, Any]]:
        """Get allocation statistics from tracemalloc."""
        if not TRACEMALLOC_AVAILABLE or not tracemalloc.is_tracing():
            return None
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')[:10]
        
        return {
            'top_allocations': [
                {
                    'file': str(stat.traceback),
                    'size_kb': round(stat.size / 1024, 2),
                    'count': stat.count
                }
                for stat in top_stats
            ],
            'traced_memory_mb': round(
                tracemalloc.get_traced_memory()[0] / (1024 * 1024), 2
            ),
            'peak_memory_mb': round(
                tracemalloc.get_traced_memory()[1] / (1024 * 1024), 2
            )
        }
    
    def get_reports(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent leak reports."""
        with self._lock:
            return [r.to_dict() for r in self._reports[-limit:]]
    
    def get_snapshots(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent snapshots."""
        with self._lock:
            return [s.to_dict() for s in self._snapshots[-limit:]]
    
    def clear_history(self) -> None:
        """Clear snapshots and reports."""
        with self._lock:
            self._snapshots.clear()
            self._reports.clear()
    
    def get_status(self) -> Dict[str, Any]:
        """Get detector status."""
        with self._lock:
            return {
                'snapshots': len(self._snapshots),
                'reports': len(self._reports),
                'tracked_objects': self._tracker.get_alive_count(),
                'tracemalloc_enabled': TRACEMALLOC_AVAILABLE and tracemalloc.is_tracing(),
                'monitoring': self._monitor_thread is not None and self._monitor_thread.is_alive()
            }


# Global detector
_detector: Optional[MemoryLeakDetector] = None


def get_leak_detector() -> MemoryLeakDetector:
    """Get global leak detector."""
    global _detector
    if _detector is None:
        _detector = MemoryLeakDetector()
    return _detector


def take_memory_snapshot() -> ObjectSnapshot:
    """Take a memory snapshot."""
    return get_leak_detector().take_snapshot()


def detect_memory_leaks() -> Optional[LeakReport]:
    """Detect memory leaks."""
    return get_leak_detector().detect_leaks()


def track_for_leaks(obj: Any, label: str = "") -> None:
    """Track an object for leak detection."""
    get_leak_detector().track_object(obj, label)


if __name__ == "__main__":
    print("Memory Leak Detector Module Demo")
    print("=" * 50)
    
    # Create detector
    detector = MemoryLeakDetector(LeakConfig(
        growth_threshold=50,
        min_snapshots=2
    ))
    
    print("\n1. Take Initial Snapshot")
    print("-" * 30)
    
    snapshot1 = detector.take_snapshot()
    print(f"Total objects: {snapshot1.total_objects}")
    print(f"Total bytes: {snapshot1.total_bytes / (1024*1024):.2f} MB")
    
    print("\n\n2. Create Objects (Simulate Activity)")
    print("-" * 30)
    
    # Create some objects that could be leaks
    leaked_lists = []
    for i in range(200):
        leaked_lists.append(list(range(100)))
    
    leaked_dicts = []
    for i in range(150):
        leaked_dicts.append({f"key_{j}": j for j in range(50)})
    
    print(f"Created {len(leaked_lists)} lists")
    print(f"Created {len(leaked_dicts)} dicts")
    
    print("\n\n3. Take Second Snapshot")
    print("-" * 30)
    
    snapshot2 = detector.take_snapshot()
    print(f"Total objects: {snapshot2.total_objects}")
    print(f"Growth: {snapshot2.total_objects - snapshot1.total_objects}")
    
    print("\n\n4. Detect Leaks")
    print("-" * 30)
    
    report = detector.detect_leaks()
    if report:
        print(f"Severity: {report.severity}")
        print(f"Total growth: {report.total_growth}")
        print(f"Growing types:")
        for type_name, count in list(report.growing_types.items())[:5]:
            print(f"  {type_name}: +{count}")
    else:
        print("No significant leaks detected")
    
    print("\n\n5. Object Tracking")
    print("-" * 30)
    
    class MyResource:
        def __init__(self, name: str):
            self.name = name
            self.data = bytearray(1000)
    
    resources = []
    for i in range(10):
        res = MyResource(f"resource_{i}")
        detector.track_object(res, "MyResource")
        resources.append(res)
    
    print(f"Tracked counts: {detector.get_tracked_counts()}")
    
    # Delete some
    del resources[0:5]
    gc.collect()
    
    print(f"After deletion: {detector.get_tracked_counts()}")
    
    print("\n\n6. Memory Diff")
    print("-" * 30)
    
    diff = detector.get_memory_diff()
    if diff:
        for key, value in diff.items():
            print(f"  {key}: {value}")
    
    print("\n\n7. Allocation Stats (if tracemalloc available)")
    print("-" * 30)
    
    alloc_stats = detector.get_allocation_stats()
    if alloc_stats:
        print(f"Traced memory: {alloc_stats['traced_memory_mb']} MB")
        print(f"Peak memory: {alloc_stats['peak_memory_mb']} MB")
    else:
        print("Tracemalloc not available")
    
    print("\n\n8. Detector Status")
    print("-" * 30)
    
    status = detector.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print("\n\n9. Snapshot History")
    print("-" * 30)
    
    for snap in detector.get_snapshots():
        print(f"  {snap['timestamp']}: {snap['total_objects']} objects")
    
    print("\n\n10. Cleanup")
    print("-" * 30)
    
    del leaked_lists
    del leaked_dicts
    gc.collect()
    
    snapshot3 = detector.take_snapshot()
    print(f"After cleanup: {snapshot3.total_objects} objects")
    print(f"Reduction: {snapshot2.total_objects - snapshot3.total_objects}")
    
    print("\n\n✓ Leak detector demo complete")
