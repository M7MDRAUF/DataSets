"""
Memory Limits Module for CineMatch V2.1.6

Implements memory limit enforcement:
- Memory quotas
- Usage limits
- Automatic throttling
- Memory budgets

Phase 6 - Task 6.6: Memory Limits
"""

import logging
import time
import threading
import gc
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import psutil for memory info
try:
    import psutil  # type: ignore
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class LimitAction(Enum):
    """Action to take when limit exceeded."""
    LOG = "log"
    GC = "gc"
    THROTTLE = "throttle"
    REJECT = "reject"
    RAISE = "raise"


class ThrottleLevel(Enum):
    """Throttle level."""
    NONE = "none"
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"
    BLOCKED = "blocked"


@dataclass
class MemoryLimitConfig:
    """Memory limit configuration."""
    soft_limit_mb: float = 500.0
    hard_limit_mb: float = 800.0
    critical_limit_mb: float = 1000.0
    check_interval: float = 5.0
    gc_on_soft_limit: bool = True
    throttle_on_hard_limit: bool = True


@dataclass
class MemoryBudget:
    """Memory budget for a component."""
    name: str
    allocated_mb: float
    used_mb: float = 0.0
    priority: int = 50  # Higher = more important
    
    @property
    def remaining_mb(self) -> float:
        return max(0, self.allocated_mb - self.used_mb)
    
    @property
    def usage_percent(self) -> float:
        if self.allocated_mb == 0:
            return 0.0
        return (self.used_mb / self.allocated_mb) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'allocated_mb': round(self.allocated_mb, 2),
            'used_mb': round(self.used_mb, 2),
            'remaining_mb': round(self.remaining_mb, 2),
            'usage_percent': round(self.usage_percent, 2),
            'priority': self.priority
        }


@dataclass
class LimitViolation:
    """Record of limit violation."""
    timestamp: float
    limit_type: str  # soft, hard, critical
    memory_mb: float
    limit_mb: float
    action_taken: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': datetime.fromtimestamp(self.timestamp).isoformat(),
            'limit_type': self.limit_type,
            'memory_mb': round(self.memory_mb, 2),
            'limit_mb': round(self.limit_mb, 2),
            'action_taken': self.action_taken
        }


class MemoryLimiter:
    """
    Enforces memory limits.
    
    Features:
    - Soft/hard/critical limits
    - Automatic GC
    - Throttling
    - Budget management
    """
    
    def __init__(self, config: Optional[MemoryLimitConfig] = None):
        self.config = config or MemoryLimitConfig()
        self._lock = threading.RLock()
        self._throttle_level = ThrottleLevel.NONE
        self._budgets: Dict[str, MemoryBudget] = {}
        self._violations: List[LimitViolation] = []
        self._callbacks: Dict[str, List[Callable[[float, str], None]]] = {
            'soft': [],
            'hard': [],
            'critical': []
        }
        self._process = psutil.Process() if PSUTIL_AVAILABLE else None
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
    
    def get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        if self._process:
            return self._process.memory_info().rss / (1024 * 1024)
        return 0.0
    
    @property
    def throttle_level(self) -> ThrottleLevel:
        return self._throttle_level
    
    def check_limits(self) -> Optional[str]:
        """
        Check memory limits.
        
        Returns limit type if exceeded, None otherwise.
        """
        memory_mb = self.get_current_memory_mb()
        
        with self._lock:
            if memory_mb >= self.config.critical_limit_mb:
                self._handle_violation('critical', memory_mb)
                return 'critical'
            
            if memory_mb >= self.config.hard_limit_mb:
                self._handle_violation('hard', memory_mb)
                return 'hard'
            
            if memory_mb >= self.config.soft_limit_mb:
                self._handle_violation('soft', memory_mb)
                return 'soft'
            
            # Under limits - reset throttle
            if self._throttle_level != ThrottleLevel.NONE:
                self._throttle_level = ThrottleLevel.NONE
                logger.info("Memory under limits, throttle reset")
        
        return None
    
    def _handle_violation(self, limit_type: str, memory_mb: float) -> None:
        """Handle limit violation."""
        action_taken = []
        
        if limit_type == 'soft':
            limit_mb = self.config.soft_limit_mb
            
            if self.config.gc_on_soft_limit:
                gc.collect()
                action_taken.append("gc")
            
            if self._throttle_level == ThrottleLevel.NONE:
                self._throttle_level = ThrottleLevel.LIGHT
                action_taken.append("throttle_light")
        
        elif limit_type == 'hard':
            limit_mb = self.config.hard_limit_mb
            
            gc.collect(2)  # Full collection
            action_taken.append("gc_full")
            
            if self.config.throttle_on_hard_limit:
                self._throttle_level = ThrottleLevel.MEDIUM
                action_taken.append("throttle_medium")
        
        else:  # critical
            limit_mb = self.config.critical_limit_mb
            
            gc.collect(2)
            action_taken.append("gc_full")
            
            self._throttle_level = ThrottleLevel.HEAVY
            action_taken.append("throttle_heavy")
        
        violation = LimitViolation(
            timestamp=time.time(),
            limit_type=limit_type,
            memory_mb=memory_mb,
            limit_mb=limit_mb,
            action_taken=", ".join(action_taken)
        )
        
        self._violations.append(violation)
        if len(self._violations) > 1000:
            self._violations = self._violations[-1000:]
        
        logger.warning(
            f"Memory limit exceeded: {limit_type} "
            f"({memory_mb:.1f}MB >= {limit_mb:.1f}MB)"
        )
        
        # Notify callbacks
        for callback in self._callbacks.get(limit_type, []):
            try:
                callback(memory_mb, limit_type)
            except Exception as e:
                logger.error(f"Limit callback error: {e}")
    
    def on_limit(
        self,
        limit_type: str,
        callback: Callable[[float, str], None]
    ) -> None:
        """Register callback for limit violation."""
        if limit_type in self._callbacks:
            self._callbacks[limit_type].append(callback)
    
    def should_throttle(self) -> float:
        """
        Get throttle delay in seconds.
        
        Returns 0 if no throttle needed.
        """
        if self._throttle_level == ThrottleLevel.NONE:
            return 0.0
        elif self._throttle_level == ThrottleLevel.LIGHT:
            return 0.01
        elif self._throttle_level == ThrottleLevel.MEDIUM:
            return 0.05
        elif self._throttle_level == ThrottleLevel.HEAVY:
            return 0.2
        elif self._throttle_level == ThrottleLevel.BLOCKED:
            return float('inf')
        return 0.0
    
    def request_memory(self, size_mb: float) -> bool:
        """
        Request memory allocation.
        
        Returns True if allowed, False if limit would be exceeded.
        """
        current = self.get_current_memory_mb()
        projected = current + size_mb
        
        if projected >= self.config.critical_limit_mb:
            return False
        
        if projected >= self.config.hard_limit_mb:
            # Try GC first
            gc.collect()
            current = self.get_current_memory_mb()
            projected = current + size_mb
            
            if projected >= self.config.hard_limit_mb:
                return False
        
        return True
    
    # Budget management
    
    def create_budget(
        self,
        name: str,
        allocated_mb: float,
        priority: int = 50
    ) -> MemoryBudget:
        """Create a memory budget."""
        with self._lock:
            budget = MemoryBudget(
                name=name,
                allocated_mb=allocated_mb,
                priority=priority
            )
            self._budgets[name] = budget
            return budget
    
    def get_budget(self, name: str) -> Optional[MemoryBudget]:
        """Get budget by name."""
        return self._budgets.get(name)
    
    def update_budget_usage(self, name: str, used_mb: float) -> bool:
        """Update budget usage."""
        with self._lock:
            budget = self._budgets.get(name)
            if not budget:
                return False
            
            budget.used_mb = used_mb
            
            if budget.used_mb > budget.allocated_mb:
                logger.warning(f"Budget '{name}' exceeded: {budget.used_mb:.1f}/{budget.allocated_mb:.1f} MB")
            
            return True
    
    def request_from_budget(self, name: str, size_mb: float) -> bool:
        """Request memory from a budget."""
        with self._lock:
            budget = self._budgets.get(name)
            if not budget:
                return False
            
            if budget.remaining_mb >= size_mb:
                budget.used_mb += size_mb
                return True
            
            return False
    
    def release_to_budget(self, name: str, size_mb: float) -> bool:
        """Release memory back to budget."""
        with self._lock:
            budget = self._budgets.get(name)
            if not budget:
                return False
            
            budget.used_mb = max(0, budget.used_mb - size_mb)
            return True
    
    def get_total_budget_usage(self) -> Dict[str, Any]:
        """Get total budget usage."""
        with self._lock:
            total_allocated = sum(b.allocated_mb for b in self._budgets.values())
            total_used = sum(b.used_mb for b in self._budgets.values())
            
            return {
                'budget_count': len(self._budgets),
                'total_allocated_mb': round(total_allocated, 2),
                'total_used_mb': round(total_used, 2),
                'budgets': {
                    name: budget.to_dict()
                    for name, budget in self._budgets.items()
                }
            }
    
    # Monitoring
    
    def start_monitoring(self) -> None:
        """Start background monitoring."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Memory limit monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._stop_monitoring.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_monitoring.wait(self.config.check_interval):
            try:
                self.check_limits()
            except Exception as e:
                logger.error(f"Memory limit check error: {e}")
    
    def get_violations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent violations."""
        with self._lock:
            return [v.to_dict() for v in self._violations[-limit:]]
    
    def get_status(self) -> Dict[str, Any]:
        """Get limiter status."""
        memory_mb = self.get_current_memory_mb()
        
        return {
            'current_mb': round(memory_mb, 2),
            'soft_limit_mb': self.config.soft_limit_mb,
            'hard_limit_mb': self.config.hard_limit_mb,
            'critical_limit_mb': self.config.critical_limit_mb,
            'throttle_level': self._throttle_level.value,
            'violation_count': len(self._violations),
            'budget_count': len(self._budgets)
        }


class MemoryGuard:
    """Context manager for memory-safe operations."""
    
    def __init__(
        self,
        limiter: MemoryLimiter,
        budget_name: Optional[str] = None,
        required_mb: float = 0.0
    ):
        self.limiter = limiter
        self.budget_name = budget_name
        self.required_mb = required_mb
        self.acquired = False
    
    def __enter__(self) -> 'MemoryGuard':
        if self.required_mb > 0:
            if self.budget_name:
                if not self.limiter.request_from_budget(self.budget_name, self.required_mb):
                    raise MemoryError(f"Budget '{self.budget_name}' exhausted")
            elif not self.limiter.request_memory(self.required_mb):
                raise MemoryError("Memory limit would be exceeded")
            
            self.acquired = True
        
        # Apply throttle
        delay = self.limiter.should_throttle()
        if delay > 0 and delay != float('inf'):
            time.sleep(delay)
        
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.acquired and self.budget_name:
            self.limiter.release_to_budget(self.budget_name, self.required_mb)


# Global limiter
_limiter: Optional[MemoryLimiter] = None


def get_memory_limiter() -> MemoryLimiter:
    """Get global memory limiter."""
    global _limiter
    if _limiter is None:
        _limiter = MemoryLimiter()
    return _limiter


def check_memory_limits() -> Optional[str]:
    """Check memory limits."""
    return get_memory_limiter().check_limits()


def request_memory(size_mb: float) -> bool:
    """Request memory allocation."""
    return get_memory_limiter().request_memory(size_mb)


def memory_guard(
    budget_name: Optional[str] = None,
    required_mb: float = 0.0
) -> MemoryGuard:
    """Create memory guard context."""
    return MemoryGuard(get_memory_limiter(), budget_name, required_mb)


if __name__ == "__main__":
    print("Memory Limits Module Demo")
    print("=" * 50)
    
    # Create limiter with low limits for demo
    limiter = MemoryLimiter(MemoryLimitConfig(
        soft_limit_mb=50.0,
        hard_limit_mb=100.0,
        critical_limit_mb=200.0
    ))
    
    print("\n1. Current Status")
    print("-" * 30)
    
    status = limiter.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print("\n\n2. Memory Budgets")
    print("-" * 30)
    
    cache_budget = limiter.create_budget("cache", allocated_mb=50.0, priority=60)
    model_budget = limiter.create_budget("models", allocated_mb=100.0, priority=80)
    temp_budget = limiter.create_budget("temp", allocated_mb=20.0, priority=30)
    
    print("Created budgets:")
    for name, budget in limiter.get_total_budget_usage()['budgets'].items():
        print(f"  {name}: {budget['allocated_mb']} MB")
    
    print("\n\n3. Budget Requests")
    print("-" * 30)
    
    # Request from budget
    success = limiter.request_from_budget("cache", 30.0)
    print(f"Request 30MB from cache: {success}")
    
    success = limiter.request_from_budget("cache", 30.0)
    print(f"Request another 30MB from cache: {success}")
    
    budget = limiter.get_budget("cache")
    print(f"Cache budget usage: {budget.used_mb}/{budget.allocated_mb} MB")
    
    print("\n\n4. Memory Guard")
    print("-" * 30)
    
    try:
        with MemoryGuard(limiter, budget_name="temp", required_mb=10.0) as guard:
            print("Inside memory guard, 10MB reserved")
            temp = limiter.get_budget("temp")
            print(f"Temp budget: {temp.used_mb}/{temp.allocated_mb} MB")
        
        temp = limiter.get_budget("temp")
        print(f"After guard: {temp.used_mb}/{temp.allocated_mb} MB")
    except MemoryError as e:
        print(f"Memory error: {e}")
    
    print("\n\n5. Budget Exceeded")
    print("-" * 30)
    
    try:
        with MemoryGuard(limiter, budget_name="temp", required_mb=50.0):
            print("This should not print")
    except MemoryError as e:
        print(f"Expected error: {e}")
    
    print("\n\n6. Limit Callbacks")
    print("-" * 30)
    
    def on_soft_limit(memory_mb: float, limit_type: str):
        print(f"  Callback: Soft limit hit at {memory_mb:.1f} MB")
    
    limiter.on_limit('soft', on_soft_limit)
    
    # Check limits (may or may not trigger based on actual memory)
    result = limiter.check_limits()
    print(f"Limit check result: {result}")
    
    print("\n\n7. Throttling")
    print("-" * 30)
    
    for level in ThrottleLevel:
        limiter._throttle_level = level
        delay = limiter.should_throttle()
        print(f"  {level.value}: {delay}s delay")
    
    limiter._throttle_level = ThrottleLevel.NONE
    
    print("\n\n8. Request Memory")
    print("-" * 30)
    
    allowed = limiter.request_memory(10.0)
    print(f"Request 10MB: {allowed}")
    
    allowed = limiter.request_memory(1000.0)
    print(f"Request 1000MB: {allowed}")
    
    print("\n\n9. Budget Summary")
    print("-" * 30)
    
    usage = limiter.get_total_budget_usage()
    print(f"Total allocated: {usage['total_allocated_mb']} MB")
    print(f"Total used: {usage['total_used_mb']} MB")
    
    print("\n\n10. Release Budget")
    print("-" * 30)
    
    limiter.release_to_budget("cache", 30.0)
    budget = limiter.get_budget("cache")
    print(f"Cache after release: {budget.used_mb}/{budget.allocated_mb} MB")
    
    print("\n\n✓ Memory limits demo complete")
