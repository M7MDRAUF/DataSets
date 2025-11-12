"""
CineMatch V2.0.0 - Algorithm Manager Components

Decomposed components for algorithm management with single responsibility.

Phase 3 Refactoring: Extracted from algorithm_manager.py to enable:
- Single Responsibility Principle
- Better testability for each component
- Easier maintenance and extension
- Reusable components

Author: CineMatch Development Team
Date: November 12, 2025
"""

from .algorithm_factory import AlgorithmFactory
from .lifecycle_manager import LifecycleManager
from .performance_monitor import PerformanceMonitor

__all__ = [
    'AlgorithmFactory',
    'LifecycleManager',
    'PerformanceMonitor'
]
