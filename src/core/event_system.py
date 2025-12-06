"""
CineMatch V2.1.6 - Event System

Event-driven architecture implementation with pub/sub pattern,
async support, and event sourcing capabilities.

Author: CineMatch Development Team
"""

from abc import ABC, abstractmethod
from typing import (
    Dict, List, Callable, Any, TypeVar, Generic, Optional,
    Type, Union, Set, Awaitable
)
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading
import asyncio
import uuid
import logging
import weakref
from functools import wraps
from queue import Queue, Empty
import json


logger = logging.getLogger(__name__)

T = TypeVar('T', bound='Event')


# =============================================================================
# Event Base Classes
# =============================================================================

@dataclass
class Event:
    """Base event class"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def event_type(self) -> str:
        """Get event type name"""
        return self.__class__.__name__
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize event to dictionary"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'timestamp': self.timestamp.isoformat(),
            'correlation_id': self.correlation_id,
            'causation_id': self.causation_id,
            'metadata': self.metadata,
            'data': self._get_data()
        }
    
    def _get_data(self) -> Dict[str, Any]:
        """Get event-specific data (override in subclasses)"""
        return {}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Deserialize event from dictionary"""
        return cls(
            event_id=data.get('event_id', str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.utcnow(),
            correlation_id=data.get('correlation_id'),
            causation_id=data.get('causation_id'),
            metadata=data.get('metadata', {})
        )


# =============================================================================
# Domain Events
# =============================================================================

@dataclass
class RecommendationRequestedEvent(Event):
    """Event when recommendation is requested"""
    user_id: int = 0
    algorithm: str = ""
    count: int = 10
    filters: Dict[str, Any] = field(default_factory=dict)
    
    def _get_data(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'algorithm': self.algorithm,
            'count': self.count,
            'filters': self.filters
        }


@dataclass
class RecommendationGeneratedEvent(Event):
    """Event when recommendations are generated"""
    user_id: int = 0
    algorithm: str = ""
    movie_ids: List[int] = field(default_factory=list)
    processing_time_ms: float = 0.0
    
    def _get_data(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'algorithm': self.algorithm,
            'movie_ids': self.movie_ids,
            'processing_time_ms': self.processing_time_ms
        }


@dataclass
class RatingAddedEvent(Event):
    """Event when user rates a movie"""
    user_id: int = 0
    movie_id: int = 0
    rating: float = 0.0
    previous_rating: Optional[float] = None
    
    def _get_data(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'movie_id': self.movie_id,
            'rating': self.rating,
            'previous_rating': self.previous_rating
        }


@dataclass
class SearchPerformedEvent(Event):
    """Event when search is performed"""
    query: str = ""
    filters: Dict[str, Any] = field(default_factory=dict)
    result_count: int = 0
    processing_time_ms: float = 0.0
    
    def _get_data(self) -> Dict[str, Any]:
        return {
            'query': self.query,
            'filters': self.filters,
            'result_count': self.result_count,
            'processing_time_ms': self.processing_time_ms
        }


@dataclass
class ModelTrainingStartedEvent(Event):
    """Event when model training starts"""
    algorithm: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def _get_data(self) -> Dict[str, Any]:
        return {
            'algorithm': self.algorithm,
            'parameters': self.parameters
        }


@dataclass
class ModelTrainingCompletedEvent(Event):
    """Event when model training completes"""
    algorithm: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    training_time_seconds: float = 0.0
    success: bool = True
    error: Optional[str] = None
    
    def _get_data(self) -> Dict[str, Any]:
        return {
            'algorithm': self.algorithm,
            'metrics': self.metrics,
            'training_time_seconds': self.training_time_seconds,
            'success': self.success,
            'error': self.error
        }


@dataclass
class UserSessionStartedEvent(Event):
    """Event when user session starts"""
    user_id: int = 0
    session_id: str = ""
    
    def _get_data(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'session_id': self.session_id
        }


@dataclass
class ErrorOccurredEvent(Event):
    """Event when error occurs"""
    error_code: str = ""
    error_message: str = ""
    component: str = ""
    stack_trace: Optional[str] = None
    
    def _get_data(self) -> Dict[str, Any]:
        return {
            'error_code': self.error_code,
            'error_message': self.error_message,
            'component': self.component,
            'stack_trace': self.stack_trace
        }


# =============================================================================
# Event Handler Types
# =============================================================================

# Sync handler
EventHandler = Callable[[Event], None]

# Async handler
AsyncEventHandler = Callable[[Event], Awaitable[None]]

# Handler with result
EventHandlerWithResult = Callable[[Event], Any]


@dataclass
class HandlerRegistration:
    """Registration info for event handler"""
    handler: Union[EventHandler, AsyncEventHandler]
    event_type: Type[Event]
    priority: int = 0
    is_async: bool = False
    is_weak: bool = False
    filter_func: Optional[Callable[[Event], bool]] = None


# =============================================================================
# Event Bus
# =============================================================================

class EventBus:
    """
    Central event bus for publish-subscribe pattern.
    
    Supports both synchronous and asynchronous handlers,
    with priority ordering and event filtering.
    """
    
    def __init__(self):
        self._handlers: Dict[Type[Event], List[HandlerRegistration]] = {}
        self._global_handlers: List[HandlerRegistration] = []
        self._lock = threading.RLock()
        self._event_queue: Queue = Queue()
        self._processing = False
        self._middleware: List[Callable[[Event], Event]] = []
    
    def subscribe(
        self,
        event_type: Type[T],
        handler: EventHandler,
        priority: int = 0,
        filter_func: Optional[Callable[[T], bool]] = None,
        weak: bool = False
    ) -> Callable[[], None]:
        """
        Subscribe to event type.
        
        Args:
            event_type: Type of event to subscribe to
            handler: Handler function
            priority: Handler priority (higher = earlier)
            filter_func: Optional filter function
            weak: Use weak reference (auto-cleanup when handler object is GC'd)
            
        Returns:
            Unsubscribe function
        """
        with self._lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            
            registration = HandlerRegistration(
                handler=weakref.WeakMethod(handler) if weak and hasattr(handler, '__self__') else handler,
                event_type=event_type,
                priority=priority,
                is_async=asyncio.iscoroutinefunction(handler),
                is_weak=weak,
                filter_func=filter_func
            )
            
            self._handlers[event_type].append(registration)
            self._handlers[event_type].sort(key=lambda r: -r.priority)
            
            logger.debug(f"Subscribed to {event_type.__name__}")
        
        # Return unsubscribe function
        def unsubscribe():
            self._unsubscribe(event_type, registration)
        
        return unsubscribe
    
    def subscribe_all(
        self,
        handler: EventHandler,
        priority: int = 0
    ) -> Callable[[], None]:
        """Subscribe to all events"""
        with self._lock:
            registration = HandlerRegistration(
                handler=handler,
                event_type=Event,  # Base type
                priority=priority,
                is_async=asyncio.iscoroutinefunction(handler)
            )
            
            self._global_handlers.append(registration)
            self._global_handlers.sort(key=lambda r: -r.priority)
        
        def unsubscribe():
            with self._lock:
                if registration in self._global_handlers:
                    self._global_handlers.remove(registration)
        
        return unsubscribe
    
    def _unsubscribe(
        self,
        event_type: Type[Event],
        registration: HandlerRegistration
    ) -> None:
        """Remove handler registration"""
        with self._lock:
            if event_type in self._handlers:
                handlers = self._handlers[event_type]
                if registration in handlers:
                    handlers.remove(registration)
    
    def publish(self, event: Event) -> None:
        """
        Publish event synchronously.
        
        All handlers are called in priority order.
        """
        # Apply middleware
        for middleware in self._middleware:
            event = middleware(event)
        
        # Get handlers
        handlers = self._get_handlers(type(event))
        
        # Call handlers
        for registration in handlers:
            try:
                handler = self._resolve_handler(registration)
                if handler is None:
                    continue
                
                # Check filter
                if registration.filter_func and not registration.filter_func(event):
                    continue
                
                if registration.is_async:
                    # Run async handler in event loop
                    try:
                        loop = asyncio.get_event_loop()
                        loop.run_until_complete(handler(event))
                    except RuntimeError:
                        # No event loop running
                        asyncio.run(handler(event))
                else:
                    handler(event)
                    
            except Exception as e:
                logger.error(f"Error in event handler: {e}", exc_info=True)
    
    async def publish_async(self, event: Event) -> None:
        """
        Publish event asynchronously.
        
        Async handlers are awaited, sync handlers are called normally.
        """
        # Apply middleware
        for middleware in self._middleware:
            event = middleware(event)
        
        # Get handlers
        handlers = self._get_handlers(type(event))
        
        # Call handlers
        tasks = []
        for registration in handlers:
            try:
                handler = self._resolve_handler(registration)
                if handler is None:
                    continue
                
                # Check filter
                if registration.filter_func and not registration.filter_func(event):
                    continue
                
                if registration.is_async:
                    tasks.append(handler(event))
                else:
                    handler(event)
                    
            except Exception as e:
                logger.error(f"Error in event handler: {e}", exc_info=True)
        
        # Await async handlers
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def publish_deferred(self, event: Event) -> None:
        """Queue event for later processing"""
        self._event_queue.put(event)
    
    def process_queue(self, max_events: int = 100) -> int:
        """Process queued events"""
        processed = 0
        
        while processed < max_events:
            try:
                event = self._event_queue.get_nowait()
                self.publish(event)
                processed += 1
            except Empty:
                break
        
        return processed
    
    def _get_handlers(
        self,
        event_type: Type[Event]
    ) -> List[HandlerRegistration]:
        """Get all handlers for event type including parent types"""
        handlers = []
        
        # Add global handlers
        handlers.extend(self._global_handlers)
        
        # Add type-specific handlers (including parent types)
        for registered_type, type_handlers in self._handlers.items():
            if issubclass(event_type, registered_type):
                handlers.extend(type_handlers)
        
        # Sort by priority
        handlers.sort(key=lambda r: -r.priority)
        
        return handlers
    
    def _resolve_handler(
        self,
        registration: HandlerRegistration
    ) -> Optional[Callable]:
        """Resolve handler from registration (handles weak refs)"""
        if registration.is_weak:
            if isinstance(registration.handler, weakref.WeakMethod):
                return registration.handler()
        return registration.handler
    
    def add_middleware(
        self,
        middleware: Callable[[Event], Event]
    ) -> None:
        """Add middleware to process events before handlers"""
        self._middleware.append(middleware)
    
    def clear(self) -> None:
        """Clear all handlers"""
        with self._lock:
            self._handlers.clear()
            self._global_handlers.clear()


# =============================================================================
# Event Decorators
# =============================================================================

# Global event bus instance (can be replaced with DI)
_default_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get or create default event bus"""
    global _default_bus
    if _default_bus is None:
        _default_bus = EventBus()
    return _default_bus


def set_event_bus(bus: EventBus) -> None:
    """Set default event bus"""
    global _default_bus
    _default_bus = bus


def on_event(
    event_type: Type[T],
    priority: int = 0,
    filter_func: Optional[Callable[[T], bool]] = None
):
    """
    Decorator to register event handler.
    
    Usage:
        @on_event(RecommendationRequestedEvent)
        def handle_recommendation(event: RecommendationRequestedEvent):
            ...
    """
    def decorator(func: EventHandler) -> EventHandler:
        bus = get_event_bus()
        bus.subscribe(event_type, func, priority, filter_func)
        return func
    return decorator


def emit(event: Event) -> None:
    """Emit event to default bus"""
    bus = get_event_bus()
    bus.publish(event)


async def emit_async(event: Event) -> None:
    """Emit event asynchronously to default bus"""
    bus = get_event_bus()
    await bus.publish_async(event)


# =============================================================================
# Event Store (Event Sourcing)
# =============================================================================

class IEventStore(ABC):
    """Abstract interface for event persistence"""
    
    @abstractmethod
    def append(self, stream_id: str, event: Event) -> None:
        """Append event to stream"""
        pass
    
    @abstractmethod
    def get_events(
        self,
        stream_id: str,
        from_version: int = 0,
        to_version: Optional[int] = None
    ) -> List[Event]:
        """Get events from stream"""
        pass
    
    @abstractmethod
    def get_all_events(
        self,
        from_timestamp: Optional[datetime] = None,
        event_types: Optional[List[Type[Event]]] = None
    ) -> List[Event]:
        """Get all events with optional filters"""
        pass


class InMemoryEventStore(IEventStore):
    """In-memory event store implementation"""
    
    def __init__(self):
        self._streams: Dict[str, List[Event]] = {}
        self._all_events: List[Event] = []
        self._lock = threading.RLock()
    
    def append(self, stream_id: str, event: Event) -> None:
        with self._lock:
            if stream_id not in self._streams:
                self._streams[stream_id] = []
            
            self._streams[stream_id].append(event)
            self._all_events.append(event)
    
    def get_events(
        self,
        stream_id: str,
        from_version: int = 0,
        to_version: Optional[int] = None
    ) -> List[Event]:
        with self._lock:
            events = self._streams.get(stream_id, [])
            
            if to_version is not None:
                return events[from_version:to_version]
            return events[from_version:]
    
    def get_all_events(
        self,
        from_timestamp: Optional[datetime] = None,
        event_types: Optional[List[Type[Event]]] = None
    ) -> List[Event]:
        with self._lock:
            events = self._all_events.copy()
            
            if from_timestamp:
                events = [e for e in events if e.timestamp >= from_timestamp]
            
            if event_types:
                events = [e for e in events if type(e) in event_types]
            
            return events
    
    def clear(self) -> None:
        """Clear all events"""
        with self._lock:
            self._streams.clear()
            self._all_events.clear()


# =============================================================================
# Event Aggregator
# =============================================================================

class EventAggregator:
    """
    Aggregates events for batch processing and analytics.
    
    Useful for tracking metrics and generating reports.
    """
    
    def __init__(self, event_bus: EventBus):
        self._bus = event_bus
        self._counts: Dict[str, int] = {}
        self._recent: List[Event] = []
        self._max_recent = 1000
        self._lock = threading.RLock()
        
        # Subscribe to all events
        self._unsubscribe = event_bus.subscribe_all(self._on_event)
    
    def _on_event(self, event: Event) -> None:
        """Handle incoming event"""
        with self._lock:
            # Count by type
            event_type = event.event_type
            self._counts[event_type] = self._counts.get(event_type, 0) + 1
            
            # Track recent
            self._recent.append(event)
            if len(self._recent) > self._max_recent:
                self._recent.pop(0)
    
    def get_counts(self) -> Dict[str, int]:
        """Get event counts by type"""
        with self._lock:
            return self._counts.copy()
    
    def get_recent(
        self,
        count: int = 100,
        event_type: Optional[Type[Event]] = None
    ) -> List[Event]:
        """Get recent events"""
        with self._lock:
            events = self._recent.copy()
            
            if event_type:
                events = [e for e in events if isinstance(e, event_type)]
            
            return events[-count:]
    
    def reset(self) -> None:
        """Reset aggregator"""
        with self._lock:
            self._counts.clear()
            self._recent.clear()
    
    def dispose(self) -> None:
        """Clean up"""
        self._unsubscribe()
