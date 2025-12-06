"""
CineMatch V2.1.6 - Dependency Injection Container

Lightweight dependency injection framework for managing object lifecycles
and dependencies throughout the application.

Author: CineMatch Development Team
"""

from abc import ABC, abstractmethod
from typing import (
    Dict, Type, TypeVar, Generic, Optional, Any, Callable, 
    List, Union, get_type_hints, Set
)
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import threading
import inspect
import logging
from contextlib import contextmanager
import weakref


logger = logging.getLogger(__name__)

T = TypeVar('T')


class Lifetime(Enum):
    """Service lifetime options"""
    TRANSIENT = "transient"     # New instance every time
    SINGLETON = "singleton"     # Single instance for application
    SCOPED = "scoped"          # Single instance per scope


@dataclass
class ServiceDescriptor:
    """Describes how to create and manage a service"""
    service_type: Type
    implementation_type: Optional[Type] = None
    factory: Optional[Callable[..., Any]] = None
    instance: Optional[Any] = None
    lifetime: Lifetime = Lifetime.TRANSIENT
    
    def __post_init__(self):
        if self.implementation_type is None and self.factory is None and self.instance is None:
            self.implementation_type = self.service_type


class DependencyError(Exception):
    """Raised when dependency resolution fails"""
    pass


class CircularDependencyError(DependencyError):
    """Raised when circular dependency detected"""
    pass


class ServiceNotFoundError(DependencyError):
    """Raised when service is not registered"""
    pass


class IServiceProvider(ABC):
    """Abstract interface for service resolution"""
    
    @abstractmethod
    def get_service(self, service_type: Type[T]) -> Optional[T]:
        """Get service or None if not registered"""
        pass
    
    @abstractmethod
    def get_required_service(self, service_type: Type[T]) -> T:
        """Get service or raise if not registered"""
        pass


class IServiceScope(ABC):
    """Abstract interface for scoped services"""
    
    @property
    @abstractmethod
    def service_provider(self) -> IServiceProvider:
        """Get scoped service provider"""
        pass
    
    @abstractmethod
    def dispose(self) -> None:
        """Dispose scoped services"""
        pass


class ServiceCollection:
    """
    Collection of service descriptors for configuring DI container.
    
    Fluent API for registering services before building container.
    """
    
    def __init__(self):
        self._descriptors: Dict[Type, ServiceDescriptor] = {}
        self._aliases: Dict[Type, Type] = {}  # Interface -> Implementation mapping
    
    def add_transient(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
        factory: Optional[Callable[..., T]] = None
    ) -> 'ServiceCollection':
        """Register transient service (new instance each time)"""
        self._add_service(service_type, implementation_type, factory, Lifetime.TRANSIENT)
        return self
    
    def add_singleton(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
        factory: Optional[Callable[..., T]] = None,
        instance: Optional[T] = None
    ) -> 'ServiceCollection':
        """Register singleton service (single instance)"""
        self._add_service(service_type, implementation_type, factory, Lifetime.SINGLETON, instance)
        return self
    
    def add_scoped(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
        factory: Optional[Callable[..., T]] = None
    ) -> 'ServiceCollection':
        """Register scoped service (per-scope instance)"""
        self._add_service(service_type, implementation_type, factory, Lifetime.SCOPED)
        return self
    
    def _add_service(
        self,
        service_type: Type,
        implementation_type: Optional[Type],
        factory: Optional[Callable],
        lifetime: Lifetime,
        instance: Optional[Any] = None
    ) -> None:
        """Add service descriptor"""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            instance=instance,
            lifetime=lifetime
        )
        self._descriptors[service_type] = descriptor
        
        # Register alias if different implementation
        if implementation_type and implementation_type != service_type:
            self._aliases[service_type] = implementation_type
        
        logger.debug(
            f"Registered {service_type.__name__} as {lifetime.value}"
            + (f" -> {implementation_type.__name__}" if implementation_type else "")
        )
    
    def add_instance(self, service_type: Type[T], instance: T) -> 'ServiceCollection':
        """Register existing instance as singleton"""
        return self.add_singleton(service_type, instance=instance)
    
    def try_add_transient(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None
    ) -> 'ServiceCollection':
        """Register if not already registered"""
        if service_type not in self._descriptors:
            self.add_transient(service_type, implementation_type)
        return self
    
    def try_add_singleton(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None
    ) -> 'ServiceCollection':
        """Register if not already registered"""
        if service_type not in self._descriptors:
            self.add_singleton(service_type, implementation_type)
        return self
    
    def replace(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
        lifetime: Lifetime = Lifetime.TRANSIENT
    ) -> 'ServiceCollection':
        """Replace existing registration"""
        if lifetime == Lifetime.TRANSIENT:
            self.add_transient(service_type, implementation_type)
        elif lifetime == Lifetime.SINGLETON:
            self.add_singleton(service_type, implementation_type)
        else:
            self.add_scoped(service_type, implementation_type)
        return self
    
    def build(self) -> 'ServiceProvider':
        """Build service provider from collection"""
        return ServiceProvider(self._descriptors.copy())
    
    def __contains__(self, service_type: Type) -> bool:
        return service_type in self._descriptors


class ServiceScope(IServiceScope):
    """Scoped service container for managing per-request services"""
    
    def __init__(self, provider: 'ServiceProvider'):
        self._provider = provider
        self._scoped_instances: Dict[Type, Any] = {}
        self._disposed = False
        self._lock = threading.RLock()
    
    @property
    def service_provider(self) -> IServiceProvider:
        return ScopedServiceProvider(self._provider, self)
    
    def get_or_create(self, service_type: Type[T], factory: Callable[[], T]) -> T:
        """Get existing scoped instance or create new one"""
        if self._disposed:
            raise DependencyError("Cannot resolve from disposed scope")
        
        with self._lock:
            if service_type not in self._scoped_instances:
                self._scoped_instances[service_type] = factory()
            return self._scoped_instances[service_type]
    
    def dispose(self) -> None:
        """Dispose all scoped instances"""
        with self._lock:
            for instance in self._scoped_instances.values():
                if hasattr(instance, 'dispose'):
                    try:
                        instance.dispose()
                    except Exception as e:
                        logger.warning(f"Error disposing {type(instance)}: {e}")
                elif hasattr(instance, 'close'):
                    try:
                        instance.close()
                    except Exception as e:
                        logger.warning(f"Error closing {type(instance)}: {e}")
            
            self._scoped_instances.clear()
            self._disposed = True
    
    def __enter__(self) -> 'ServiceScope':
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.dispose()


class ScopedServiceProvider(IServiceProvider):
    """Service provider for scoped resolution"""
    
    def __init__(self, root_provider: 'ServiceProvider', scope: ServiceScope):
        self._root = root_provider
        self._scope = scope
    
    def get_service(self, service_type: Type[T]) -> Optional[T]:
        return self._root._resolve(service_type, self._scope)
    
    def get_required_service(self, service_type: Type[T]) -> T:
        service = self.get_service(service_type)
        if service is None:
            raise ServiceNotFoundError(f"Service {service_type.__name__} not registered")
        return service


class ServiceProvider(IServiceProvider):
    """
    Main dependency injection container.
    
    Resolves dependencies with support for constructor injection,
    factory functions, and multiple lifetimes.
    """
    
    def __init__(self, descriptors: Dict[Type, ServiceDescriptor]):
        self._descriptors = descriptors
        self._singletons: Dict[Type, Any] = {}
        self._resolving: Set[Type] = set()  # Track resolution stack
        self._lock = threading.RLock()
    
    def get_service(self, service_type: Type[T]) -> Optional[T]:
        """Get service or None if not registered"""
        try:
            return self._resolve(service_type)
        except ServiceNotFoundError:
            return None
    
    def get_required_service(self, service_type: Type[T]) -> T:
        """Get service or raise if not registered"""
        service = self._resolve(service_type)
        if service is None:
            raise ServiceNotFoundError(f"Service {service_type.__name__} not registered")
        return service
    
    def create_scope(self) -> ServiceScope:
        """Create new service scope"""
        return ServiceScope(self)
    
    @contextmanager
    def scope(self):
        """Context manager for scoped resolution"""
        scope = self.create_scope()
        try:
            yield scope.service_provider
        finally:
            scope.dispose()
    
    def _resolve(
        self,
        service_type: Type[T],
        scope: Optional[ServiceScope] = None
    ) -> T:
        """Resolve service with full dependency injection"""
        
        # Check for circular dependency
        if service_type in self._resolving:
            cycle = " -> ".join(t.__name__ for t in self._resolving)
            raise CircularDependencyError(
                f"Circular dependency detected: {cycle} -> {service_type.__name__}"
            )
        
        # Get descriptor
        descriptor = self._descriptors.get(service_type)
        if descriptor is None:
            raise ServiceNotFoundError(f"Service {service_type.__name__} not registered")
        
        # Handle existing singleton instance
        if descriptor.lifetime == Lifetime.SINGLETON:
            if descriptor.instance is not None:
                return descriptor.instance
            
            with self._lock:
                if service_type in self._singletons:
                    return self._singletons[service_type]
        
        # Handle scoped
        if descriptor.lifetime == Lifetime.SCOPED:
            if scope is None:
                raise DependencyError(
                    f"Cannot resolve scoped service {service_type.__name__} without scope"
                )
            return scope.get_or_create(
                service_type,
                lambda: self._create_instance(descriptor, scope)
            )
        
        # Track resolution to detect cycles
        self._resolving.add(service_type)
        try:
            instance = self._create_instance(descriptor, scope)
            
            # Cache singleton
            if descriptor.lifetime == Lifetime.SINGLETON:
                with self._lock:
                    self._singletons[service_type] = instance
            
            return instance
        finally:
            self._resolving.discard(service_type)
    
    def _create_instance(
        self,
        descriptor: ServiceDescriptor,
        scope: Optional[ServiceScope]
    ) -> Any:
        """Create service instance"""
        
        # Use factory if provided
        if descriptor.factory is not None:
            return self._call_with_injection(descriptor.factory, scope)
        
        # Use implementation type
        impl_type = descriptor.implementation_type or descriptor.service_type
        return self._construct(impl_type, scope)
    
    def _construct(self, impl_type: Type, scope: Optional[ServiceScope]) -> Any:
        """Construct instance with constructor injection"""
        
        # Get constructor parameters
        try:
            init_signature = inspect.signature(impl_type.__init__)
        except (ValueError, TypeError):
            # No explicit __init__, use default constructor
            return impl_type()
        
        # Get type hints for parameters
        try:
            hints = get_type_hints(impl_type.__init__)
        except Exception:
            hints = {}
        
        # Resolve dependencies
        kwargs = {}
        for name, param in init_signature.parameters.items():
            if name == 'self':
                continue
            
            # Get parameter type
            param_type = hints.get(name)
            
            if param_type is None:
                # Can't inject without type hint
                if param.default == inspect.Parameter.empty:
                    raise DependencyError(
                        f"Cannot resolve parameter '{name}' of {impl_type.__name__}: "
                        "missing type hint"
                    )
                continue  # Use default value
            
            # Try to resolve dependency
            try:
                kwargs[name] = self._resolve(param_type, scope)
            except ServiceNotFoundError:
                if param.default == inspect.Parameter.empty:
                    raise DependencyError(
                        f"Cannot resolve dependency '{name}: {param_type.__name__}' "
                        f"for {impl_type.__name__}"
                    )
                # Use default value
        
        return impl_type(**kwargs)
    
    def _call_with_injection(
        self,
        func: Callable,
        scope: Optional[ServiceScope]
    ) -> Any:
        """Call function with injected parameters"""
        try:
            signature = inspect.signature(func)
            hints = get_type_hints(func)
        except Exception:
            return func()
        
        kwargs = {}
        for name, param in signature.parameters.items():
            param_type = hints.get(name)
            
            if param_type is None:
                continue
            
            try:
                kwargs[name] = self._resolve(param_type, scope)
            except ServiceNotFoundError:
                if param.default == inspect.Parameter.empty:
                    raise
        
        return func(**kwargs)


# =============================================================================
# Decorators
# =============================================================================

def injectable(cls: Type[T]) -> Type[T]:
    """
    Mark class as injectable.
    
    This is primarily for documentation and IDE support.
    """
    cls._injectable = True
    return cls


def inject(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to inject dependencies into function.
    
    Requires ServiceProvider to be passed or available in context.
    """
    @wraps(func)
    def wrapper(*args, provider: Optional[ServiceProvider] = None, **kwargs):
        if provider is None:
            return func(*args, **kwargs)
        
        # Get unresolved parameters
        signature = inspect.signature(func)
        hints = get_type_hints(func)
        
        for name, param in signature.parameters.items():
            if name in kwargs or name == 'provider':
                continue
            
            param_type = hints.get(name)
            if param_type is None:
                continue
            
            try:
                kwargs[name] = provider.get_required_service(param_type)
            except ServiceNotFoundError:
                if param.default == inspect.Parameter.empty:
                    raise
        
        return func(*args, **kwargs)
    
    return wrapper


# =============================================================================
# Application Builder
# =============================================================================

class ApplicationBuilder:
    """
    Builder pattern for configuring and building application.
    
    Provides fluent API for DI configuration.
    """
    
    def __init__(self):
        self._services = ServiceCollection()
        self._configuration: Dict[str, Any] = {}
        self._startup_actions: List[Callable[[ServiceProvider], None]] = []
        self._shutdown_actions: List[Callable[[ServiceProvider], None]] = []
    
    @property
    def services(self) -> ServiceCollection:
        """Get service collection for registration"""
        return self._services
    
    def configure(self, key: str, value: Any) -> 'ApplicationBuilder':
        """Add configuration value"""
        self._configuration[key] = value
        return self
    
    def configure_services(
        self,
        configure_func: Callable[[ServiceCollection], None]
    ) -> 'ApplicationBuilder':
        """Configure services using callback"""
        configure_func(self._services)
        return self
    
    def on_startup(
        self,
        action: Callable[[ServiceProvider], None]
    ) -> 'ApplicationBuilder':
        """Register startup action"""
        self._startup_actions.append(action)
        return self
    
    def on_shutdown(
        self,
        action: Callable[[ServiceProvider], None]
    ) -> 'ApplicationBuilder':
        """Register shutdown action"""
        self._shutdown_actions.append(action)
        return self
    
    def build(self) -> 'Application':
        """Build application"""
        provider = self._services.build()
        return Application(
            provider=provider,
            configuration=self._configuration,
            startup_actions=self._startup_actions,
            shutdown_actions=self._shutdown_actions
        )


class Application:
    """
    Configured application with DI container.
    
    Manages application lifecycle and service resolution.
    """
    
    def __init__(
        self,
        provider: ServiceProvider,
        configuration: Dict[str, Any],
        startup_actions: List[Callable[[ServiceProvider], None]],
        shutdown_actions: List[Callable[[ServiceProvider], None]]
    ):
        self._provider = provider
        self._configuration = configuration
        self._startup_actions = startup_actions
        self._shutdown_actions = shutdown_actions
        self._started = False
    
    @property
    def services(self) -> ServiceProvider:
        """Get service provider"""
        return self._provider
    
    @property
    def configuration(self) -> Dict[str, Any]:
        """Get configuration"""
        return self._configuration
    
    def start(self) -> None:
        """Run startup actions"""
        if self._started:
            return
        
        for action in self._startup_actions:
            action(self._provider)
        
        self._started = True
        logger.info("Application started")
    
    def stop(self) -> None:
        """Run shutdown actions"""
        if not self._started:
            return
        
        for action in reversed(self._shutdown_actions):
            try:
                action(self._provider)
            except Exception as e:
                logger.error(f"Error in shutdown action: {e}")
        
        self._started = False
        logger.info("Application stopped")
    
    def __enter__(self) -> 'Application':
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


# =============================================================================
# Helper Functions
# =============================================================================

def create_service_collection() -> ServiceCollection:
    """Create new service collection"""
    return ServiceCollection()


def create_container(
    configure: Callable[[ServiceCollection], None]
) -> ServiceProvider:
    """Create container with configuration callback"""
    services = ServiceCollection()
    configure(services)
    return services.build()
