"""
CineMatch V2.1.6 - Structured Logging Module

Comprehensive logging system with structured output, log levels,
correlation IDs, and multiple output handlers.

Author: CineMatch Development Team
"""

from abc import ABC, abstractmethod
from typing import (
    Dict, Any, Optional, List, Callable, Union, TypeVar
)
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import threading
import logging
import json
import sys
import os
import traceback
from contextlib import contextmanager
from functools import wraps
import uuid


# =============================================================================
# Log Levels and Configuration
# =============================================================================

class LogLevel(Enum):
    """Log severity levels"""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


@dataclass
class LogConfig:
    """Logging configuration"""
    level: LogLevel = LogLevel.INFO
    format: str = "structured"  # "structured", "text", "json"
    output: str = "console"  # "console", "file", "both"
    file_path: Optional[str] = None
    max_file_size_mb: int = 100
    backup_count: int = 5
    include_timestamp: bool = True
    include_caller: bool = True
    include_correlation_id: bool = True
    colorize: bool = True
    app_name: str = "cinematch"
    environment: str = "development"


# =============================================================================
# Log Context (Thread-Local)
# =============================================================================

class LogContext:
    """
    Thread-local context for logging.
    
    Provides correlation IDs and contextual data.
    """
    
    _local = threading.local()
    
    @classmethod
    def get_correlation_id(cls) -> Optional[str]:
        """Get current correlation ID"""
        return getattr(cls._local, 'correlation_id', None)
    
    @classmethod
    def set_correlation_id(cls, correlation_id: str) -> None:
        """Set correlation ID"""
        cls._local.correlation_id = correlation_id
    
    @classmethod
    def generate_correlation_id(cls) -> str:
        """Generate and set new correlation ID"""
        correlation_id = str(uuid.uuid4())[:8]
        cls.set_correlation_id(correlation_id)
        return correlation_id
    
    @classmethod
    def get_context(cls) -> Dict[str, Any]:
        """Get all context data"""
        return getattr(cls._local, 'context', {}).copy()
    
    @classmethod
    def set_context(cls, **kwargs) -> None:
        """Set context data"""
        if not hasattr(cls._local, 'context'):
            cls._local.context = {}
        cls._local.context.update(kwargs)
    
    @classmethod
    def clear_context(cls) -> None:
        """Clear all context"""
        cls._local.correlation_id = None
        cls._local.context = {}
    
    @classmethod
    @contextmanager
    def scope(cls, **kwargs):
        """Context manager for scoped context"""
        old_context = cls.get_context()
        old_correlation_id = cls.get_correlation_id()
        
        try:
            if 'correlation_id' in kwargs:
                cls.set_correlation_id(kwargs.pop('correlation_id'))
            cls.set_context(**kwargs)
            yield
        finally:
            cls._local.context = old_context
            cls._local.correlation_id = old_correlation_id


# =============================================================================
# Structured Log Entry
# =============================================================================

@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: datetime
    level: LogLevel
    message: str
    logger_name: str
    correlation_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[Dict[str, Any]] = None
    caller: Optional[Dict[str, str]] = None
    duration_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.name,
            'message': self.message,
            'logger': self.logger_name
        }
        
        if self.correlation_id:
            result['correlation_id'] = self.correlation_id
        
        if self.context:
            result['context'] = self.context
        
        if self.exception:
            result['exception'] = self.exception
        
        if self.caller:
            result['caller'] = self.caller
        
        if self.duration_ms is not None:
            result['duration_ms'] = self.duration_ms
        
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), default=str)
    
    def to_text(self, colorize: bool = False) -> str:
        """Convert to formatted text"""
        # Color codes
        colors = {
            LogLevel.DEBUG: '\033[36m',    # Cyan
            LogLevel.INFO: '\033[32m',     # Green
            LogLevel.WARNING: '\033[33m',  # Yellow
            LogLevel.ERROR: '\033[31m',    # Red
            LogLevel.CRITICAL: '\033[35m'  # Magenta
        }
        reset = '\033[0m'
        
        level_str = self.level.name.ljust(8)
        if colorize:
            level_str = f"{colors.get(self.level, '')}{level_str}{reset}"
        
        parts = [
            self.timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            level_str,
            f"[{self.logger_name}]",
        ]
        
        if self.correlation_id:
            parts.append(f"[{self.correlation_id}]")
        
        parts.append(self.message)
        
        if self.duration_ms is not None:
            parts.append(f"({self.duration_ms:.2f}ms)")
        
        text = " ".join(parts)
        
        if self.context:
            context_str = " ".join(f"{k}={v}" for k, v in self.context.items())
            text += f" | {context_str}"
        
        if self.exception:
            text += f"\n  Exception: {self.exception.get('type')}: {self.exception.get('message')}"
            if self.exception.get('traceback'):
                text += f"\n{self.exception['traceback']}"
        
        return text


# =============================================================================
# Log Handlers
# =============================================================================

class ILogHandler(ABC):
    """Abstract log handler interface"""
    
    @abstractmethod
    def handle(self, entry: LogEntry) -> None:
        """Handle log entry"""
        pass
    
    @abstractmethod
    def flush(self) -> None:
        """Flush pending logs"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close handler"""
        pass


class ConsoleHandler(ILogHandler):
    """Console output handler"""
    
    def __init__(
        self,
        output_format: str = "text",
        colorize: bool = True,
        stream: Any = None
    ):
        self.output_format = output_format
        self.colorize = colorize and sys.stdout.isatty()
        self.stream = stream or sys.stdout
        self._lock = threading.Lock()
    
    def handle(self, entry: LogEntry) -> None:
        with self._lock:
            if self.output_format == "json":
                output = entry.to_json()
            else:
                output = entry.to_text(self.colorize)
            
            print(output, file=self.stream)
    
    def flush(self) -> None:
        self.stream.flush()
    
    def close(self) -> None:
        pass


class FileHandler(ILogHandler):
    """File output handler with rotation"""
    
    def __init__(
        self,
        file_path: str,
        output_format: str = "json",
        max_size_mb: int = 100,
        backup_count: int = 5
    ):
        self.file_path = file_path
        self.output_format = output_format
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.backup_count = backup_count
        self._file: Optional[Any] = None
        self._lock = threading.Lock()
        self._open_file()
    
    def _open_file(self) -> None:
        """Open log file"""
        os.makedirs(os.path.dirname(self.file_path) or '.', exist_ok=True)
        self._file = open(self.file_path, 'a', encoding='utf-8')
    
    def _rotate_if_needed(self) -> None:
        """Rotate log file if size exceeded"""
        if not self._file:
            return
        
        try:
            size = os.path.getsize(self.file_path)
            if size < self.max_size_bytes:
                return
            
            self._file.close()
            
            # Rotate files
            for i in range(self.backup_count - 1, 0, -1):
                old_path = f"{self.file_path}.{i}"
                new_path = f"{self.file_path}.{i + 1}"
                if os.path.exists(old_path):
                    os.replace(old_path, new_path)
            
            os.replace(self.file_path, f"{self.file_path}.1")
            self._open_file()
            
        except Exception:
            pass  # Don't fail on rotation error
    
    def handle(self, entry: LogEntry) -> None:
        with self._lock:
            self._rotate_if_needed()
            
            if self.output_format == "json":
                output = entry.to_json()
            else:
                output = entry.to_text(colorize=False)
            
            if self._file:
                self._file.write(output + '\n')
    
    def flush(self) -> None:
        with self._lock:
            if self._file:
                self._file.flush()
    
    def close(self) -> None:
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None


# =============================================================================
# Logger Implementation
# =============================================================================

class Logger:
    """
    Structured logger implementation.
    
    Provides logging methods with automatic context and caller info.
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[LogConfig] = None,
        handlers: Optional[List[ILogHandler]] = None
    ):
        self.name = name
        self.config = config or LogConfig()
        self._handlers = handlers or []
        self._level = self.config.level
        self._lock = threading.Lock()
        
        # Setup default handlers
        if not self._handlers:
            self._setup_default_handlers()
    
    def _setup_default_handlers(self) -> None:
        """Setup default handlers based on config"""
        if self.config.output in ("console", "both"):
            self._handlers.append(ConsoleHandler(
                output_format=self.config.format,
                colorize=self.config.colorize
            ))
        
        if self.config.output in ("file", "both") and self.config.file_path:
            self._handlers.append(FileHandler(
                file_path=self.config.file_path,
                output_format="json",
                max_size_mb=self.config.max_file_size_mb,
                backup_count=self.config.backup_count
            ))
    
    def _should_log(self, level: LogLevel) -> bool:
        """Check if level should be logged"""
        return level.value >= self._level.value
    
    def _get_caller_info(self) -> Optional[Dict[str, str]]:
        """Get caller file and line info"""
        if not self.config.include_caller:
            return None
        
        try:
            frame = sys._getframe(3)  # Adjust frame depth
            return {
                'file': os.path.basename(frame.f_code.co_filename),
                'line': str(frame.f_lineno),
                'function': frame.f_code.co_name
            }
        except Exception:
            return None
    
    def _create_entry(
        self,
        level: LogLevel,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        duration_ms: Optional[float] = None
    ) -> LogEntry:
        """Create log entry"""
        exc_info = None
        if exception:
            exc_info = {
                'type': type(exception).__name__,
                'message': str(exception),
                'traceback': traceback.format_exc()
            }
        
        # Merge contexts
        full_context = LogContext.get_context()
        if context:
            full_context.update(context)
        
        return LogEntry(
            timestamp=datetime.utcnow(),
            level=level,
            message=message,
            logger_name=self.name,
            correlation_id=LogContext.get_correlation_id(),
            context=full_context,
            exception=exc_info,
            caller=self._get_caller_info(),
            duration_ms=duration_ms
        )
    
    def _log(
        self,
        level: LogLevel,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        duration_ms: Optional[float] = None
    ) -> None:
        """Internal log method"""
        if not self._should_log(level):
            return
        
        entry = self._create_entry(level, message, context, exception, duration_ms)
        
        for handler in self._handlers:
            try:
                handler.handle(entry)
            except Exception as e:
                # Fallback to stderr
                print(f"Log handler error: {e}", file=sys.stderr)
    
    def debug(
        self,
        message: str,
        **context
    ) -> None:
        """Log debug message"""
        self._log(LogLevel.DEBUG, message, context)
    
    def info(
        self,
        message: str,
        **context
    ) -> None:
        """Log info message"""
        self._log(LogLevel.INFO, message, context)
    
    def warning(
        self,
        message: str,
        **context
    ) -> None:
        """Log warning message"""
        self._log(LogLevel.WARNING, message, context)
    
    def error(
        self,
        message: str,
        exception: Optional[Exception] = None,
        **context
    ) -> None:
        """Log error message"""
        self._log(LogLevel.ERROR, message, context, exception)
    
    def critical(
        self,
        message: str,
        exception: Optional[Exception] = None,
        **context
    ) -> None:
        """Log critical message"""
        self._log(LogLevel.CRITICAL, message, context, exception)
    
    def exception(
        self,
        message: str,
        exc: Exception,
        **context
    ) -> None:
        """Log exception with traceback"""
        self.error(message, exception=exc, **context)
    
    def timed(
        self,
        message: str,
        **context
    ):
        """Context manager for timing operations"""
        return TimedOperation(self, message, context)
    
    def set_level(self, level: LogLevel) -> None:
        """Set logging level"""
        self._level = level
    
    def add_handler(self, handler: ILogHandler) -> None:
        """Add log handler"""
        self._handlers.append(handler)
    
    def flush(self) -> None:
        """Flush all handlers"""
        for handler in self._handlers:
            handler.flush()
    
    def close(self) -> None:
        """Close all handlers"""
        for handler in self._handlers:
            handler.close()


class TimedOperation:
    """Context manager for timed logging"""
    
    def __init__(
        self,
        logger: Logger,
        message: str,
        context: Dict[str, Any]
    ):
        self.logger = logger
        self.message = message
        self.context = context
        self.start_time: Optional[float] = None
    
    def __enter__(self) -> 'TimedOperation':
        import time
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        import time
        duration_ms = (time.perf_counter() - self.start_time) * 1000
        
        if exc_val:
            self.logger._log(
                LogLevel.ERROR,
                f"{self.message} (failed)",
                self.context,
                exc_val,
                duration_ms
            )
        else:
            self.logger._log(
                LogLevel.INFO,
                f"{self.message} (completed)",
                self.context,
                None,
                duration_ms
            )


# =============================================================================
# Logging Decorators
# =============================================================================

def log_call(
    logger: Optional[Logger] = None,
    level: LogLevel = LogLevel.DEBUG,
    include_args: bool = True,
    include_result: bool = False
):
    """
    Decorator to log function calls.
    
    Usage:
        @log_call(my_logger)
        def my_function(x, y):
            ...
    """
    def decorator(func: Callable) -> Callable:
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            context = {'function': func.__name__}
            
            if include_args:
                context['args'] = str(args)[:100]
                context['kwargs'] = str(kwargs)[:100]
            
            logger._log(level, f"Calling {func.__name__}", context)
            
            try:
                result = func(*args, **kwargs)
                
                if include_result:
                    context['result'] = str(result)[:100]
                
                logger._log(level, f"Completed {func.__name__}", context)
                return result
                
            except Exception as e:
                logger.exception(f"Error in {func.__name__}", e, **context)
                raise
        
        return wrapper
    return decorator


def log_performance(
    logger: Optional[Logger] = None,
    threshold_ms: float = 100
):
    """
    Decorator to log slow operations.
    
    Only logs if operation exceeds threshold.
    """
    def decorator(func: Callable) -> Callable:
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            start = time.perf_counter()
            
            try:
                return func(*args, **kwargs)
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                
                if duration_ms > threshold_ms:
                    logger.warning(
                        f"Slow operation: {func.__name__}",
                        duration_ms=duration_ms,
                        threshold_ms=threshold_ms
                    )
        
        return wrapper
    return decorator


# =============================================================================
# Logger Factory
# =============================================================================

_loggers: Dict[str, Logger] = {}
_default_config: Optional[LogConfig] = None
_lock = threading.Lock()


def configure_logging(config: LogConfig) -> None:
    """Configure global logging settings"""
    global _default_config
    _default_config = config


def get_logger(name: str) -> Logger:
    """Get or create logger by name"""
    global _loggers, _default_config
    
    with _lock:
        if name not in _loggers:
            _loggers[name] = Logger(name, _default_config)
        return _loggers[name]


def get_root_logger() -> Logger:
    """Get root logger"""
    return get_logger("root")


# =============================================================================
# Integration with Standard Library
# =============================================================================

class StandardLibraryAdapter(logging.Handler):
    """
    Adapter to integrate with standard library logging.
    
    Routes standard library logs to our structured logger.
    """
    
    def __init__(self, logger: Logger):
        super().__init__()
        self.structured_logger = logger
    
    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Map standard levels to our levels
            level_map = {
                logging.DEBUG: LogLevel.DEBUG,
                logging.INFO: LogLevel.INFO,
                logging.WARNING: LogLevel.WARNING,
                logging.ERROR: LogLevel.ERROR,
                logging.CRITICAL: LogLevel.CRITICAL
            }
            
            level = level_map.get(record.levelno, LogLevel.INFO)
            message = self.format(record)
            
            exc = None
            if record.exc_info:
                exc = record.exc_info[1]
            
            self.structured_logger._log(level, message, exception=exc)
            
        except Exception:
            self.handleError(record)


def integrate_standard_logging(logger: Logger) -> None:
    """
    Redirect standard library logging to structured logger.
    
    Usage:
        integrate_standard_logging(get_logger("app"))
    """
    adapter = StandardLibraryAdapter(logger)
    logging.root.addHandler(adapter)
    logging.root.setLevel(logging.DEBUG)
