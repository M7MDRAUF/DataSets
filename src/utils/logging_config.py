"""
CineMatch V2.1.6 - Centralized Logging Configuration

Provides consistent logging configuration across all modules with:
- Configurable log levels (via environment variables)
- Structured JSON format for production
- Human-readable format for development
- Log rotation with size and time-based policies
- Separate handlers for console and file output

Usage:
    from src.utils.logging_config import configure_logging, get_logger
    
    # At application startup
    configure_logging()
    
    # In each module
    logger = get_logger(__name__)
    logger.info("Module initialized")

Author: CineMatch Development Team
Date: December 2025
"""

import json
import logging
import logging.handlers
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class LoggingConfig:
    """
    Logging configuration settings.
    
    All settings can be overridden via environment variables with CINEMATCH_ prefix.
    """
    # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    level: str = "INFO"
    
    # Output format: 'text' (human-readable) or 'json' (structured)
    format: str = "text"
    
    # Console output settings
    console_enabled: bool = True
    console_level: str = "INFO"
    
    # File output settings
    file_enabled: bool = True
    file_level: str = "DEBUG"
    file_path: str = "logs/cinematch.log"
    
    # Rotation settings
    max_bytes: int = 10 * 1024 * 1024  # 10 MB
    backup_count: int = 5
    
    # Module-specific log levels (module_name -> level)
    module_levels: Dict[str, str] = field(default_factory=dict)
    
    # Include extra context in logs
    include_hostname: bool = True
    include_process_id: bool = True
    
    @classmethod
    def from_env(cls) -> 'LoggingConfig':
        """Create configuration from environment variables."""
        config = cls()
        
        # Core settings
        config.level = os.getenv('CINEMATCH_LOG_LEVEL', config.level).upper()
        config.format = os.getenv('CINEMATCH_LOG_FORMAT', config.format).lower()
        
        # Console settings
        config.console_enabled = os.getenv(
            'CINEMATCH_LOG_CONSOLE', 'true'
        ).lower() == 'true'
        config.console_level = os.getenv(
            'CINEMATCH_LOG_CONSOLE_LEVEL', config.level
        ).upper()
        
        # File settings
        config.file_enabled = os.getenv(
            'CINEMATCH_LOG_FILE', 'true'
        ).lower() == 'true'
        config.file_level = os.getenv(
            'CINEMATCH_LOG_FILE_LEVEL', 'DEBUG'
        ).upper()
        config.file_path = os.getenv(
            'CINEMATCH_LOG_FILE_PATH', config.file_path
        )
        
        # Rotation settings
        config.max_bytes = int(os.getenv(
            'CINEMATCH_LOG_MAX_BYTES', str(config.max_bytes)
        ))
        config.backup_count = int(os.getenv(
            'CINEMATCH_LOG_BACKUP_COUNT', str(config.backup_count)
        ))
        
        # Module-specific levels (e.g., CINEMATCH_LOG_LEVEL_src.algorithms=DEBUG)
        for key, value in os.environ.items():
            if key.startswith('CINEMATCH_LOG_LEVEL_'):
                module_name = key.replace('CINEMATCH_LOG_LEVEL_', '').replace('_', '.')
                config.module_levels[module_name] = value.upper()
        
        return config


# =============================================================================
# Formatters
# =============================================================================

class TextFormatter(logging.Formatter):
    """
    Human-readable text formatter with colors for console.
    
    Format: TIMESTAMP | LEVEL | MODULE:LINE | MESSAGE
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors and sys.stdout.isatty()
        super().__init__(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with optional colors."""
        # Create a copy to avoid modifying the original
        record = logging.makeLogRecord(record.__dict__)
        
        if self.use_colors:
            color = self.COLORS.get(record.levelname, '')
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """
    Structured JSON formatter for production environments.
    
    Outputs JSON objects with consistent fields for log aggregation.
    """
    
    def __init__(
        self,
        include_hostname: bool = True,
        include_process_id: bool = True
    ):
        self.include_hostname = include_hostname
        self.include_process_id = include_process_id
        self._hostname = None
        super().__init__()
    
    @property
    def hostname(self) -> str:
        """Get hostname (cached)."""
        if self._hostname is None:
            import socket
            self._hostname = socket.gethostname()
        return self._hostname
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add optional fields
        if self.include_hostname:
            log_data['hostname'] = self.hostname
        
        if self.include_process_id:
            log_data['pid'] = record.process
            log_data['thread'] = record.thread
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields from record
        extra_fields = {
            k: v for k, v in record.__dict__.items()
            if k not in [
                'name', 'msg', 'args', 'created', 'filename', 'funcName',
                'levelname', 'levelno', 'lineno', 'module', 'msecs',
                'pathname', 'process', 'processName', 'relativeCreated',
                'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
                'message', 'taskName'
            ] and not k.startswith('_')
        }
        if extra_fields:
            log_data['extra'] = extra_fields
        
        return json.dumps(log_data, default=str)


# =============================================================================
# Filters
# =============================================================================

class ContextFilter(logging.Filter):
    """
    Filter that adds context information to log records.
    """
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.context = context or {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to record."""
        for key, value in self.context.items():
            setattr(record, key, value)
        return True


class ModuleLevelFilter(logging.Filter):
    """
    Filter that applies module-specific log levels.
    """
    
    def __init__(self, module_levels: Dict[str, str]):
        super().__init__()
        self.module_levels = {
            name: getattr(logging, level)
            for name, level in module_levels.items()
        }
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Check if record should be logged based on module level."""
        for module_name, level in self.module_levels.items():
            if record.name.startswith(module_name):
                return record.levelno >= level
        return True


# =============================================================================
# Configuration Functions
# =============================================================================

_configured = False


def configure_logging(
    config: Optional[LoggingConfig] = None,
    force: bool = False
) -> None:
    """
    Configure logging for the application.
    
    Should be called once at application startup, typically in main.py.
    Subsequent calls are ignored unless force=True.
    
    Args:
        config: LoggingConfig instance. If None, loads from environment.
        force: If True, reconfigure even if already configured.
        
    Example:
        # Basic setup from environment
        configure_logging()
        
        # Custom configuration
        config = LoggingConfig(level='DEBUG', format='json')
        configure_logging(config)
    """
    global _configured
    
    if _configured and not force:
        return
    
    # Load configuration
    if config is None:
        config = LoggingConfig.from_env()
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Allow all levels, handlers filter
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Create formatter based on format setting
    if config.format == 'json':
        console_formatter = JSONFormatter(
            include_hostname=config.include_hostname,
            include_process_id=config.include_process_id
        )
        file_formatter = JSONFormatter(
            include_hostname=config.include_hostname,
            include_process_id=config.include_process_id
        )
    else:
        console_formatter = TextFormatter(use_colors=True)
        file_formatter = TextFormatter(use_colors=False)
    
    # Console handler
    if config.console_enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, config.console_level))
        console_handler.setFormatter(console_formatter)
        
        if config.module_levels:
            console_handler.addFilter(ModuleLevelFilter(config.module_levels))
        
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if config.file_enabled:
        # Ensure log directory exists
        log_path = Path(config.file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            filename=config.file_path,
            maxBytes=config.max_bytes,
            backupCount=config.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, config.file_level))
        file_handler.setFormatter(file_formatter)
        
        if config.module_levels:
            file_handler.addFilter(ModuleLevelFilter(config.module_levels))
        
        root_logger.addHandler(file_handler)
    
    # Set module-specific levels
    for module_name, level in config.module_levels.items():
        module_logger = logging.getLogger(module_name)
        module_logger.setLevel(getattr(logging, level))
    
    # Reduce noise from third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    
    _configured = True
    
    # Log startup message
    root_logger.info(
        f"Logging configured: level={config.level}, format={config.format}, "
        f"console={config.console_enabled}, file={config.file_enabled}"
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the specified module.
    
    This is the recommended way to get a logger in CineMatch modules.
    
    Args:
        name: Module name (typically __name__)
        
    Returns:
        Configured Logger instance
        
    Example:
        logger = get_logger(__name__)
        logger.info("Processing started")
    """
    return logging.getLogger(name)


def add_context(logger: logging.Logger, **context: Any) -> logging.Logger:
    """
    Add context to a logger's records.
    
    Context fields will appear in log output (especially JSON format).
    
    Args:
        logger: Logger to add context to
        **context: Key-value pairs to add to all log records
        
    Returns:
        Logger with added context
        
    Example:
        logger = add_context(get_logger(__name__), user_id=123, session_id="abc")
        logger.info("User action")  # Will include user_id and session_id
    """
    filter_obj = ContextFilter(context)
    logger.addFilter(filter_obj)
    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds context to all log messages.
    
    Alternative to add_context() that doesn't modify the base logger.
    
    Example:
        base_logger = get_logger(__name__)
        logger = LoggerAdapter(base_logger, {'request_id': 'xyz'})
        logger.info("Processing request")
    """
    
    def process(
        self,
        msg: str,
        kwargs: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any]]:
        """Add extra context to log record."""
        extra = kwargs.get('extra', {})
        extra.update(self.extra)
        kwargs['extra'] = extra
        return msg, kwargs


# =============================================================================
# Convenience Functions
# =============================================================================

def log_function_call(logger: logging.Logger, level: int = logging.DEBUG):
    """
    Decorator to log function calls with arguments and return values.
    
    Args:
        logger: Logger to use
        level: Log level (default: DEBUG)
        
    Example:
        @log_function_call(logger)
        def process_data(data):
            return data * 2
    """
    import functools
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Log call
            args_str = ', '.join(repr(a) for a in args[:3])  # Limit args
            kwargs_str = ', '.join(f'{k}={v!r}' for k, v in list(kwargs.items())[:3])
            call_str = f"{func.__name__}({args_str}{', ' + kwargs_str if kwargs_str else ''})"
            logger.log(level, f"Calling {call_str}")
            
            try:
                result = func(*args, **kwargs)
                result_str = repr(result)[:100]  # Limit result length
                logger.log(level, f"Returned {result_str} from {func.__name__}")
                return result
            except Exception as e:
                logger.exception(f"Exception in {func.__name__}: {e}")
                raise
        
        return wrapper
    return decorator
