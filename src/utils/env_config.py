"""
CineMatch V2.1.6 - Environment Configuration Management
Task 4.6: Environment-specific configurations

Centralized configuration management with:
- Environment-based configuration loading
- Type validation and conversion
- Secret management integration
- Hot reloading support

Author: CineMatch Team
"""

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Log output formats."""
    SIMPLE = "simple"
    DETAILED = "detailed"
    JSON = "json"


class CacheSettings(BaseSettings):
    """Cache configuration settings."""
    
    enabled: bool = True
    ttl: int = Field(default=300, ge=0)
    max_size: int = Field(default=1000, ge=0)
    
    # Redis settings
    redis_enabled: bool = False
    redis_host: str = "localhost"
    redis_port: int = Field(default=6379, ge=1, le=65535)
    redis_db: int = Field(default=0, ge=0, le=15)
    redis_password: Optional[str] = None
    redis_ssl: bool = False
    redis_cluster_enabled: bool = False
    
    model_config = SettingsConfigDict(env_prefix="CACHE_")


class SecuritySettings(BaseSettings):
    """Security configuration settings."""
    
    # CORS
    cors_enabled: bool = True
    cors_origins: List[str] = ["*"]
    cors_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    cors_headers: List[str] = ["*"]
    cors_credentials: bool = False
    cors_max_age: int = 3600
    
    # Authentication
    auth_enabled: bool = False
    auth_token_expiry: int = 3600
    auth_refresh_token_expiry: int = 86400
    auth_algorithm: str = "HS256"
    
    # TLS
    tls_enabled: bool = False
    tls_cert_path: Optional[str] = None
    tls_key_path: Optional[str] = None
    tls_min_version: str = "1.2"
    
    # Rate limiting
    rate_limit_enabled: bool = False
    rate_limit_requests: int = 1000
    rate_limit_window: int = 60
    
    # Security headers
    security_headers_enabled: bool = True
    hsts_enabled: bool = False
    hsts_max_age: int = 31536000
    
    model_config = SettingsConfigDict(env_prefix="")
    
    @field_validator("cors_origins", "cors_methods", "cors_headers", mode="before")
    @classmethod
    def split_string_list(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(",")]
        return v


class MonitoringSettings(BaseSettings):
    """Monitoring and observability settings."""
    
    # Metrics
    metrics_enabled: bool = True
    metrics_port: int = Field(default=9090, ge=1, le=65535)
    metrics_path: str = "/metrics"
    
    # Tracing
    tracing_enabled: bool = False
    tracing_sample_rate: float = Field(default=0.1, ge=0, le=1)
    
    # Profiling
    profiling_enabled: bool = False
    
    # Sentry
    sentry_dsn: Optional[str] = None
    sentry_environment: Optional[str] = None
    sentry_traces_sample_rate: float = Field(default=0.1, ge=0, le=1)
    
    model_config = SettingsConfigDict(env_prefix="")


class ModelSettings(BaseSettings):
    """ML model configuration settings."""
    
    cache_enabled: bool = True
    cache_size: int = Field(default=100, ge=0)
    preload: bool = False
    default_algorithm: str = "hybrid"
    warm_up_enabled: bool = False
    
    # Model Loading Performance (Task 26-30)
    fast_load: bool = True
    use_joblib_mmap: bool = True
    skip_hash_verification: bool = True
    preload_on_startup: bool = False
    preload_background: bool = False
    preload_priority: str = "hybrid,svd,content_based,item_knn,user_knn"
    verify_cache_ttl: int = Field(default=3600, ge=0)
    
    model_config = SettingsConfigDict(env_prefix="MODEL_")


class FeatureFlags(BaseSettings):
    """Feature flag settings."""
    
    analytics: bool = True
    user_profiles: bool = True
    watchlist: bool = True
    real_time_updates: bool = False
    a_b_testing: bool = False
    
    model_config = SettingsConfigDict(env_prefix="FEATURE_")


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    url: Optional[str] = None
    echo: bool = False
    pool_size: int = Field(default=5, ge=1)
    max_overflow: int = Field(default=10, ge=0)
    pool_timeout: int = Field(default=30, ge=1)
    pool_recycle: int = Field(default=3600, ge=0)
    ssl_mode: Optional[str] = None
    
    model_config = SettingsConfigDict(env_prefix="DATABASE_")


class ServerSettings(BaseSettings):
    """Server configuration settings."""
    
    # Streamlit
    streamlit_port: int = Field(default=8501, ge=1, le=65535)
    streamlit_address: str = "0.0.0.0"
    streamlit_headless: bool = True
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_workers: int = Field(default=4, ge=1)
    api_reload: bool = False
    api_debug: bool = False
    api_timeout: int = Field(default=60, ge=1)
    
    model_config = SettingsConfigDict(env_prefix="")


class EnvSettings(BaseSettings):
    """
    Main application settings.
    
    Loads configuration from:
    1. Environment variables
    2. .env file (based on CINEMATCH_ENV)
    3. Default values
    
    Usage:
        settings = get_env_settings()
        print(settings.environment)
        print(settings.cache.redis_host)
    """
    
    # Core settings
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    log_level: LogLevel = LogLevel.INFO
    log_format: LogFormat = LogFormat.DETAILED
    
    # Application info
    app_name: str = "CineMatch"
    app_version: str = "2.1.6"
    secret_key: str = Field(default="change-me-in-production", min_length=16)
    
    # Paths
    data_dir: Path = Path("./data")
    models_dir: Path = Path("./models")
    processed_data_dir: Path = Path("./data/processed")
    ml_data_dir: Path = Path("./data/ml-32m")
    logs_dir: Path = Path("./logs")
    
    # Nested settings
    server: ServerSettings = Field(default_factory=ServerSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    
    # Resource limits
    max_recommendations: int = Field(default=100, ge=1)
    max_search_results: int = Field(default=50, ge=1)
    request_timeout: int = Field(default=30, ge=1)
    batch_size: int = Field(default=1000, ge=1)
    
    model_config = SettingsConfigDict(
        env_prefix="",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore"
    )
    
    @model_validator(mode="after")
    def validate_settings(self) -> "EnvSettings":
        """Validate settings after loading."""
        # Ensure directories exist in development
        if self.environment == Environment.DEVELOPMENT:
            for path in [self.data_dir, self.models_dir, self.logs_dir]:
                path.mkdir(parents=True, exist_ok=True)
        
        # Warn about insecure settings in production
        if self.environment == Environment.PRODUCTION:
            if self.debug:
                import warnings
                warnings.warn("Debug mode should be disabled in production")
            
            if self.secret_key == "change-me-in-production":
                raise ValueError("SECRET_KEY must be set in production")
            
            if self.security.cors_origins == ["*"]:
                import warnings
                warnings.warn("CORS should not allow all origins in production")
        
        return self
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == Environment.DEVELOPMENT
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.environment == Environment.TESTING
    
    def get_redis_url(self) -> Optional[str]:
        """Get Redis connection URL."""
        if not self.cache.redis_enabled:
            return None
        
        password_part = f":{self.cache.redis_password}@" if self.cache.redis_password else ""
        protocol = "rediss" if self.cache.redis_ssl else "redis"
        
        return f"{protocol}://{password_part}{self.cache.redis_host}:{self.cache.redis_port}/{self.cache.redis_db}"
    
    def to_dict(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Export settings as dictionary."""
        data = self.model_dump()
        
        if not include_secrets:
            # Mask sensitive values
            sensitive_keys = {"secret_key", "password", "api_key", "token", "dsn"}
            
            def mask_sensitive(d: dict) -> dict:
                for key, value in d.items():
                    if isinstance(value, dict):
                        d[key] = mask_sensitive(value)
                    elif any(s in key.lower() for s in sensitive_keys):
                        d[key] = "***MASKED***"
                return d
            
            data = mask_sensitive(data)
        
        return data


def _determine_env_file() -> Optional[str]:
    """Determine which .env file to load based on environment."""
    env = os.getenv("CINEMATCH_ENV", "development").lower()
    
    config_dir = Path(__file__).parent.parent.parent / "config"
    
    # Check for environment-specific file
    env_file = config_dir / f".env.{env}"
    if env_file.exists():
        return str(env_file)
    
    # Fall back to .env
    default_env = config_dir / ".env"
    if default_env.exists():
        return str(default_env)
    
    # Also check project root
    root_env = Path(__file__).parent.parent.parent / ".env"
    if root_env.exists():
        return str(root_env)
    
    return None


@lru_cache()
def get_env_settings() -> EnvSettings:
    """
    Get cached settings instance.
    
    Settings are loaded once and cached for performance.
    To reload, call `get_env_settings.cache_clear()`.
    
    Returns:
        EnvSettings instance
    """
    env_file = _determine_env_file()
    
    if env_file:
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file, override=True)
        except ImportError:
            pass
    
    return EnvSettings()


def reload_env_settings() -> EnvSettings:
    """
    Reload settings from environment.
    
    Clears the cache and returns fresh settings.
    
    Returns:
        Fresh EnvSettings instance
    """
    get_env_settings.cache_clear()
    return get_env_settings()


# =============================================================================
# Convenience exports
# =============================================================================

__all__ = [
    "EnvSettings",
    "Environment",
    "LogLevel",
    "LogFormat",
    "CacheSettings",
    "SecuritySettings",
    "MonitoringSettings",
    "ModelSettings",
    "FeatureFlags",
    "DatabaseSettings",
    "ServerSettings",
    "get_env_settings",
    "reload_env_settings",
]
