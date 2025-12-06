"""
CineMatch V2.1.6 - Environment Configuration

Secure configuration management for API keys and settings.
Supports environment variables and .env file.

Author: CineMatch Development Team
Date: December 5, 2025

Security Features:
    - Environment variable precedence over defaults
    - API key masking in logs
    - Validation of required settings
    - Support for key rotation
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from functools import lru_cache


# Try to load dotenv if available
try:
    from dotenv import load_dotenv
    # Load .env file from project root
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv is optional


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass


class Config:
    """
    Application configuration management.
    
    Loads settings from environment variables with sensible defaults.
    API keys are never stored in code - must be provided via environment.
    
    Environment Variables:
        TMDB_API_KEY: TMDB API key (optional - only for advanced features)
        TMDB_READ_ACCESS_TOKEN: TMDB v4 read access token (optional)
        CINEMATCH_DATA_PATH: Path to data directory
        CINEMATCH_MODELS_PATH: Path to models directory
        CINEMATCH_DEBUG: Enable debug mode (true/false)
        CINEMATCH_LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)
        CINEMATCH_MAX_RECOMMENDATIONS: Max recommendations per request
        CINEMATCH_ENABLE_MODEL_HASH_VERIFICATION: Enable hash verification
    """
    
    def __init__(self):
        """Initialize configuration from environment."""
        self._load_config()
    
    def _load_config(self):
        """Load all configuration from environment."""
        # Paths
        self.data_path = Path(os.getenv('CINEMATCH_DATA_PATH', 'data/ml-32m'))
        self.models_path = Path(os.getenv('CINEMATCH_MODELS_PATH', 'models'))
        
        # Debug & Logging
        self.debug = os.getenv('CINEMATCH_DEBUG', 'false').lower() == 'true'
        self.log_level = os.getenv('CINEMATCH_LOG_LEVEL', 'INFO').upper()
        
        # Recommendations
        self.max_recommendations = int(os.getenv('CINEMATCH_MAX_RECOMMENDATIONS', '100'))
        
        # Security
        self.enable_model_hash_verification = os.getenv(
            'CINEMATCH_ENABLE_MODEL_HASH_VERIFICATION', 'true'
        ).lower() == 'true'
        
        # TMDB API (optional - only for future features that need authenticated access)
        # Note: Current poster display uses public image CDN and doesn't require API key
        self._tmdb_api_key = os.getenv('TMDB_API_KEY')
        self._tmdb_read_access_token = os.getenv('TMDB_READ_ACCESS_TOKEN')
    
    @property
    def tmdb_api_key(self) -> Optional[str]:
        """
        Get TMDB API key.
        
        Returns None if not configured.
        Never logs or displays the actual key.
        """
        return self._tmdb_api_key
    
    @property
    def has_tmdb_api_key(self) -> bool:
        """Check if TMDB API key is configured."""
        return bool(self._tmdb_api_key)
    
    @property
    def tmdb_read_access_token(self) -> Optional[str]:
        """
        Get TMDB v4 read access token.
        
        Returns None if not configured.
        """
        return self._tmdb_read_access_token
    
    @property
    def has_tmdb_token(self) -> bool:
        """Check if TMDB token is configured."""
        return bool(self._tmdb_read_access_token)
    
    def get_masked_api_key(self) -> str:
        """
        Get masked version of API key for logging.
        
        Example: "abc123xyz" -> "abc***xyz"
        """
        if not self._tmdb_api_key:
            return "(not configured)"
        
        key = self._tmdb_api_key
        if len(key) <= 6:
            return "*" * len(key)
        
        return f"{key[:3]}***{key[-3:]}"
    
    def validate(self) -> Dict[str, Any]:
        """
        Validate configuration and return status.
        
        Returns:
            Dict with validation results
        """
        results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check paths exist
        if not self.data_path.exists():
            results['errors'].append(f"Data path not found: {self.data_path}")
            results['valid'] = False
        
        if not self.models_path.exists():
            results['warnings'].append(f"Models path not found: {self.models_path}")
        
        # Check model manifest for hash verification
        if self.enable_model_hash_verification:
            manifest_path = self.models_path / 'model_manifest.json'
            if not manifest_path.exists():
                results['warnings'].append(
                    "Hash verification enabled but model_manifest.json not found"
                )
        
        return results
    
    def as_dict(self, mask_secrets: bool = True) -> Dict[str, Any]:
        """
        Export configuration as dictionary.
        
        Args:
            mask_secrets: If True, mask API keys and tokens
        """
        return {
            'data_path': str(self.data_path),
            'models_path': str(self.models_path),
            'debug': self.debug,
            'log_level': self.log_level,
            'max_recommendations': self.max_recommendations,
            'enable_model_hash_verification': self.enable_model_hash_verification,
            'tmdb_api_key': self.get_masked_api_key() if mask_secrets else self._tmdb_api_key,
            'has_tmdb_api_key': self.has_tmdb_api_key,
            'has_tmdb_token': self.has_tmdb_token
        }
    
    def __repr__(self):
        return f"Config({self.as_dict()})"


# Singleton configuration instance
@lru_cache(maxsize=1)
def get_config() -> Config:
    """
    Get singleton configuration instance.
    
    Returns:
        Config instance
    """
    return Config()


# Convenience function
def reload_config() -> Config:
    """
    Reload configuration (clear cache and reload).
    
    Returns:
        Fresh Config instance
    """
    get_config.cache_clear()
    return get_config()


# =============================================================================
# ENVIRONMENT TEMPLATE
# =============================================================================

ENV_TEMPLATE = """# CineMatch Environment Configuration
# Copy this file to .env and fill in values

# TMDB API (optional - for future authenticated features)
# Get your API key from https://www.themoviedb.org/settings/api
# TMDB_API_KEY=your_api_key_here
# TMDB_READ_ACCESS_TOKEN=your_v4_token_here

# Data paths
CINEMATCH_DATA_PATH=data/ml-32m
CINEMATCH_MODELS_PATH=models

# Application settings
CINEMATCH_DEBUG=false
CINEMATCH_LOG_LEVEL=INFO
CINEMATCH_MAX_RECOMMENDATIONS=100

# Security settings
CINEMATCH_ENABLE_MODEL_HASH_VERIFICATION=true
"""


def create_env_template(path: Optional[Path] = None) -> Path:
    """
    Create .env.example template file.
    
    Args:
        path: Path to write template (default: project root)
        
    Returns:
        Path to created template file
    """
    if path is None:
        path = Path(__file__).parent.parent.parent / '.env.example'
    
    with open(path, 'w') as f:
        f.write(ENV_TEMPLATE)
    
    return path


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'create-template':
        template_path = create_env_template()
        print(f"Created template: {template_path}")
    else:
        config = get_config()
        print("Current Configuration:")
        print("-" * 40)
        for key, value in config.as_dict().items():
            print(f"  {key}: {value}")
        
        print("\nValidation:")
        print("-" * 40)
        results = config.validate()
        print(f"  Valid: {results['valid']}")
        for warning in results['warnings']:
            print(f"  ⚠ Warning: {warning}")
        for error in results['errors']:
            print(f"  ✗ Error: {error}")
