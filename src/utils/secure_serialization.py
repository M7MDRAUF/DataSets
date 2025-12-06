"""
CineMatch V2.1.6 - Secure Model Serialization

Security-hardened model loading to prevent arbitrary code execution.
Implements:
1. Restricted Unpickler - blocks dangerous built-in types
2. Hash verification - validates model integrity before loading
3. Safe loading wrapper - combines all security measures

Author: CineMatch Development Team
Date: December 5, 2025

Security Note:
    Standard pickle.load() is vulnerable to arbitrary code execution attacks.
    A malicious .pkl file can execute any Python code during deserialization.
    This module provides defense-in-depth against such attacks.
"""

import pickle
import hashlib
import json
import io
import warnings
import logging
from pathlib import Path
from typing import Any, Union, Optional, Set, Dict
from datetime import datetime

# Setup module logger
logger = logging.getLogger(__name__)


# Define custom SecurityWarning (builtin only has generic Warning)
class ModelSecurityWarning(UserWarning):
    """Warning for model loading security concerns."""
    pass


# =============================================================================
# RESTRICTED UNPICKLER - Defense against malicious pickle files
# =============================================================================

# Allowlist of safe modules that can be unpickled
SAFE_MODULES: Set[str] = {
    # NumPy (required for ML models)
    'numpy',
    'numpy.core.multiarray',
    'numpy.core.numeric',
    'numpy.core._multiarray_umath',
    'numpy._core.multiarray',
    'numpy._core._multiarray_umath',
    'numpy.dtype',
    'numpy.ndarray',
    
    # Pandas (required for DataFrames)
    'pandas',
    'pandas.core.frame',
    'pandas.core.series',
    'pandas.core.indexes.base',
    'pandas.core.indexes.range',
    'pandas.core.indexes.numeric',
    'pandas.core.arrays.categorical',
    'pandas._libs.lib',
    
    # Scipy (required for sparse matrices)
    'scipy.sparse',
    'scipy.sparse._csr',
    'scipy.sparse._csc',
    'scipy.sparse._coo',
    'scipy.sparse.csr_matrix',
    'scipy.sparse.csc_matrix',
    'scipy.sparse.coo_matrix',
    
    # Sklearn (required for ML models)
    'sklearn',
    'sklearn.decomposition',
    'sklearn.decomposition._truncated_svd',
    'sklearn.neighbors',
    'sklearn.metrics.pairwise',
    'sklearn.feature_extraction.text',
    'sklearn.preprocessing',
    'sklearn.utils',
    
    # Standard library safe types
    'builtins',
    'collections',
    'datetime',
    'decimal',
    'fractions',
    
    # CineMatch models (our code)
    'src.algorithms.svd_recommender',
    'src.algorithms.user_knn_recommender',
    'src.algorithms.item_knn_recommender',
    'src.algorithms.content_based_recommender',
    'src.algorithms.hybrid_recommender',
    'src.algorithms.base_recommender',
    'src.svd_model_sklearn',
}

# PERFORMANCE OPTIMIZATION: Pre-compute frozensets and lookup structures
# This converts O(n) any() checks to O(1) set lookups
SAFE_MODULES_FROZENSET: frozenset = frozenset(SAFE_MODULES)

# Pre-compute base module prefixes for faster prefix matching
# E.g., 'numpy.core.multiarray' -> 'numpy'
SAFE_MODULE_BASES: frozenset = frozenset(
    mod.split('.')[0] for mod in SAFE_MODULES
)

# Module lookup cache for find_class results (prevents repeated checks)
_MODULE_ALLOWED_CACHE: Dict[str, bool] = {}


def _is_module_allowed(module: str) -> bool:
    """
    Check if a module is in the allowlist with O(1) performance.
    
    Uses caching and optimized lookups for maximum speed.
    
    Args:
        module: The module name to check
        
    Returns:
        True if module is allowed, False otherwise
    """
    # Check cache first (most common path)
    if module in _MODULE_ALLOWED_CACHE:
        return _MODULE_ALLOWED_CACHE[module]
    
    # Direct match check (O(1))
    if module in SAFE_MODULES_FROZENSET:
        _MODULE_ALLOWED_CACHE[module] = True
        return True
    
    # Fast base module check (O(1))
    base_module = module.split('.')[0]
    if base_module not in SAFE_MODULE_BASES:
        _MODULE_ALLOWED_CACHE[module] = False
        return False
    
    # Prefix match for submodules (only if base is allowed)
    for safe_mod in SAFE_MODULES_FROZENSET:
        if module.startswith(f"{safe_mod}."):
            _MODULE_ALLOWED_CACHE[module] = True
            return True
    
    _MODULE_ALLOWED_CACHE[module] = False
    return False


# Blocklist of dangerous classes that should NEVER be unpickled
BLOCKED_CLASSES: Set[str] = {
    'os.system',
    'subprocess.Popen',
    'subprocess.call',
    'subprocess.run',
    'subprocess.check_output',
    'builtins.eval',
    'builtins.exec',
    'builtins.compile',
    'builtins.__import__',
    'builtins.open',
    'io.open',
    'codecs.open',
    'posix.system',
    'nt.system',
    'commands.getoutput',
    'popen2.popen2',
    'platform.popen',
}


class RestrictedUnpickler(pickle.Unpickler):
    """
    Restricted unpickler that only allows safe modules/classes.
    
    Prevents arbitrary code execution by blocking dangerous built-in
    types like os.system, subprocess, eval, exec, etc.
    
    Performance Optimizations (V2.1.6):
        - Uses frozenset for O(1) module lookups
        - Caches module allowed results
        - Pre-computed base module prefixes for fast rejection
    
    Security Model:
        - Uses allowlist approach (only explicitly allowed modules)
        - Blocks all code execution primitives
        - Logs warnings for unknown modules (audit trail)
    """
    
    # Class-level cache for find_class results (shared across instances)
    _find_class_cache: Dict[str, type] = {}
    
    def __init__(self, file, *, allow_unknown: bool = False, **kwargs):
        """
        Initialize restricted unpickler.
        
        Args:
            file: File-like object to read from
            allow_unknown: If True, allow unknown modules with warning (less secure)
        """
        super().__init__(file, **kwargs)
        self.allow_unknown = allow_unknown
        self._unknown_modules: Set[str] = set()
    
    def find_class(self, module: str, name: str) -> type:
        """
        Override find_class to restrict what can be unpickled.
        
        Uses optimized O(1) lookups instead of O(n) linear scans.
        
        Args:
            module: Module name
            name: Class/function name
            
        Returns:
            The class/type if allowed
            
        Raises:
            pickle.UnpicklingError: If module/class is not allowed
        """
        full_name = f"{module}.{name}"
        
        # Check class-level cache first (fastest path)
        if full_name in self._find_class_cache:
            return self._find_class_cache[full_name]
        
        # Check blocklist first (always blocked, O(1) set lookup)
        if full_name in BLOCKED_CLASSES:
            raise pickle.UnpicklingError(
                f"SECURITY: Blocked dangerous class '{full_name}'. "
                f"This may indicate a malicious model file!"
            )
        
        # Use optimized module check (O(1) with caching)
        module_allowed = _is_module_allowed(module)
        
        if module_allowed:
            result = super().find_class(module, name)
            # Cache successful lookups
            self._find_class_cache[full_name] = result
            return result
        
        # Unknown module
        if self.allow_unknown:
            self._unknown_modules.add(full_name)
            warnings.warn(
                f"Loading unknown module '{full_name}' - review for security",
                ModelSecurityWarning
            )
            return super().find_class(module, name)
        else:
            raise pickle.UnpicklingError(
                f"SECURITY: Module '{module}' is not in the allowlist. "
                f"Cannot load class '{name}'. If this is a legitimate model, "
                f"add the module to SAFE_MODULES in secure_serialization.py"
            )
    
    @property
    def unknown_modules(self) -> Set[str]:
        """Return set of unknown modules that were loaded (if allow_unknown=True)."""
        return self._unknown_modules


def restricted_loads(data: bytes, *, allow_unknown: bool = False) -> Any:
    """
    Safely load pickled data with restricted unpickler.
    
    Args:
        data: Pickled bytes
        allow_unknown: If True, allow unknown modules with warning
        
    Returns:
        Unpickled object
    """
    return RestrictedUnpickler(io.BytesIO(data), allow_unknown=allow_unknown).load()


# =============================================================================
# HASH VERIFICATION - Model integrity checking
# =============================================================================

def compute_file_hash(file_path: Union[str, Path], algorithm: str = 'sha256') -> str:
    """
    Compute cryptographic hash of a file.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm (sha256, sha384, sha512)
        
    Returns:
        Hex digest of file hash
    """
    file_path = Path(file_path)
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(8192), b''):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def create_model_manifest(model_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Create a manifest of all model files with their hashes.
    
    Args:
        model_dir: Directory containing model files
        
    Returns:
        Dict with file hashes and metadata
    """
    model_dir = Path(model_dir)
    manifest = {
        'version': '1.0',
        'created_at': datetime.now().isoformat(),
        'algorithm': 'sha256',
        'files': {}
    }
    
    for pkl_file in model_dir.glob('*.pkl'):
        manifest['files'][pkl_file.name] = {
            'hash': compute_file_hash(pkl_file),
            'size_bytes': pkl_file.stat().st_size,
            'modified': datetime.fromtimestamp(pkl_file.stat().st_mtime).isoformat()
        }
    
    return manifest


def save_model_manifest(model_dir: Union[str, Path], manifest_name: str = 'model_manifest.json') -> Path:
    """
    Create and save model manifest to disk.
    
    Args:
        model_dir: Directory containing model files
        manifest_name: Name of manifest file
        
    Returns:
        Path to created manifest file
    """
    model_dir = Path(model_dir)
    manifest = create_model_manifest(model_dir)
    
    manifest_path = model_dir / manifest_name
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return manifest_path


def verify_model_hash(
    model_path: Union[str, Path],
    manifest_path: Optional[Union[str, Path]] = None,
    expected_hash: Optional[str] = None
) -> bool:
    """
    Verify model file integrity against manifest or expected hash.
    
    Args:
        model_path: Path to model file
        manifest_path: Path to manifest file (optional)
        expected_hash: Expected SHA256 hash (optional)
        
    Returns:
        True if verification passes
        
    Raises:
        ValueError: If neither manifest nor expected_hash provided
        SecurityError: If hash verification fails
    """
    model_path = Path(model_path)
    actual_hash = compute_file_hash(model_path)
    
    if expected_hash:
        if actual_hash != expected_hash:
            raise SecurityError(
                f"Hash mismatch for {model_path.name}! "
                f"Expected: {expected_hash[:16]}... "
                f"Got: {actual_hash[:16]}... "
                f"Model file may have been tampered with!"
            )
        return True
    
    if manifest_path:
        manifest_path = Path(manifest_path)
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        if model_path.name not in manifest.get('files', {}):
            raise ValueError(f"Model {model_path.name} not found in manifest")
        
        expected = manifest['files'][model_path.name]['hash']
        if actual_hash != expected:
            raise SecurityError(
                f"Hash mismatch for {model_path.name}! "
                f"Model file may have been tampered with!"
            )
        return True
    
    raise ValueError("Must provide either manifest_path or expected_hash")


class SecurityError(Exception):
    """Raised when a security check fails."""
    pass


# =============================================================================
# SAFE LOADING WRAPPER - Main API
# =============================================================================

def load_model_secure(
    model_path: Union[str, Path],
    *,
    verify_hash: bool = True,
    manifest_path: Optional[Union[str, Path]] = None,
    expected_hash: Optional[str] = None,
    allow_unknown_modules: bool = False
) -> Any:
    """
    Securely load a pickled model with all security measures.
    
    This is the recommended way to load model files. It:
    1. Verifies file integrity via hash (if enabled)
    2. Uses restricted unpickler to block dangerous code
    3. Handles both direct instance and dict wrapper formats
    
    Args:
        model_path: Path to model file
        verify_hash: Whether to verify hash before loading
        manifest_path: Path to manifest file for hash verification
        expected_hash: Expected SHA256 hash (alternative to manifest)
        allow_unknown_modules: Allow unknown modules (less secure, logs warnings)
        
    Returns:
        Loaded model object
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        SecurityError: If hash verification fails
        pickle.UnpicklingError: If unpickling is blocked for security
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Step 1: Verify hash if requested
    if verify_hash and (manifest_path or expected_hash):
        verify_model_hash(model_path, manifest_path, expected_hash)
        logger.info(f"Hash verification passed for {model_path.name}")
    
    # Step 2: Load with restricted unpickler
    with open(model_path, 'rb') as f:
        data = f.read()
    
    loaded_data = restricted_loads(data, allow_unknown=allow_unknown_modules)
    
    # Step 3: Handle dict wrapper format
    if isinstance(loaded_data, dict) and 'model' in loaded_data:
        return loaded_data['model']
    
    return loaded_data


def load_model_legacy(model_path: Union[str, Path]) -> Any:
    """
    Load model using legacy (insecure) pickle.load().
    
    WARNING: This function is provided for backwards compatibility only.
    It is vulnerable to arbitrary code execution attacks.
    Use load_model_secure() instead for production.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Loaded model object
    """
    warnings.warn(
        "load_model_legacy() uses insecure pickle.load(). "
        "Use load_model_secure() for production systems.",
        ModelSecurityWarning
    )
    
    model_path = Path(model_path)
    
    with open(model_path, 'rb') as f:
        loaded_data = pickle.load(f)
    
    if isinstance(loaded_data, dict) and 'model' in loaded_data:
        return loaded_data['model']
    
    return loaded_data


# =============================================================================
# MIGRATION HELPERS - For transitioning existing code
# =============================================================================

def audit_pickle_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Audit a pickle file to see what modules/classes it contains.
    
    Useful for determining if a file is safe to load and what
    modules need to be added to SAFE_MODULES.
    
    Args:
        file_path: Path to pickle file
        
    Returns:
        Dict with audit results
    """
    file_path = Path(file_path)
    audit_result = {
        'file': str(file_path),
        'size_bytes': file_path.stat().st_size,
        'hash': compute_file_hash(file_path),
        'modules_found': set(),
        'blocked_found': set(),
        'unknown_found': set(),
        'safe': True,
        'error': None
    }
    
    class AuditUnpickler(pickle.Unpickler):
        def find_class(inner_self, module: str, name: str) -> type:
            full_name = f"{module}.{name}"
            audit_result['modules_found'].add(full_name)
            
            if full_name in BLOCKED_CLASSES:
                audit_result['blocked_found'].add(full_name)
                audit_result['safe'] = False
            elif not any(module == m or module.startswith(f"{m}.") for m in SAFE_MODULES):
                audit_result['unknown_found'].add(full_name)
            
            return super().find_class(module, name)
    
    try:
        with open(file_path, 'rb') as f:
            AuditUnpickler(f).load()
    except Exception as e:
        audit_result['error'] = str(e)
        audit_result['safe'] = False
    
    # Convert sets to lists for JSON serialization
    audit_result['modules_found'] = sorted(audit_result['modules_found'])
    audit_result['blocked_found'] = sorted(audit_result['blocked_found'])
    audit_result['unknown_found'] = sorted(audit_result['unknown_found'])
    
    return audit_result


def generate_manifest_for_models(model_dir: Union[str, Path] = 'models') -> None:
    """
    Generate manifest file for all models in directory.
    
    Run this after training to create hash manifest for verification.
    
    Args:
        model_dir: Path to models directory
    """
    model_dir = Path(model_dir)
    
    logger.info(f"Generating security manifest for models in {model_dir}")
    
    manifest_path = save_model_manifest(model_dir)
    
    # Read back and display
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    logger.info(f"Manifest created: {manifest_path}")
    logger.info(f"Files hashed: {len(manifest['files'])}")
    
    for filename, info in manifest['files'].items():
        size_mb = info['size_bytes'] / (1024 * 1024)
        logger.debug(f"{filename}: {info['hash'][:16]}... ({size_mb:.1f} MB)")


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python secure_serialization.py audit <file.pkl>")
        print("  python secure_serialization.py manifest <model_dir>")
        print("  python secure_serialization.py verify <file.pkl> <manifest.json>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'audit':
        if len(sys.argv) < 3:
            print("Usage: python secure_serialization.py audit <file.pkl>")
            sys.exit(1)
        
        result = audit_pickle_file(sys.argv[2])
        print(json.dumps(result, indent=2))
        
    elif command == 'manifest':
        model_dir = sys.argv[2] if len(sys.argv) > 2 else 'models'
        generate_manifest_for_models(model_dir)
        
    elif command == 'verify':
        if len(sys.argv) < 4:
            print("Usage: python secure_serialization.py verify <file.pkl> <manifest.json>")
            sys.exit(1)
        
        try:
            verify_model_hash(sys.argv[2], manifest_path=sys.argv[3])
            print("✓ Verification passed!")
        except SecurityError as e:
            print(f"✗ SECURITY ERROR: {e}")
            sys.exit(1)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
