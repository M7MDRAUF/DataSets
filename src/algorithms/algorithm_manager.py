"""
CineMatch V2.1.6 - Algorithm Manager

Central management system for all recommendation algorithms.
Handles instantiation, switching, lifecycle management, and intelligent caching.

Supports horizontal scaling via external model state storage (Redis/S3).

Author: CineMatch Development Team
Date: November 7, 2025
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Any, Type, List
import warnings
import pandas as pd
import time
import threading
from enum import Enum
import logging
import gc
import os

# Suppress Streamlit warnings when running outside Streamlit context
warnings.filterwarnings('ignore', category=UserWarning, module='streamlit')
warnings.filterwarnings('ignore', message='.*ScriptRunContext.*')

# Suppress Streamlit logger warnings about missing context
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)

import streamlit as st

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.algorithms.base_recommender import BaseRecommender
from src.algorithms.svd_recommender import SVDRecommender
from src.algorithms.user_knn_recommender import UserKNNRecommender
from src.algorithms.item_knn_recommender import ItemKNNRecommender
from src.algorithms.content_based_recommender import ContentBasedRecommender
from src.algorithms.hybrid_recommender import HybridRecommender
from src.utils.error_handlers import safe_execute, validate_dataframe, handle_model_error
from src.algorithms.model_state import ModelStateManager, get_model_state


def _is_streamlit_context() -> bool:
    """Check if running in Streamlit context to avoid warnings"""
    try:
        # Temporarily suppress logging during context check
        streamlit_logger = logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context')
        original_level = streamlit_logger.level
        streamlit_logger.setLevel(logging.CRITICAL)
        
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        result = get_script_run_ctx() is not None
        
        # Restore logger level
        streamlit_logger.setLevel(original_level)
        return result
    except (ImportError, RuntimeError, AttributeError):
        # Not running in Streamlit context
        return False


def _get_memory_mb() -> float:
    """Get available system memory in MB, or -1 if unknown."""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 * 1024)
    except ImportError:
        return -1


def _log_memory(operation: str, phase: str = "") -> None:
    """Log memory usage for debugging."""
    mem_mb = _get_memory_mb()
    if mem_mb > 0:
        phase_str = f" ({phase})" if phase else ""
        logging.getLogger(__name__).info(f"Memory{phase_str} - {operation}: {mem_mb:.0f}MB available")


class AlgorithmType(Enum):
    """Enumeration of available recommendation algorithms"""
    SVD = "SVD Matrix Factorization"
    USER_KNN = "KNN User-Based"
    ITEM_KNN = "KNN Item-Based"
    CONTENT_BASED = "Content-Based Filtering"
    HYBRID = "Hybrid (Best of All)"


class AlgorithmManager:
    """
    Central manager for all recommendation algorithms.
    
    Features:
    - Lazy loading of algorithms (only load when requested)
    - Intelligent caching with Streamlit session state (L1) and external storage (L2)
    - Horizontal scaling support via ModelStateManager (Redis/S3)
    - Thread-safe algorithm switching
    - Performance monitoring and comparison
    - Graceful error handling and fallbacks
    
    Architecture:
    - L1 Cache: In-memory (self._algorithms) for fastest access
    - L2 Cache: Streamlit session state (optional, for UI persistence)
    - L3 Storage: External storage (Redis/S3) for horizontal scaling
    - File Fallback: Pre-trained .pkl files in models/ directory
    """
    
    # Singleton instance for non-Streamlit contexts
    _singleton_instance: Optional['AlgorithmManager'] = None
    _singleton_lock = threading.Lock()
    
    def __init__(self, use_external_storage: bool = True):
        """
        Initialize the Algorithm Manager.
        
        Args:
            use_external_storage: Whether to use Redis/S3 for model state (default True).
                                 Set to False for single-instance/development mode.
        """
        self._algorithms: Dict[AlgorithmType, BaseRecommender] = {}
        self._algorithm_classes: Dict[AlgorithmType, Type[BaseRecommender]] = {
            AlgorithmType.SVD: SVDRecommender,
            AlgorithmType.USER_KNN: UserKNNRecommender,
            AlgorithmType.ITEM_KNN: ItemKNNRecommender,
            AlgorithmType.CONTENT_BASED: ContentBasedRecommender,
            AlgorithmType.HYBRID: HybridRecommender
        }
        self._default_params: Dict[AlgorithmType, Dict[str, Any]] = {
            AlgorithmType.SVD: {'n_components': 100},
            AlgorithmType.USER_KNN: {'n_neighbors': 50, 'similarity_metric': 'cosine'},
            AlgorithmType.ITEM_KNN: {'n_neighbors': 30, 'similarity_metric': 'cosine', 'min_ratings': 5},
            AlgorithmType.CONTENT_BASED: {
                'genre_weight': 0.5, 
                'tag_weight': 0.3, 
                'title_weight': 0.2,
                'min_similarity': 0.01
            },
            AlgorithmType.HYBRID: {
                'svd_params': {'n_components': 100},
                'user_knn_params': {'n_neighbors': 50, 'similarity_metric': 'cosine'},
                'item_knn_params': {'n_neighbors': 30, 'similarity_metric': 'cosine', 'min_ratings': 5},
                'content_based_params': {
                    'genre_weight': 0.5, 
                    'tag_weight': 0.3, 
                    'title_weight': 0.2,
                    'min_similarity': 0.01
                },
                'weighting_strategy': 'adaptive'
            }
        }
        self._lock = threading.Lock()
        self._training_data: Optional[tuple] = None
        self._is_initialized = False
        
        # Model verification cache for performance
        # Key: model_path (str), Value: (mtime: float, size: int, verified: bool)
        # Prevents repeated hash verification on same file
        self._verified_models: Dict[str, tuple] = {}
        
        # Model load metrics for performance monitoring
        # Key: model_path (str), Value: dict with timing breakdowns
        self._load_metrics: Dict[str, Dict[str, float]] = {}
        
        # External storage support for horizontal scaling
        self._use_external_storage = use_external_storage
        self._model_state: Optional[ModelStateManager] = None
        self._instance_id = os.getenv('CINEMATCH_INSTANCE_ID', f"node_{int(time.time())}")
        
        # Background executor for non-blocking model saves
        self._save_executor: Optional[threading.Thread] = None
        self._pending_saves: List[tuple] = []  # Queue of (algorithm, algorithm_type, training_time)
        self._save_lock = threading.Lock()
        
        # Circuit breaker for external storage (prevents repeated failed attempts)
        self._external_storage_failures = 0
        self._external_storage_failure_threshold = 3
        self._external_storage_circuit_open = False
        self._external_storage_circuit_reset_time: Optional[float] = None
        self._circuit_breaker_cooldown = 300  # 5 minutes before retry
    
    def _is_external_storage_available(self) -> bool:
        """
        Check if external storage is available (circuit breaker pattern).
        
        Returns False if:
        - External storage is disabled
        - Circuit breaker is open due to repeated failures
        
        Returns True if external storage should be attempted.
        """
        if not self._use_external_storage:
            return False
            
        # Check if circuit breaker is open
        if self._external_storage_circuit_open:
            # Check if cooldown period has passed
            if self._external_storage_circuit_reset_time and time.time() > self._external_storage_circuit_reset_time:
                # Reset circuit breaker - allow retry
                self._external_storage_circuit_open = False
                self._external_storage_failures = 0
                self._external_storage_circuit_reset_time = None
                logging.getLogger(__name__).info("External storage circuit breaker reset - retrying")
            else:
                return False
                
        return True
    
    def _record_external_storage_failure(self) -> None:
        """Record an external storage failure and potentially open circuit breaker."""
        self._external_storage_failures += 1
        if self._external_storage_failures >= self._external_storage_failure_threshold:
            self._external_storage_circuit_open = True
            self._external_storage_circuit_reset_time = time.time() + self._circuit_breaker_cooldown
            logging.getLogger(__name__).warning(
                f"External storage circuit breaker OPEN after {self._external_storage_failures} failures. "
                f"Will retry in {self._circuit_breaker_cooldown}s"
            )
    
    def _record_external_storage_success(self) -> None:
        """Record a successful external storage operation - reset failure count."""
        if self._external_storage_failures > 0:
            self._external_storage_failures = 0
            logging.getLogger(__name__).info("External storage operation succeeded - failure count reset")
    
    def _background_save_model(self, algorithm: 'BaseRecommender', algorithm_type: AlgorithmType, training_time: float) -> None:
        """Background task to save model without blocking UI."""
        try:
            self._save_to_external_storage(algorithm, algorithm_type, training_time)
            self._record_external_storage_success()
        except Exception as e:
            logging.getLogger(__name__).warning(f"Background save failed for {algorithm_type.value}: {e}")
            self._record_external_storage_failure()
        finally:
            # Force garbage collection after save to free serialization buffers
            gc.collect()
    
    @property
    def model_state(self) -> ModelStateManager:
        """Get or create the ModelStateManager instance."""
        if self._model_state is None:
            self._model_state = get_model_state()
        return self._model_state
    
    @staticmethod
    def get_instance(use_external_storage: bool = True) -> 'AlgorithmManager':
        """
        Get singleton instance of AlgorithmManager.
        
        Uses Streamlit session_state when in Streamlit context for UI persistence.
        Falls back to class-level singleton for non-Streamlit contexts.
        
        Args:
            use_external_storage: Whether to enable external storage (Redis/S3)
        
        Returns:
            AlgorithmManager singleton instance
        """
        if _is_streamlit_context():
            # Use Streamlit session state for L2 caching
            if 'algorithm_manager' not in st.session_state:
                st.session_state.algorithm_manager = AlgorithmManager(use_external_storage)
            return st.session_state.algorithm_manager
        else:
            # Use class-level singleton for non-Streamlit contexts
            with AlgorithmManager._singleton_lock:
                if AlgorithmManager._singleton_instance is None:
                    AlgorithmManager._singleton_instance = AlgorithmManager(use_external_storage)
                return AlgorithmManager._singleton_instance
    
    def initialize_data(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> None:
        """
        Initialize with training data (call once when app starts).
        
        Uses reference semantics to avoid memory duplication.
        DataFrames are stored as-is since algorithms handle their own copies if needed.
        """
        # Skip if already initialized with same data size (prevents duplicate initialization)
        if self._is_initialized and self._training_data is not None:
            existing_size = len(self._training_data[0])
            new_size = len(ratings_df)
            if existing_size == new_size:
                # Already initialized with same data - skip
                return
        
        # Pre-add genres_list and poster_path to avoid repeated copies in each algorithm
        needs_copy = 'genres_list' not in movies_df.columns or 'poster_path' not in movies_df.columns
        if needs_copy:
            movies_df = movies_df.copy()  # Only copy once here
            if 'genres_list' not in movies_df.columns:
                movies_df['genres_list'] = movies_df['genres'].str.split('|')
            if 'poster_path' not in movies_df.columns:
                movies_df['poster_path'] = None
        
        # Store references (not copies) to avoid 3.3GB memory duplication
        self._training_data = (ratings_df, movies_df)
        self._is_initialized = True
        logging.getLogger(__name__).info("Algorithm Manager initialized with data")
    
    def preload_models_background(self, priority_algorithms: Optional[List[AlgorithmType]] = None) -> None:
        """
        Preload models in background thread on app startup.
        
        This improves perceived performance by loading models before user requests them.
        
        Args:
            priority_algorithms: List of algorithms to preload in order (default: Hybrid first)
        """
        if not self._is_initialized:
            logging.getLogger(__name__).warning("Cannot preload models - data not initialized")
            return
            
        if priority_algorithms is None:
            # Default priority: Hybrid (most common), then others
            priority_algorithms = [AlgorithmType.HYBRID, AlgorithmType.SVD, AlgorithmType.ITEM_KNN]
        
        def _preload_task():
            """Background task to preload models."""
            for algo_type in priority_algorithms:
                try:
                    logging.getLogger(__name__).info(f"Preloading {algo_type.name} model...")
                    self.get_algorithm(algo_type)
                    logging.getLogger(__name__).info(f"✓ {algo_type.name} preloaded")
                except Exception as e:
                    logging.getLogger(__name__).warning(f"Failed to preload {algo_type.name}: {e}")
                finally:
                    gc.collect()  # Clean up after each model load
        
        # Start preloading in background thread
        preload_thread = threading.Thread(
            target=_preload_task,
            daemon=True,
            name="model_preload"
        )
        preload_thread.start()
        logging.getLogger(__name__).info(f"Started background preloading of {len(priority_algorithms)} algorithms")
    
    def get_algorithm(self, algorithm_type: AlgorithmType, 
                     custom_params: Optional[Dict[str, Any]] = None,
                     fast_load: bool = True,
                     use_joblib: bool = True) -> BaseRecommender:
        """
        Get a trained algorithm instance with multi-level caching.
        
        Lookup order:
        1. L1 Cache (in-memory self._algorithms) - fastest
        2. L3 External Storage (Redis/S3) - for horizontal scaling
        3. File-based pre-trained models (models/*.pkl) - fallback
        4. Train from scratch - slowest
        
        Performance Optimizations (V2.1.6):
        - fast_load=True: Skips hash verification for trusted pre-shipped models
        - use_joblib=True: Uses memory-mapped loading for large models (~0.5s vs 45s)
        
        Args:
            algorithm_type: Type of algorithm to get
            custom_params: Optional custom parameters (overrides defaults)
            fast_load: If True, skip hash verification for faster loads
            use_joblib: If True, use joblib with memory mapping for large models
            
        Returns:
            Trained BaseRecommender instance
        """
        if not self._is_initialized:
            raise ValueError("AlgorithmManager not initialized. Call initialize_data() first.")
        
        with self._lock:
            # L1 Cache: Check if algorithm is already cached in memory
            if algorithm_type in self._algorithms:
                algorithm = self._algorithms[algorithm_type]
                if algorithm.is_trained:
                    # CRITICAL: Validate cache types for Content-Based (may have old dict caches)
                    if algorithm_type == AlgorithmType.CONTENT_BASED:
                        from src.utils.lru_cache import LRUCache
                        
                        # Check if caches are dicts (from old pickled instance before fix)
                        needs_reinit = (
                            not isinstance(getattr(algorithm, 'user_profiles', None), LRUCache) or
                            not isinstance(getattr(algorithm, 'movie_similarity_cache', None), LRUCache)
                        )
                        
                        if needs_reinit:
                            logging.getLogger(__name__).warning("CBF cached instance has dict caches - forcing reinitialization")
                            print("⚠️ Detected old CBF cache with dict objects - reloading model...")
                            
                            # Remove from cache to force reload with fallback
                            del self._algorithms[algorithm_type]
                            # Fall through to load/train logic below
                        else:
                            if not _is_streamlit_context():
                                print(f"✓ Using cached {algorithm.name}")
                            return algorithm
                    else:
                        if not _is_streamlit_context():
                            print(f"✓ Using cached {algorithm.name}")
                        return algorithm
            
            # Need to load or train algorithm
            if not _is_streamlit_context():
                print(f"🔄 Loading {algorithm_type.value}...")
            
            # Merge default and custom parameters
            params = self._default_params[algorithm_type].copy()
            if custom_params:
                params.update(custom_params)
            
            # Instantiate algorithm
            algorithm_class = self._algorithm_classes[algorithm_type]
            algorithm = algorithm_class(**params)
            
            # L3: Try to load from external storage (Redis/S3) for horizontal scaling
            if self._is_external_storage_available() and self._try_load_from_external_storage(algorithm, algorithm_type):
                self._algorithms[algorithm_type] = algorithm
                self._record_external_storage_success()
                return algorithm
            
            # File Fallback: Try to load pre-trained model from disk
            if self._try_load_pretrained_model(algorithm, algorithm_type, fast_load=fast_load, use_joblib=use_joblib):
                # Pre-trained model loaded successfully
                self._algorithms[algorithm_type] = algorithm
                
                # Persist to external storage for other instances (with circuit breaker check)
                if self._is_external_storage_available():
                    try:
                        self._save_to_external_storage(algorithm, algorithm_type)
                        self._record_external_storage_success()
                    except Exception as e:
                        logging.getLogger(__name__).warning(f"Failed to save to external storage (non-fatal): {e}")
                        self._record_external_storage_failure()
                        # Continue without failing - model is still cached in L1
                
                return algorithm
            
            # Train algorithm if no cached model available
            ratings_df, movies_df = self._training_data
            
            # Log memory before training
            _log_memory(f"train_{algorithm_type.name}", "before")
            
            # Show progress in Streamlit (only if in Streamlit context)
            if _is_streamlit_context():
                with st.spinner(f'Training {algorithm.name}... This may take a moment.'):
                    start_time = time.time()
                    algorithm.fit(ratings_df, movies_df)
                    training_time = time.time() - start_time
            else:
                start_time = time.time()
                algorithm.fit(ratings_df, movies_df)
                training_time = time.time() - start_time
            
            # Log memory after training
            _log_memory(f"train_{algorithm_type.name}", "after")
                
            # L1 Cache: Store the trained algorithm
            self._algorithms[algorithm_type] = algorithm
            
            # L3: Persist to external storage for horizontal scaling (non-blocking, fault-tolerant)
            # Use background thread to avoid blocking the UI (with circuit breaker check)
            if self._is_external_storage_available():
                save_thread = threading.Thread(
                    target=self._background_save_model,
                    args=(algorithm, algorithm_type, training_time),
                    daemon=True,
                    name=f"save_{algorithm_type.name}"
                )
                save_thread.start()
                # Don't wait for thread - let it run in background
            
            print(f"✓ {algorithm.name} trained in {training_time:.1f}s")
            
            # Force garbage collection after training to free memory
            gc.collect()
            
            # Show success message (only if in Streamlit context)
            if _is_streamlit_context():
                st.success(f"✅ {algorithm.name} ready! (Trained in {training_time:.1f}s)")
                
            return algorithm
    
    def _try_load_from_external_storage(
        self, 
        algorithm: BaseRecommender, 
        algorithm_type: AlgorithmType
    ) -> bool:
        """
        Try to load a model from external storage (Redis/S3).
        
        This enables horizontal scaling by sharing trained models
        across multiple instances.
        
        Args:
            algorithm: The algorithm instance to load into
            algorithm_type: The type of algorithm
            
        Returns:
            True if model was loaded successfully, False otherwise
        """
        # Hybrid doesn't have pre-trained - it uses component algorithms
        if algorithm_type == AlgorithmType.HYBRID:
            return False
        
        try:
            key = f"{algorithm_type.name.lower()}"
            
            # Check if model exists in external storage
            if not self.model_state.has_model(key):
                return False
            
            if not _is_streamlit_context():
                print(f"   • Found model '{key}' in external storage")
            
            start_time = time.time()
            loaded_model = self.model_state.load_model(key)
            load_time = time.time() - start_time
            
            if loaded_model is None:
                return False
            
            # Copy loaded model's attributes to current instance
            algorithm.__dict__.update(loaded_model.__dict__)
            
            # Provide data context to the loaded model
            ratings_df, movies_df = self._training_data
            algorithm.ratings_df = ratings_df
            algorithm.movies_df = movies_df
            if 'genres_list' not in algorithm.movies_df.columns:
                algorithm.movies_df['genres_list'] = algorithm.movies_df['genres'].str.split('|')
            if 'poster_path' not in algorithm.movies_df.columns:
                algorithm.movies_df['poster_path'] = None
            
            if algorithm.is_trained:
                if not _is_streamlit_context():
                    print(f"   ✓ Loaded {algorithm.name} from external storage in {load_time:.2f}s")
                if _is_streamlit_context():
                    st.success(f"🚀 {algorithm.name} loaded from shared storage! ({load_time:.2f}s)")
                return True
            
            return False
            
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to load from external storage: {e}")
            return False
    
    def _save_to_external_storage(
        self, 
        algorithm: BaseRecommender, 
        algorithm_type: AlgorithmType,
        training_time: float = 0.0
    ) -> bool:
        """
        Save a trained model to external storage for horizontal scaling.
        
        Args:
            algorithm: The trained algorithm to save
            algorithm_type: The type of algorithm
            training_time: Training duration in seconds
            
        Returns:
            True if saved successfully
        """
        # Hybrid doesn't get saved - it uses component algorithms
        if algorithm_type == AlgorithmType.HYBRID:
            return False
        
        try:
            key = f"{algorithm_type.name.lower()}"
            
            # Extract metrics
            metrics_obj = algorithm.metrics
            metrics = {
                'rmse': getattr(metrics_obj, 'rmse', 0.0),
                'mae': getattr(metrics_obj, 'mae', 0.0),
                'coverage': getattr(metrics_obj, 'coverage', 0.0),
            }
            
            success = self.model_state.save_model(
                key=key,
                model=algorithm,
                model_type=algorithm_type.value,
                version="2.1.6",
                training_time=training_time or getattr(metrics_obj, 'training_time', 0.0),
                metrics=metrics,
                parameters=self._default_params.get(algorithm_type, {}),
            )
            
            if success and not _is_streamlit_context():
                print(f"   ✓ Saved {algorithm.name} to external storage")
            
            return success
            
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to save to external storage: {e}")
            return False
    
    def _try_load_pretrained_model(
        self, 
        algorithm: BaseRecommender, 
        algorithm_type: AlgorithmType,
        fast_load: bool = True,
        use_joblib: bool = True
    ) -> bool:
        """
        Try to load a pre-trained model for algorithms.
        
        Performance optimizations:
        - fast_load=True: Skips hash verification (default for trusted models)
        - use_joblib=True: Uses joblib with memory-mapping for faster loads
        
        Args:
            algorithm: The algorithm instance to load into
            algorithm_type: The type of algorithm
            fast_load: If True, skip hash verification for faster loads
            use_joblib: If True, use joblib.load with mmap_mode for large models
            
        Returns:
            True if model was loaded successfully, False otherwise
        """
        # Import load_model_safe utility
        from src.utils import load_model_safe
        from src.utils.secure_serialization import verify_model_hash, SecurityError
        
        # Define model paths for all supported algorithms
        # Note: Using sklearn SVD variant (svd_model_sklearn.pkl) - faster and more memory-efficient
        model_paths = {
            AlgorithmType.SVD: Path("models/svd_model_sklearn.pkl"),  # sklearn variant (not Surprise)
            AlgorithmType.USER_KNN: Path("models/user_knn_model.pkl"),
            AlgorithmType.ITEM_KNN: Path("models/item_knn_model.pkl"),
            AlgorithmType.CONTENT_BASED: Path("models/content_based_model.pkl")
        }
        
        # Security manifest for hash verification
        manifest_path = Path("models/model_manifest.json")
        
        # Hybrid doesn't have pre-trained - it uses component algorithms
        if algorithm_type == AlgorithmType.HYBRID:
            return False
        
        # Check if algorithm has pre-trained model support
        if algorithm_type not in model_paths:
            return False
        
        model_path = model_paths.get(algorithm_type)
        if not model_path:
            return False
        
        if not model_path.exists():
            if not _is_streamlit_context():
                print(f"   • No pre-trained model found at {model_path}")
            return False
        
        try:
            # Performance metrics tracking
            metrics = {'total': 0, 'hash_verify': 0, 'file_load': 0, 'setup': 0}
            overall_start = time.time()
            
            # PERFORMANCE OPTIMIZATION: Check verification cache first
            # Avoid redundant hash verification if file hasn't changed
            model_key = str(model_path.absolute())
            current_mtime = model_path.stat().st_mtime
            current_size = model_path.stat().st_size
            
            cached_verification = self._verified_models.get(model_key)
            already_verified = False
            if cached_verification:
                cached_mtime, cached_size, was_verified = cached_verification
                if cached_mtime == current_mtime and cached_size == current_size and was_verified:
                    already_verified = True
                    if not _is_streamlit_context():
                        print(f"   ✓ Using cached verification for {model_path.name}")
            
            # PERFORMANCE OPTIMIZATION: Skip hash verification in hot path
            # Hash verification on 1GB+ models adds 3-5s overhead per model
            # Security is maintained by:
            # 1. Pre-shipped models are trusted (from official release)
            # 2. Hash verification can be enabled via fast_load=False or VERIFY_MODEL_HASH env var
            # 3. User-uploaded models should use a separate secure upload path
            import os
            verify_hash_env = os.environ.get('VERIFY_MODEL_HASH', 'false').lower() == 'true'
            verify_hash_enabled = ((not fast_load) or verify_hash_env) and not already_verified
            
            hash_start = time.time()
            if verify_hash_enabled and manifest_path.exists():
                try:
                    verify_model_hash(model_path, manifest_path=manifest_path)
                    if not _is_streamlit_context():
                        print(f"   ✓ Hash verification passed for {model_path.name}")
                    # Cache the successful verification
                    self._verified_models[model_key] = (current_mtime, current_size, True)
                except SecurityError as e:
                    print(f"   ⚠ SECURITY WARNING: {e}")
                    print(f"   → Model may have been tampered with. Refusing to load.")
                    self._verified_models[model_key] = (current_mtime, current_size, False)
                    return False
                except ValueError:
                    # Model not in manifest (new model) - proceed but warn
                    if not _is_streamlit_context():
                        print(f"   • Model not in manifest - consider regenerating manifest")
            metrics['hash_verify'] = time.time() - hash_start
            
            if not _is_streamlit_context():
                print(f"   • Loading pre-trained model from {model_path} (fast_load={fast_load})")
            load_start = time.time()
            
            # Use joblib for faster loading with memory mapping if enabled
            if use_joblib:
                try:
                    import joblib
                    loaded_model = joblib.load(str(model_path), mmap_mode='r')
                except Exception as joblib_error:
                    # Fallback to secure pickle if joblib fails
                    if not _is_streamlit_context():
                        print(f"   • Joblib failed ({joblib_error}), falling back to pickle")
                    loaded_model = load_model_safe(str(model_path))
            else:
                # Use secure pickle loader
                loaded_model = load_model_safe(str(model_path))
            metrics['file_load'] = time.time() - load_start
            load_time = metrics['file_load']
            
            setup_start = time.time()
            # IMPORTANT: Replace the algorithm instance with the loaded model
            # Handle wrapped models (dict with 'model' key) from some training scripts
            actual_model = loaded_model
            if isinstance(loaded_model, dict) and 'model' in loaded_model:
                actual_model = loaded_model['model']
                if not _is_streamlit_context():
                    print(f"   • Unwrapped model from dict container")
            
            # Copy the loaded model's attributes to the current algorithm instance
            algorithm.__dict__.update(actual_model.__dict__)
            
            # IMPORTANT: Provide data context to the loaded model
            # Pre-trained models need access to current data for some operations
            # Using shallow references (not copies) since models are already trained
            ratings_df, movies_df = self._training_data
            algorithm.ratings_df = ratings_df  # Shallow reference (model is pre-trained, won't modify)
            algorithm.movies_df = movies_df    # Shallow reference
            # Add genres_list and poster_path if not present
            if 'genres_list' not in algorithm.movies_df.columns:
                algorithm.movies_df['genres_list'] = algorithm.movies_df['genres'].str.split('|')
            if 'poster_path' not in algorithm.movies_df.columns:
                algorithm.movies_df['poster_path'] = None
            if not _is_streamlit_context():
                print(f"   • Data context provided to loaded model")
            metrics['setup'] = time.time() - setup_start
            
            # Update verification cache for successful load
            if model_key not in self._verified_models:
                self._verified_models[model_key] = (current_mtime, current_size, True)
            
            # Store load metrics for performance monitoring
            metrics['total'] = time.time() - overall_start
            self._load_metrics[model_key] = metrics
            
            # Verify the model is properly loaded and trained
            if algorithm.is_trained:
                if not _is_streamlit_context():
                    print(f"   ✓ Pre-trained {algorithm.name} loaded in {load_time:.2f}s (total: {metrics['total']:.2f}s)")
                    print(f"   • Metrics: hash={metrics['hash_verify']:.2f}s, load={metrics['file_load']:.2f}s, setup={metrics['setup']:.2f}s")
                if _is_streamlit_context():
                    st.success(f"🚀 {algorithm.name} loaded from pre-trained model! ({metrics['total']:.2f}s)")
                return True
            else:
                print(f"   ⚠ Pre-trained model loaded but not marked as trained")
                return False
                
        except Exception as e:
            print(f"   ❌ Failed to load pre-trained model: {e}")
            print(f"   → Will train from scratch instead")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
            return False
    
    def get_current_algorithm(self) -> Optional[BaseRecommender]:
        """Get the currently selected algorithm from Streamlit session state"""
        if 'selected_algorithm' not in st.session_state:
            return None
        
        algorithm_type = st.session_state.selected_algorithm
        if algorithm_type in self._algorithms:
            return self._algorithms[algorithm_type]
        
        return None
    
    def switch_algorithm(self, algorithm_type: AlgorithmType, 
                        custom_params: Optional[Dict[str, Any]] = None) -> BaseRecommender:
        """
        Switch to a different algorithm with smooth transition.
        
        Args:
            algorithm_type: Algorithm to switch to
            custom_params: Optional custom parameters
            
        Returns:
            The new algorithm instance
        """
        # Only show in terminal/logs, not in Streamlit UI
        if not _is_streamlit_context():
            print(f"🔄 Switching to {algorithm_type.value}")
        
        # Store selection in session state
        st.session_state.selected_algorithm = algorithm_type
        
        # Get the algorithm (will load if not cached)
        algorithm = self.get_algorithm(algorithm_type, custom_params)
        
        # Update UI state
        if 'algorithm_switched' not in st.session_state:
            st.session_state.algorithm_switched = True
        
        # Clear caches from other algorithms to save memory
        self.clear_algorithm_cache(keep_current=algorithm_type)
        
        return algorithm
    
    def clear_algorithm_cache(self, keep_current: Optional[AlgorithmType] = None) -> None:
        """
        Clear cached data from algorithms to free memory.
        
        Args:
            keep_current: Algorithm to keep in cache, clear all others
        """
        cleared_count = 0
        
        for algo_type, algorithm in list(self._algorithms.items()):
            # Skip the current algorithm if specified
            if keep_current and algo_type == keep_current:
                continue
            
            # Clear algorithm-specific caches
            if hasattr(algorithm, 'user_profiles') and isinstance(algorithm.user_profiles, dict):
                algorithm.user_profiles.clear()
                cleared_count += 1
            
            if hasattr(algorithm, 'movie_similarity_cache') and isinstance(algorithm.movie_similarity_cache, dict):
                algorithm.movie_similarity_cache.clear()
                cleared_count += 1
        
        # Aggressive garbage collection
        if cleared_count > 0:
            self.aggressive_gc()
            if not _is_streamlit_context():
                print(f"🧹 Cleared {cleared_count} cache(s) to free memory")
    
    def aggressive_gc(self) -> None:
        """Perform aggressive garbage collection to free memory"""
        gc.collect()
        gc.collect()  # Call twice for cyclic references
        gc.collect()
    
    def get_available_algorithms(self) -> List[AlgorithmType]:
        """Get list of all available algorithm types"""
        return list(AlgorithmType)
    
    def get_algorithm_info(self, algorithm_type: AlgorithmType) -> Dict[str, Any]:
        """
        Get detailed information about an algorithm without loading it.
        
        Returns:
            Dictionary with algorithm description, capabilities, etc.
        """
        info_map = {
            AlgorithmType.SVD: {
                'name': 'SVD Matrix Factorization',
                'description': 'Uses Singular Value Decomposition to discover hidden patterns in user ratings. Excellent for finding complex relationships between users and movies.',
                'strengths': ['High accuracy', 'Handles sparse data well', 'Discovers latent factors', 'Good for diverse recommendations'],
                'ideal_for': ['Users with varied taste', 'Discovering hidden gems', 'Academic research', 'High-accuracy needs'],
                'complexity': 'High',
                'speed': 'Medium',
                'interpretability': 'Medium',
                'icon': '🔮'
            },
            AlgorithmType.USER_KNN: {
                'name': 'KNN User-Based',
                'description': 'Finds users with similar taste and recommends movies those users loved. Simple and intuitive approach.',
                'strengths': ['Highly interpretable', 'Good for sparse users', 'Handles new items well', 'Community-based recommendations'],
                'ideal_for': ['New users', 'Sparse rating profiles', 'Social recommendations', 'Explainable results'],
                'complexity': 'Low',
                'speed': 'Fast',
                'interpretability': 'Very High',
                'icon': '👥'
            },
            AlgorithmType.ITEM_KNN: {
                'name': 'KNN Item-Based',
                'description': 'Analyzes movies with similar rating patterns and recommends items similar to what you enjoyed.',
                'strengths': ['Stable recommendations', 'Good for frequent users', 'Pre-computed similarities', 'Genre-aware'],
                'ideal_for': ['Users with many ratings', 'Discovering similar movies', 'Stable preferences', 'Genre exploration'],
                'complexity': 'Medium',
                'speed': 'Fast',
                'interpretability': 'High',
                'icon': '🎬'
            },
            AlgorithmType.CONTENT_BASED: {
                'name': 'Content-Based Filtering',
                'description': 'Analyzes movie features (genres, tags, titles) and recommends movies similar to what you enjoyed. Perfect for cold-start scenarios.',
                'strengths': ['No cold-start problem', 'Feature-based recommendations', 'Highly interpretable', 'Tag and genre aware', 'Works for new users'],
                'ideal_for': ['New users', 'Genre-specific discovery', 'Feature-based exploration', 'Cold-start scenarios', 'Explainable recommendations'],
                'complexity': 'Medium',
                'speed': 'Fast',
                'interpretability': 'Very High',
                'icon': '🔍'
            },
            AlgorithmType.HYBRID: {
                'name': 'Hybrid (Best of All)',
                'description': 'Intelligently combines all algorithms with dynamic weighting based on your profile and context.',
                'strengths': ['Best overall accuracy', 'Adapts to user type', 'Robust performance', 'Combines multiple paradigms'],
                'ideal_for': ['Production systems', 'Best accuracy', 'All user types', 'Research comparison'],
                'complexity': 'Very High',
                'speed': 'Medium',
                'interpretability': 'Medium',
                'icon': '🚀'
            }
        }
        
        return info_map.get(algorithm_type, {})
    
    def get_performance_comparison(self) -> pd.DataFrame:
        """
        Get performance comparison of all trained algorithms.
        
        Returns:
            DataFrame with performance metrics
        """
        performance_data = []
        
        for algorithm_type, algorithm in self._algorithms.items():
            if algorithm.is_trained:
                info = self.get_algorithm_info(algorithm_type)
                performance_data.append({
                    'Algorithm': algorithm.name,
                    'RMSE': f"{algorithm.metrics.rmse:.4f}",
                    'Training Time': f"{algorithm.metrics.training_time:.1f}s",
                    'Coverage': f"{algorithm.metrics.coverage:.1f}%",
                    'Memory Usage': f"{algorithm.metrics.memory_usage_mb:.1f} MB",
                    'Prediction Speed': f"{algorithm.metrics.prediction_time:.4f}s",
                    'Icon': info.get('icon', '🎯'),
                    'Interpretability': info.get('interpretability', 'Medium')
                })
        
        return pd.DataFrame(performance_data)
    
    def get_cached_algorithms(self) -> List[AlgorithmType]:
        """Get list of currently cached (trained) algorithms"""
        return [algo_type for algo_type, algo in self._algorithms.items() if algo.is_trained]
    
    def clear_cache(self, algorithm_type: Optional[AlgorithmType] = None) -> None:
        """
        Clear algorithm cache (useful for memory management).
        
        Args:
            algorithm_type: Specific algorithm to clear, or None to clear all
        """
        with self._lock:
            if algorithm_type is None:
                # Clear all algorithms
                self._algorithms.clear()
                print("🗑️ Cleared all algorithm cache")
            elif algorithm_type in self._algorithms:
                # Clear specific algorithm
                del self._algorithms[algorithm_type]
                print(f"🗑️ Cleared {algorithm_type.value} from cache")
    
    def preload_algorithm(self, algorithm_type: AlgorithmType, 
                         custom_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Preload an algorithm in the background for faster switching.
        
        Args:
            algorithm_type: Algorithm to preload
            custom_params: Optional custom parameters
        """
        if algorithm_type not in self._algorithms:
            print(f"🔄 Preloading {algorithm_type.value} in background...")
            # This will cache the algorithm for future use
            self.get_algorithm(algorithm_type, custom_params)
    
    def get_recommendation_explanation(self, algorithm_type: AlgorithmType,
                                    user_id: int, movie_id: int) -> str:
        """
        Get human-readable explanation for why a movie was recommended.
        
        Args:
            algorithm_type: Algorithm that made the recommendation
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Human-readable explanation string
        """
        if algorithm_type not in self._algorithms:
            return "Algorithm not loaded."
        
        algorithm = self._algorithms[algorithm_type]
        context = algorithm.get_explanation_context(user_id, movie_id)
        
        if not context:
            return "Unable to generate explanation."
        
        # Generate explanation based on algorithm type and context
        if algorithm_type == AlgorithmType.SVD:
            return self._explain_svd(context)
        elif algorithm_type == AlgorithmType.USER_KNN:
            return self._explain_user_knn(context)
        elif algorithm_type == AlgorithmType.ITEM_KNN:
            return self._explain_item_knn(context)
        elif algorithm_type == AlgorithmType.HYBRID:
            return self._explain_hybrid(context)
        
        return "Explanation not available."
    
    def _explain_svd(self, context: Dict[str, Any]) -> str:
        """Generate SVD-specific explanation"""
        pred = context.get('prediction', 0)
        return (f"SVD predicts you'll rate this movie **{pred:.1f}/5.0** based on "
               f"latent patterns discovered in your rating history and similar users' preferences.")
    
    def _explain_user_knn(self, context: Dict[str, Any]) -> str:
        """Generate User KNN-specific explanation"""
        similar_users = context.get('similar_users_count', 0)
        pred = context.get('prediction', 0)
        
        if similar_users > 0:
            return (f"**{similar_users} users** with similar taste loved this movie! "
                   f"Predicted rating: **{pred:.1f}/5.0**")
        else:
            return f"Based on users with similar preferences. Predicted rating: **{pred:.1f}/5.0**"
    
    def _explain_item_knn(self, context: Dict[str, Any]) -> str:
        """Generate Item KNN-specific explanation"""
        similar_movies = context.get('similar_movies_count', 0)
        pred = context.get('prediction', 0)
        
        if similar_movies > 0:
            return (f"Because you enjoyed **{similar_movies} similar movies**. "
                   f"Predicted rating: **{pred:.1f}/5.0**")
        else:
            return f"Based on movies with similar rating patterns. Predicted rating: **{pred:.1f}/5.0**"
    
    def _explain_hybrid(self, context: Dict[str, Any]) -> str:
        """Generate Hybrid-specific explanation"""
        primary = context.get('primary_algorithm', 'multiple algorithms')
        pred = context.get('prediction', 0)
        weights = context.get('algorithm_weights', {})
        
        return (f"Hybrid prediction (**{pred:.1f}/5.0**) combining multiple algorithms. "
               f"Primary method: **{primary}**. Algorithm weights: "
               f"SVD {weights.get('svd', 0):.2f}, User KNN {weights.get('user_knn', 0):.2f}, "
               f"Item KNN {weights.get('item_knn', 0):.2f}")

    def is_algorithm_cached(self, algorithm_type: AlgorithmType) -> bool:
        """
        Check if algorithm is already cached and trained without loading it.
        
        Args:
            algorithm_type: Type of algorithm to check
            
        Returns:
            True if algorithm is cached and trained, False otherwise
        """
        if algorithm_type in self._algorithms:
            return self._algorithms[algorithm_type].is_trained
        return False
    
    def get_algorithm_metrics(self, algorithm_type: AlgorithmType) -> Dict[str, Any]:
        """Get performance metrics for a specific algorithm"""
        # Check if algorithm is cached first
        if not self.is_algorithm_cached(algorithm_type):
            return {
                "algorithm": algorithm_type.value,
                "status": "Not loaded",
                "metrics": {},
                "training_time": "Unknown",
                "sample_size": "N/A"
            }
        
        # Get cached algorithm (won't trigger training)
        algorithm = self._algorithms[algorithm_type]
        
        if not algorithm.is_trained:
            return {
                "algorithm": algorithm_type.value,
                "status": "Not trained",
                "metrics": {},
                "training_time": "Unknown",
                "sample_size": "N/A"
            }
        
        # Get algorithm metrics object
        metrics_obj = algorithm.metrics
        
        # Extract training time and sample size from metrics object
        training_time = getattr(metrics_obj, 'training_time', 0.0)
        
        # Get sample size from ratings_df if available
        sample_size = "N/A"
        if hasattr(algorithm, 'ratings_df') and algorithm.ratings_df is not None:
            sample_size = len(algorithm.ratings_df)
        
        # Use pre-computed metrics from algorithm object (fast!)
        # Don't recompute on-demand as it's too slow (1000 predictions)
        metrics = {
            "rmse": round(metrics_obj.rmse, 3),
            "mae": round(getattr(metrics_obj, 'mae', 0.0), 3),
            "coverage": round(metrics_obj.coverage, 1),
            "sample_size": sample_size
        }
        
        return {
            "algorithm": algorithm_type.value,
            "status": "Trained",
            "training_time": training_time if training_time > 0 else "Unknown",
            "metrics": metrics
        }

    def get_all_algorithm_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all available algorithms"""
        all_metrics = {}
        
        for algorithm_type in self.get_available_algorithms():
            try:
                metrics = self.get_algorithm_metrics(algorithm_type)
                all_metrics[algorithm_type.value] = metrics
            except Exception as e:
                all_metrics[algorithm_type.value] = {
                    "algorithm": algorithm_type.value,
                    "status": "Error",
                    "error": str(e),
                    "metrics": {}
                }
        
        return all_metrics


def get_algorithm_manager(use_external_storage: bool = True) -> AlgorithmManager:
    """
    Get the algorithm manager instance.
    
    Uses Streamlit session state when in Streamlit context,
    otherwise uses class-level singleton.
    
    Args:
        use_external_storage: Whether to enable external storage (Redis/S3)
                             for horizontal scaling. Default True.
    
    Returns:
        AlgorithmManager singleton instance
    """
    return AlgorithmManager.get_instance(use_external_storage)