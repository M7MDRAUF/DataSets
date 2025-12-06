"""
CineMatch V2.1.6 - Mock Services

Mock services for external dependencies in testing.
Task 3.9: Mock services for external dependencies.

Author: CineMatch Development Team
Date: December 5, 2025
"""

import pytest
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import json
import hashlib
import threading
import time
from pathlib import Path


# =============================================================================
# BASE MOCK SERVICE
# =============================================================================

class MockService:
    """Base class for mock services"""
    
    def __init__(self):
        self.call_count = 0
        self.call_history: List[Dict[str, Any]] = []
        self._delay = 0.0
        self._should_fail = False
        self._failure_exception = Exception("Mock service failure")
    
    def reset(self):
        """Reset mock state"""
        self.call_count = 0
        self.call_history.clear()
        self._delay = 0.0
        self._should_fail = False
    
    def set_delay(self, seconds: float):
        """Set artificial delay for responses"""
        self._delay = seconds
    
    def set_should_fail(self, should_fail: bool, exception: Optional[Exception] = None):
        """Configure mock to fail"""
        self._should_fail = should_fail
        if exception:
            self._failure_exception = exception
    
    def _record_call(self, method: str, **kwargs):
        """Record a call to the mock"""
        self.call_count += 1
        self.call_history.append({
            'method': method,
            'timestamp': datetime.now().isoformat(),
            'kwargs': kwargs
        })
    
    def _maybe_delay(self):
        """Apply delay if configured"""
        if self._delay > 0:
            time.sleep(self._delay)
    
    def _maybe_fail(self):
        """Raise exception if configured to fail"""
        if self._should_fail:
            raise self._failure_exception


# =============================================================================
# MOCK DATABASE SERVICE
# =============================================================================

class MockDatabaseService(MockService):
    """Mock database service for testing"""
    
    def __init__(self):
        super().__init__()
        self._data: Dict[str, Any] = {}
        self._tables: Dict[str, pd.DataFrame] = {}
        self._lock = threading.Lock()
    
    def reset(self):
        super().reset()
        self._data.clear()
        self._tables.clear()
    
    def connect(self) -> bool:
        """Mock database connection"""
        self._record_call('connect')
        self._maybe_delay()
        self._maybe_fail()
        return True
    
    def disconnect(self):
        """Mock database disconnection"""
        self._record_call('disconnect')
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Mock query execution"""
        self._record_call('execute_query', query=query, params=params)
        self._maybe_delay()
        self._maybe_fail()
        
        # Simple query parsing for testing
        if query.lower().startswith('select'):
            return self._mock_select(query)
        return []
    
    def _mock_select(self, query: str) -> List[Dict]:
        """Mock SELECT query"""
        # Return empty list or mock data
        return [{'id': 1, 'value': 'mock'}]
    
    def insert(self, table: str, data: Dict[str, Any]) -> int:
        """Mock insert"""
        self._record_call('insert', table=table, data=data)
        self._maybe_delay()
        self._maybe_fail()
        
        with self._lock:
            if table not in self._tables:
                self._tables[table] = pd.DataFrame()
            
            new_id = len(self._tables[table]) + 1
            data['id'] = new_id
            self._tables[table] = pd.concat([
                self._tables[table],
                pd.DataFrame([data])
            ], ignore_index=True)
            
            return new_id
    
    def update(self, table: str, id: int, data: Dict[str, Any]) -> bool:
        """Mock update"""
        self._record_call('update', table=table, id=id, data=data)
        self._maybe_delay()
        self._maybe_fail()
        
        with self._lock:
            if table in self._tables:
                mask = self._tables[table]['id'] == id
                if mask.any():
                    for key, value in data.items():
                        self._tables[table].loc[mask, key] = value
                    return True
            return False
    
    def delete(self, table: str, id: int) -> bool:
        """Mock delete"""
        self._record_call('delete', table=table, id=id)
        self._maybe_delay()
        self._maybe_fail()
        
        with self._lock:
            if table in self._tables:
                mask = self._tables[table]['id'] != id
                self._tables[table] = self._tables[table][mask]
                return True
            return False
    
    def get_table(self, table: str) -> pd.DataFrame:
        """Get mock table data"""
        return self._tables.get(table, pd.DataFrame())


# =============================================================================
# MOCK CACHE SERVICE
# =============================================================================

class MockCacheService(MockService):
    """Mock cache service for testing"""
    
    def __init__(self, max_size: int = 1000):
        super().__init__()
        self._cache: Dict[str, Any] = {}
        self._expiry: Dict[str, float] = {}
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
    
    def reset(self):
        super().reset()
        self._cache.clear()
        self._expiry.clear()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        self._record_call('get', key=key)
        self._maybe_delay()
        self._maybe_fail()
        
        if key in self._cache:
            # Check expiry
            if key in self._expiry and time.time() > self._expiry[key]:
                del self._cache[key]
                del self._expiry[key]
                self._misses += 1
                return None
            
            self._hits += 1
            return self._cache[key]
        
        self._misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        self._record_call('set', key=key, ttl=ttl)
        self._maybe_delay()
        self._maybe_fail()
        
        if len(self._cache) >= self._max_size:
            # Evict oldest
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            if oldest_key in self._expiry:
                del self._expiry[oldest_key]
        
        self._cache[key] = value
        if ttl:
            self._expiry[key] = time.time() + ttl
        
        return True
    
    def delete(self, key: str) -> bool:
        """Delete from cache"""
        self._record_call('delete', key=key)
        self._maybe_delay()
        self._maybe_fail()
        
        if key in self._cache:
            del self._cache[key]
            if key in self._expiry:
                del self._expiry[key]
            return True
        return False
    
    def clear(self):
        """Clear cache"""
        self._record_call('clear')
        self._cache.clear()
        self._expiry.clear()
    
    @property
    def hit_rate(self) -> float:
        """Get cache hit rate"""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    @property
    def size(self) -> int:
        """Get current cache size"""
        return len(self._cache)


# =============================================================================
# MOCK FILE STORAGE SERVICE
# =============================================================================

class MockFileStorageService(MockService):
    """Mock file storage service for testing"""
    
    def __init__(self):
        super().__init__()
        self._files: Dict[str, bytes] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    def reset(self):
        super().reset()
        self._files.clear()
        self._metadata.clear()
    
    def upload(self, path: str, content: bytes, metadata: Optional[Dict] = None) -> bool:
        """Upload file"""
        self._record_call('upload', path=path, size=len(content))
        self._maybe_delay()
        self._maybe_fail()
        
        self._files[path] = content
        self._metadata[path] = metadata or {}
        self._metadata[path]['uploaded_at'] = datetime.now().isoformat()
        self._metadata[path]['size'] = len(content)
        
        return True
    
    def download(self, path: str) -> Optional[bytes]:
        """Download file"""
        self._record_call('download', path=path)
        self._maybe_delay()
        self._maybe_fail()
        
        return self._files.get(path)
    
    def delete(self, path: str) -> bool:
        """Delete file"""
        self._record_call('delete', path=path)
        self._maybe_delay()
        self._maybe_fail()
        
        if path in self._files:
            del self._files[path]
            if path in self._metadata:
                del self._metadata[path]
            return True
        return False
    
    def exists(self, path: str) -> bool:
        """Check if file exists"""
        self._record_call('exists', path=path)
        return path in self._files
    
    def list_files(self, prefix: str = "") -> List[str]:
        """List files with prefix"""
        self._record_call('list_files', prefix=prefix)
        return [p for p in self._files.keys() if p.startswith(prefix)]
    
    def get_metadata(self, path: str) -> Optional[Dict]:
        """Get file metadata"""
        return self._metadata.get(path)


# =============================================================================
# MOCK MODEL SERVICE
# =============================================================================

class MockModelService(MockService):
    """Mock ML model service for testing"""
    
    def __init__(self):
        super().__init__()
        self._models: Dict[str, Any] = {}
        self._is_loaded: Dict[str, bool] = {}
    
    def reset(self):
        super().reset()
        self._models.clear()
        self._is_loaded.clear()
    
    def load_model(self, model_id: str, path: Optional[str] = None) -> bool:
        """Load model"""
        self._record_call('load_model', model_id=model_id, path=path)
        self._maybe_delay()
        self._maybe_fail()
        
        self._models[model_id] = {'id': model_id, 'path': path}
        self._is_loaded[model_id] = True
        
        return True
    
    def unload_model(self, model_id: str) -> bool:
        """Unload model"""
        self._record_call('unload_model', model_id=model_id)
        
        if model_id in self._models:
            del self._models[model_id]
            self._is_loaded[model_id] = False
            return True
        return False
    
    def predict(self, model_id: str, data: Any) -> Any:
        """Make prediction"""
        self._record_call('predict', model_id=model_id)
        self._maybe_delay()
        self._maybe_fail()
        
        if not self._is_loaded.get(model_id, False):
            raise RuntimeError(f"Model {model_id} not loaded")
        
        # Return mock predictions
        if isinstance(data, pd.DataFrame):
            return np.random.uniform(1, 5, len(data)).round(1)
        return 3.5
    
    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """Get model info"""
        return self._models.get(model_id)
    
    def is_loaded(self, model_id: str) -> bool:
        """Check if model is loaded"""
        return self._is_loaded.get(model_id, False)


# =============================================================================
# MOCK HTTP CLIENT
# =============================================================================

class MockHTTPClient(MockService):
    """Mock HTTP client for testing external API calls"""
    
    def __init__(self):
        super().__init__()
        self._responses: Dict[str, Dict] = {}
        self._default_response = {'status': 200, 'body': {}}
    
    def reset(self):
        super().reset()
        self._responses.clear()
    
    def register_response(self, url: str, method: str, response: Dict):
        """Register mock response for URL"""
        key = f"{method.upper()}:{url}"
        self._responses[key] = response
    
    def get(self, url: str, **kwargs) -> Dict:
        """Mock GET request"""
        return self._request('GET', url, **kwargs)
    
    def post(self, url: str, data: Any = None, **kwargs) -> Dict:
        """Mock POST request"""
        return self._request('POST', url, data=data, **kwargs)
    
    def put(self, url: str, data: Any = None, **kwargs) -> Dict:
        """Mock PUT request"""
        return self._request('PUT', url, data=data, **kwargs)
    
    def delete(self, url: str, **kwargs) -> Dict:
        """Mock DELETE request"""
        return self._request('DELETE', url, **kwargs)
    
    def _request(self, method: str, url: str, **kwargs) -> Dict:
        """Make mock request"""
        self._record_call(method.lower(), url=url, **kwargs)
        self._maybe_delay()
        self._maybe_fail()
        
        key = f"{method.upper()}:{url}"
        return self._responses.get(key, self._default_response)


# =============================================================================
# MOCK MESSAGE QUEUE
# =============================================================================

class MockMessageQueue(MockService):
    """Mock message queue for testing"""
    
    def __init__(self):
        super().__init__()
        self._queues: Dict[str, List[Any]] = {}
        self._subscribers: Dict[str, List[Callable]] = {}
    
    def reset(self):
        super().reset()
        self._queues.clear()
        self._subscribers.clear()
    
    def publish(self, queue: str, message: Any) -> bool:
        """Publish message to queue"""
        self._record_call('publish', queue=queue)
        self._maybe_delay()
        self._maybe_fail()
        
        if queue not in self._queues:
            self._queues[queue] = []
        
        self._queues[queue].append(message)
        
        # Notify subscribers
        for callback in self._subscribers.get(queue, []):
            callback(message)
        
        return True
    
    def subscribe(self, queue: str, callback: Callable):
        """Subscribe to queue"""
        self._record_call('subscribe', queue=queue)
        
        if queue not in self._subscribers:
            self._subscribers[queue] = []
        
        self._subscribers[queue].append(callback)
    
    def consume(self, queue: str, timeout: float = 1.0) -> Optional[Any]:
        """Consume message from queue"""
        self._record_call('consume', queue=queue)
        self._maybe_delay()
        self._maybe_fail()
        
        if queue in self._queues and self._queues[queue]:
            return self._queues[queue].pop(0)
        return None
    
    def queue_size(self, queue: str) -> int:
        """Get queue size"""
        return len(self._queues.get(queue, []))


# =============================================================================
# PYTEST FIXTURES
# =============================================================================

@pytest.fixture
def mock_database():
    """Provide mock database service"""
    db = MockDatabaseService()
    yield db
    db.reset()


@pytest.fixture
def mock_cache():
    """Provide mock cache service"""
    cache = MockCacheService()
    yield cache
    cache.reset()


@pytest.fixture
def mock_file_storage():
    """Provide mock file storage service"""
    storage = MockFileStorageService()
    yield storage
    storage.reset()


@pytest.fixture
def mock_model_service():
    """Provide mock model service"""
    service = MockModelService()
    yield service
    service.reset()


@pytest.fixture
def mock_http_client():
    """Provide mock HTTP client"""
    client = MockHTTPClient()
    yield client
    client.reset()


@pytest.fixture
def mock_message_queue():
    """Provide mock message queue"""
    mq = MockMessageQueue()
    yield mq
    mq.reset()


# =============================================================================
# MOCK SERVICE TESTS
# =============================================================================

class TestMockDatabase:
    """Test mock database service"""
    
    def test_connect(self, mock_database):
        """Test database connection"""
        result = mock_database.connect()
        assert result is True
        assert mock_database.call_count == 1
    
    def test_insert_and_query(self, mock_database):
        """Test insert and query"""
        mock_database.insert('users', {'name': 'Test', 'email': 'test@test.com'})
        
        table = mock_database.get_table('users')
        assert len(table) == 1
        assert table.iloc[0]['name'] == 'Test'


class TestMockCache:
    """Test mock cache service"""
    
    def test_set_and_get(self, mock_cache):
        """Test set and get"""
        mock_cache.set('key1', 'value1')
        result = mock_cache.get('key1')
        assert result == 'value1'
    
    def test_cache_miss(self, mock_cache):
        """Test cache miss"""
        result = mock_cache.get('nonexistent')
        assert result is None
    
    def test_hit_rate(self, mock_cache):
        """Test hit rate calculation"""
        mock_cache.set('key1', 'value1')
        mock_cache.get('key1')  # Hit
        mock_cache.get('key2')  # Miss
        
        assert mock_cache.hit_rate == 0.5


class TestMockModelService:
    """Test mock model service"""
    
    def test_load_and_predict(self, mock_model_service):
        """Test model loading and prediction"""
        mock_model_service.load_model('svd', 'models/svd.pkl')
        
        assert mock_model_service.is_loaded('svd')
        
        result = mock_model_service.predict('svd', pd.DataFrame({'x': [1, 2, 3]}))
        assert len(result) == 3
    
    def test_predict_unloaded(self, mock_model_service):
        """Test prediction with unloaded model"""
        with pytest.raises(RuntimeError):
            mock_model_service.predict('unloaded', {})


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
