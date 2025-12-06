"""
CineMatch V2.1.6 - Data Source Adapters

Extensible framework for connecting to various data sources
beyond MovieLens format (databases, APIs, cloud storage, etc.).

Author: CineMatch Development Team
"""

from abc import ABC, abstractmethod
from typing import (
    Dict, Any, Optional, List, Iterator, Tuple, Type, Union, AsyncIterator
)
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
import asyncio
from datetime import datetime
import json
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


# =============================================================================
# Data Source Types
# =============================================================================

class DataSourceType(Enum):
    """Types of data sources"""
    FILE = "file"               # Local file system
    DATABASE = "database"       # SQL/NoSQL databases
    API = "api"                 # REST/GraphQL APIs
    STREAMING = "streaming"     # Kafka, Pub/Sub, etc.
    CLOUD_STORAGE = "cloud"     # S3, GCS, Azure Blob
    MEMORY = "memory"           # In-memory data
    CUSTOM = "custom"           # User-defined


class DataFormat(Enum):
    """Supported data formats"""
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    AVRO = "avro"
    MOVIELENS = "movielens"
    NETFLIX = "netflix"
    AMAZON = "amazon"
    CUSTOM = "custom"


class ConnectionStatus(Enum):
    """Data source connection status"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    CONNECTING = "connecting"


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class Rating:
    """Standard rating record"""
    user_id: int
    item_id: int
    rating: float
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Item:
    """Standard item record"""
    item_id: int
    title: str
    genres: List[str] = field(default_factory=list)
    year: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class User:
    """Standard user record"""
    user_id: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataSourceConfig:
    """Configuration for a data source"""
    name: str
    source_type: DataSourceType
    connection_string: str = ""
    format: DataFormat = DataFormat.CSV
    credentials: Dict[str, str] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)
    batch_size: int = 10000
    timeout_seconds: int = 30


@dataclass
class DataSourceStats:
    """Statistics about a data source"""
    total_ratings: int = 0
    total_users: int = 0
    total_items: int = 0
    rating_range: Tuple[float, float] = (0.0, 5.0)
    date_range: Optional[Tuple[datetime, datetime]] = None
    last_updated: Optional[datetime] = None


# =============================================================================
# Base Data Adapter
# =============================================================================

class BaseDataAdapter(ABC):
    """
    Abstract base class for data source adapters.
    
    All data adapters must implement these methods to ensure
    consistent data access patterns across different sources.
    """
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.name = config.name
        self._status = ConnectionStatus.DISCONNECTED
        self._stats: Optional[DataSourceStats] = None
        self._cache: Dict[str, Any] = {}
        
    @property
    def status(self) -> ConnectionStatus:
        return self._status
    
    @property
    def stats(self) -> Optional[DataSourceStats]:
        return self._stats
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to data source"""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Close connection to data source"""
        pass
    
    @abstractmethod
    def load_ratings(
        self,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> pd.DataFrame:
        """Load ratings data"""
        pass
    
    @abstractmethod
    def load_items(self) -> pd.DataFrame:
        """Load item metadata"""
        pass
    
    @abstractmethod
    def load_users(self) -> pd.DataFrame:
        """Load user data"""
        pass
    
    def iter_ratings(
        self,
        batch_size: Optional[int] = None
    ) -> Iterator[pd.DataFrame]:
        """Iterate over ratings in batches"""
        batch_size = batch_size or self.config.batch_size
        offset = 0
        
        while True:
            batch = self.load_ratings(limit=batch_size, offset=offset)
            if batch.empty:
                break
            yield batch
            offset += len(batch)
    
    def validate_schema(self, df: pd.DataFrame, expected_columns: List[str]) -> bool:
        """Validate DataFrame has expected columns"""
        missing = set(expected_columns) - set(df.columns)
        if missing:
            logger.warning(f"Missing columns: {missing}")
            return False
        return True
    
    def refresh_stats(self):
        """Refresh data source statistics"""
        try:
            ratings = self.load_ratings(limit=1000000)
            items = self.load_items()
            users = self.load_users()
            
            self._stats = DataSourceStats(
                total_ratings=len(ratings),
                total_users=ratings['userId'].nunique() if 'userId' in ratings else len(users),
                total_items=ratings['movieId'].nunique() if 'movieId' in ratings else len(items),
                rating_range=(
                    float(ratings['rating'].min()) if 'rating' in ratings else 0.0,
                    float(ratings['rating'].max()) if 'rating' in ratings else 5.0
                ),
                last_updated=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error refreshing stats: {e}")
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


# =============================================================================
# File-based Adapters
# =============================================================================

class MovieLensAdapter(BaseDataAdapter):
    """Adapter for MovieLens dataset format"""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        name: str = "movielens"
    ):
        config = DataSourceConfig(
            name=name,
            source_type=DataSourceType.FILE,
            connection_string=str(data_dir),
            format=DataFormat.MOVIELENS
        )
        super().__init__(config)
        self.data_dir = Path(data_dir)
    
    def connect(self) -> bool:
        if not self.data_dir.exists():
            logger.error(f"Data directory not found: {self.data_dir}")
            self._status = ConnectionStatus.ERROR
            return False
        
        self._status = ConnectionStatus.CONNECTED
        return True
    
    def disconnect(self) -> bool:
        self._status = ConnectionStatus.DISCONNECTED
        return True
    
    def load_ratings(
        self,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> pd.DataFrame:
        ratings_file = self.data_dir / "ratings.csv"
        if not ratings_file.exists():
            logger.error(f"Ratings file not found: {ratings_file}")
            return pd.DataFrame()
        
        try:
            if limit is not None:
                df = pd.read_csv(
                    ratings_file,
                    skiprows=range(1, offset + 1) if offset > 0 else None,
                    nrows=limit
                )
            else:
                df = pd.read_csv(ratings_file)
                if offset > 0:
                    df = df.iloc[offset:]
            
            return df
        except Exception as e:
            logger.error(f"Error loading ratings: {e}")
            return pd.DataFrame()
    
    def load_items(self) -> pd.DataFrame:
        movies_file = self.data_dir / "movies.csv"
        if not movies_file.exists():
            logger.error(f"Movies file not found: {movies_file}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(movies_file)
            # Parse genres
            if 'genres' in df.columns:
                df['genres'] = df['genres'].str.split('|')
            return df
        except Exception as e:
            logger.error(f"Error loading movies: {e}")
            return pd.DataFrame()
    
    def load_users(self) -> pd.DataFrame:
        # MovieLens doesn't have separate user file
        # Generate from ratings
        ratings = self.load_ratings()
        if ratings.empty:
            return pd.DataFrame()
        
        users = pd.DataFrame({'userId': ratings['userId'].unique()})
        return users
    
    def load_tags(self) -> pd.DataFrame:
        """Load MovieLens tags"""
        tags_file = self.data_dir / "tags.csv"
        if not tags_file.exists():
            return pd.DataFrame()
        
        try:
            return pd.read_csv(tags_file)
        except Exception as e:
            logger.error(f"Error loading tags: {e}")
            return pd.DataFrame()
    
    def load_links(self) -> pd.DataFrame:
        """Load external IDs (IMDB, TMDB)"""
        links_file = self.data_dir / "links.csv"
        if not links_file.exists():
            return pd.DataFrame()
        
        try:
            return pd.read_csv(links_file)
        except Exception as e:
            logger.error(f"Error loading links: {e}")
            return pd.DataFrame()


class CSVAdapter(BaseDataAdapter):
    """Generic CSV file adapter"""
    
    def __init__(
        self,
        ratings_file: Union[str, Path],
        items_file: Optional[Union[str, Path]] = None,
        users_file: Optional[Union[str, Path]] = None,
        column_mapping: Optional[Dict[str, str]] = None,
        name: str = "csv"
    ):
        config = DataSourceConfig(
            name=name,
            source_type=DataSourceType.FILE,
            connection_string=str(ratings_file),
            format=DataFormat.CSV
        )
        super().__init__(config)
        
        self.ratings_file = Path(ratings_file)
        self.items_file = Path(items_file) if items_file else None
        self.users_file = Path(users_file) if users_file else None
        self.column_mapping = column_mapping or {}
    
    def connect(self) -> bool:
        if not self.ratings_file.exists():
            self._status = ConnectionStatus.ERROR
            return False
        self._status = ConnectionStatus.CONNECTED
        return True
    
    def disconnect(self) -> bool:
        self._status = ConnectionStatus.DISCONNECTED
        return True
    
    def _apply_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply column name mapping"""
        if self.column_mapping:
            df = df.rename(columns=self.column_mapping)
        return df
    
    def load_ratings(
        self,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.ratings_file, nrows=limit)
            if offset > 0:
                df = df.iloc[offset:]
            return self._apply_mapping(df)
        except Exception as e:
            logger.error(f"Error loading ratings: {e}")
            return pd.DataFrame()
    
    def load_items(self) -> pd.DataFrame:
        if self.items_file is None or not self.items_file.exists():
            return pd.DataFrame()
        try:
            return self._apply_mapping(pd.read_csv(self.items_file))
        except Exception as e:
            logger.error(f"Error loading items: {e}")
            return pd.DataFrame()
    
    def load_users(self) -> pd.DataFrame:
        if self.users_file is None or not self.users_file.exists():
            ratings = self.load_ratings()
            if 'userId' in ratings.columns:
                return pd.DataFrame({'userId': ratings['userId'].unique()})
            return pd.DataFrame()
        try:
            return self._apply_mapping(pd.read_csv(self.users_file))
        except Exception as e:
            logger.error(f"Error loading users: {e}")
            return pd.DataFrame()


class JSONAdapter(BaseDataAdapter):
    """JSON file adapter"""
    
    def __init__(
        self,
        data_file: Union[str, Path],
        ratings_key: str = "ratings",
        items_key: str = "items",
        users_key: str = "users",
        name: str = "json"
    ):
        config = DataSourceConfig(
            name=name,
            source_type=DataSourceType.FILE,
            connection_string=str(data_file),
            format=DataFormat.JSON
        )
        super().__init__(config)
        
        self.data_file = Path(data_file)
        self.ratings_key = ratings_key
        self.items_key = items_key
        self.users_key = users_key
        self._data: Optional[Dict] = None
    
    def connect(self) -> bool:
        if not self.data_file.exists():
            self._status = ConnectionStatus.ERROR
            return False
        
        try:
            with open(self.data_file, 'r') as f:
                self._data = json.load(f)
            self._status = ConnectionStatus.CONNECTED
            return True
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            self._status = ConnectionStatus.ERROR
            return False
    
    def disconnect(self) -> bool:
        self._data = None
        self._status = ConnectionStatus.DISCONNECTED
        return True
    
    def load_ratings(
        self,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> pd.DataFrame:
        if self._data is None:
            return pd.DataFrame()
        
        data = self._data.get(self.ratings_key, [])
        if offset > 0:
            data = data[offset:]
        if limit:
            data = data[:limit]
        return pd.DataFrame(data)
    
    def load_items(self) -> pd.DataFrame:
        if self._data is None:
            return pd.DataFrame()
        return pd.DataFrame(self._data.get(self.items_key, []))
    
    def load_users(self) -> pd.DataFrame:
        if self._data is None:
            return pd.DataFrame()
        return pd.DataFrame(self._data.get(self.users_key, []))


# =============================================================================
# Database Adapters
# =============================================================================

class SQLAdapter(BaseDataAdapter):
    """SQL database adapter (SQLite, PostgreSQL, MySQL)"""
    
    def __init__(
        self,
        connection_string: str,
        ratings_table: str = "ratings",
        items_table: str = "items",
        users_table: str = "users",
        name: str = "sql"
    ):
        config = DataSourceConfig(
            name=name,
            source_type=DataSourceType.DATABASE,
            connection_string=connection_string
        )
        super().__init__(config)
        
        self.ratings_table = ratings_table
        self.items_table = items_table
        self.users_table = users_table
        self._connection = None
    
    def connect(self) -> bool:
        try:
            import sqlite3
            from urllib.parse import urlparse
            
            parsed = urlparse(self.config.connection_string)
            
            if parsed.scheme in ('sqlite', '') or self.config.connection_string.endswith('.db'):
                # SQLite
                db_path = parsed.path if parsed.scheme else self.config.connection_string
                self._connection = sqlite3.connect(db_path)
            else:
                # For other databases, use sqlalchemy
                try:
                    from sqlalchemy import create_engine
                    self._engine = create_engine(self.config.connection_string)
                    self._connection = self._engine.connect()
                except ImportError:
                    logger.error("SQLAlchemy required for non-SQLite databases")
                    self._status = ConnectionStatus.ERROR
                    return False
            
            self._status = ConnectionStatus.CONNECTED
            return True
            
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            self._status = ConnectionStatus.ERROR
            return False
    
    def disconnect(self) -> bool:
        if self._connection:
            self._connection.close()
            self._connection = None
        self._status = ConnectionStatus.DISCONNECTED
        return True
    
    def load_ratings(
        self,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> pd.DataFrame:
        query = f"SELECT * FROM {self.ratings_table}"
        if limit:
            query += f" LIMIT {limit}"
        if offset:
            query += f" OFFSET {offset}"
        
        try:
            return pd.read_sql(query, self._connection)
        except Exception as e:
            logger.error(f"Error loading ratings from SQL: {e}")
            return pd.DataFrame()
    
    def load_items(self) -> pd.DataFrame:
        try:
            return pd.read_sql(f"SELECT * FROM {self.items_table}", self._connection)
        except Exception as e:
            logger.error(f"Error loading items from SQL: {e}")
            return pd.DataFrame()
    
    def load_users(self) -> pd.DataFrame:
        try:
            return pd.read_sql(f"SELECT * FROM {self.users_table}", self._connection)
        except Exception as e:
            logger.error(f"Error loading users from SQL: {e}")
            return pd.DataFrame()


# =============================================================================
# API Adapters
# =============================================================================

class RESTAPIAdapter(BaseDataAdapter):
    """REST API adapter for external recommendation services"""
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        name: str = "rest_api"
    ):
        config = DataSourceConfig(
            name=name,
            source_type=DataSourceType.API,
            connection_string=base_url,
            credentials={'api_key': api_key} if api_key else {}
        )
        super().__init__(config)
        
        self.base_url = base_url.rstrip('/')
        self.headers = headers or {}
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
        
        self._session = None
    
    def connect(self) -> bool:
        try:
            import requests
            self._session = requests.Session()
            self._session.headers.update(self.headers)
            
            # Test connection
            response = self._session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                self._status = ConnectionStatus.CONNECTED
                return True
            else:
                logger.warning(f"API health check failed: {response.status_code}")
                self._status = ConnectionStatus.CONNECTED  # May still work
                return True
                
        except ImportError:
            logger.error("requests library required for REST API adapter")
            self._status = ConnectionStatus.ERROR
            return False
        except Exception as e:
            logger.warning(f"API connection warning: {e}")
            self._status = ConnectionStatus.CONNECTED
            return True
    
    def disconnect(self) -> bool:
        if self._session:
            self._session.close()
            self._session = None
        self._status = ConnectionStatus.DISCONNECTED
        return True
    
    def _fetch(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make API request"""
        if not self._session:
            return {}
        
        try:
            response = self._session.get(
                f"{self.base_url}/{endpoint}",
                params=params,
                timeout=self.config.timeout_seconds
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API request error: {e}")
            return {}
    
    def load_ratings(
        self,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> pd.DataFrame:
        params = {}
        if limit:
            params['limit'] = limit
        if offset:
            params['offset'] = offset
        
        data = self._fetch('ratings', params)
        return pd.DataFrame(data.get('ratings', data.get('data', [])))
    
    def load_items(self) -> pd.DataFrame:
        data = self._fetch('items')
        return pd.DataFrame(data.get('items', data.get('data', [])))
    
    def load_users(self) -> pd.DataFrame:
        data = self._fetch('users')
        return pd.DataFrame(data.get('users', data.get('data', [])))


# =============================================================================
# Cloud Storage Adapters
# =============================================================================

class S3Adapter(BaseDataAdapter):
    """Amazon S3 adapter"""
    
    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        region: str = "us-east-1",
        name: str = "s3"
    ):
        config = DataSourceConfig(
            name=name,
            source_type=DataSourceType.CLOUD_STORAGE,
            connection_string=f"s3://{bucket}/{prefix}",
            credentials={
                'aws_access_key': aws_access_key or '',
                'aws_secret_key': aws_secret_key or ''
            }
        )
        super().__init__(config)
        
        self.bucket = bucket
        self.prefix = prefix
        self.region = region
        self._client = None
    
    def connect(self) -> bool:
        try:
            import boto3
            
            credentials = self.config.credentials
            if credentials.get('aws_access_key') and credentials.get('aws_secret_key'):
                self._client = boto3.client(
                    's3',
                    aws_access_key_id=credentials['aws_access_key'],
                    aws_secret_access_key=credentials['aws_secret_key'],
                    region_name=self.region
                )
            else:
                self._client = boto3.client('s3', region_name=self.region)
            
            self._status = ConnectionStatus.CONNECTED
            return True
            
        except ImportError:
            logger.error("boto3 required for S3 adapter")
            self._status = ConnectionStatus.ERROR
            return False
        except Exception as e:
            logger.error(f"S3 connection error: {e}")
            self._status = ConnectionStatus.ERROR
            return False
    
    def disconnect(self) -> bool:
        self._client = None
        self._status = ConnectionStatus.DISCONNECTED
        return True
    
    def _download_file(self, key: str) -> Optional[bytes]:
        """Download file from S3"""
        if not self._client:
            return None
        
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=key)
            return response['Body'].read()
        except Exception as e:
            logger.error(f"S3 download error: {e}")
            return None
    
    def load_ratings(
        self,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> pd.DataFrame:
        key = f"{self.prefix}ratings.csv" if self.prefix else "ratings.csv"
        data = self._download_file(key)
        
        if data is None:
            return pd.DataFrame()
        
        import io
        df = pd.read_csv(io.BytesIO(data))
        
        if offset > 0:
            df = df.iloc[offset:]
        if limit:
            df = df.head(limit)
        
        return df
    
    def load_items(self) -> pd.DataFrame:
        key = f"{self.prefix}movies.csv" if self.prefix else "movies.csv"
        data = self._download_file(key)
        
        if data is None:
            return pd.DataFrame()
        
        import io
        return pd.read_csv(io.BytesIO(data))
    
    def load_users(self) -> pd.DataFrame:
        ratings = self.load_ratings()
        if 'userId' in ratings.columns:
            return pd.DataFrame({'userId': ratings['userId'].unique()})
        return pd.DataFrame()


# =============================================================================
# Adapter Registry
# =============================================================================

class DataAdapterRegistry:
    """Registry for data source adapters"""
    
    _instance: Optional['DataAdapterRegistry'] = None
    
    def __new__(cls) -> 'DataAdapterRegistry':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._adapters: Dict[str, Type[BaseDataAdapter]] = {}
            cls._instance._instances: Dict[str, BaseDataAdapter] = {}
            cls._instance._register_builtins()
        return cls._instance
    
    def _register_builtins(self):
        """Register built-in adapters"""
        self.register_type("movielens", MovieLensAdapter)
        self.register_type("csv", CSVAdapter)
        self.register_type("json", JSONAdapter)
        self.register_type("sql", SQLAdapter)
        self.register_type("rest_api", RESTAPIAdapter)
        self.register_type("s3", S3Adapter)
    
    def register_type(self, name: str, adapter_class: Type[BaseDataAdapter]):
        """Register an adapter type"""
        self._adapters[name.lower()] = adapter_class
        logger.info(f"Registered data adapter: {name}")
    
    def create(self, adapter_type: str, **kwargs) -> Optional[BaseDataAdapter]:
        """Create a new adapter instance"""
        adapter_class = self._adapters.get(adapter_type.lower())
        if adapter_class is None:
            logger.error(f"Unknown adapter type: {adapter_type}")
            return None
        
        try:
            adapter = adapter_class(**kwargs)
            return adapter
        except Exception as e:
            logger.error(f"Error creating adapter: {e}")
            return None
    
    def register_instance(self, name: str, adapter: BaseDataAdapter):
        """Register an adapter instance"""
        self._instances[name] = adapter
    
    def get_instance(self, name: str) -> Optional[BaseDataAdapter]:
        """Get a registered adapter instance"""
        return self._instances.get(name)
    
    def list_types(self) -> List[str]:
        """List available adapter types"""
        return list(self._adapters.keys())
    
    def list_instances(self) -> List[str]:
        """List registered adapter instances"""
        return list(self._instances.keys())


# =============================================================================
# Data Transformer
# =============================================================================

class DataTransformer:
    """Transform data between different formats"""
    
    @staticmethod
    def to_standard_format(
        ratings: pd.DataFrame,
        user_col: str = 'userId',
        item_col: str = 'movieId',
        rating_col: str = 'rating',
        timestamp_col: Optional[str] = 'timestamp'
    ) -> pd.DataFrame:
        """Convert to standard CineMatch format"""
        result = ratings.rename(columns={
            user_col: 'userId',
            item_col: 'movieId',
            rating_col: 'rating'
        })
        
        if timestamp_col and timestamp_col in ratings.columns:
            result['timestamp'] = ratings[timestamp_col]
        
        return result[['userId', 'movieId', 'rating'] + 
                      (['timestamp'] if 'timestamp' in result.columns else [])]
    
    @staticmethod
    def from_netflix_format(df: pd.DataFrame) -> pd.DataFrame:
        """Convert Netflix Prize format to standard"""
        # Netflix format: MovieId, CustomerId, Rating, Date
        return DataTransformer.to_standard_format(
            df,
            user_col='CustomerId',
            item_col='MovieId',
            rating_col='Rating',
            timestamp_col='Date'
        )
    
    @staticmethod
    def from_amazon_format(df: pd.DataFrame) -> pd.DataFrame:
        """Convert Amazon Review format to standard"""
        # Amazon format: reviewerID, asin, overall, unixReviewTime
        result = DataTransformer.to_standard_format(
            df,
            user_col='reviewerID',
            item_col='asin',
            rating_col='overall',
            timestamp_col='unixReviewTime'
        )
        
        # Hash string IDs to integers
        result['userId'] = pd.factorize(result['userId'])[0] + 1
        result['movieId'] = pd.factorize(result['movieId'])[0] + 1
        
        return result


# =============================================================================
# Convenience Functions
# =============================================================================

def create_adapter(adapter_type: str, **kwargs) -> Optional[BaseDataAdapter]:
    """Create a data adapter"""
    return DataAdapterRegistry().create(adapter_type, **kwargs)


def load_movielens(data_dir: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load MovieLens dataset.
    
    Returns:
        Tuple of (ratings_df, movies_df)
    """
    adapter = MovieLensAdapter(data_dir)
    adapter.connect()
    
    ratings = adapter.load_ratings()
    movies = adapter.load_items()
    
    adapter.disconnect()
    return ratings, movies


def load_from_csv(
    ratings_file: Union[str, Path],
    items_file: Optional[Union[str, Path]] = None,
    column_mapping: Optional[Dict[str, str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data from CSV files.
    
    Returns:
        Tuple of (ratings_df, items_df)
    """
    adapter = CSVAdapter(
        ratings_file=ratings_file,
        items_file=items_file,
        column_mapping=column_mapping
    )
    adapter.connect()
    
    ratings = adapter.load_ratings()
    items = adapter.load_items()
    
    adapter.disconnect()
    return ratings, items


def load_from_database(
    connection_string: str,
    ratings_table: str = "ratings",
    items_table: str = "items"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data from database.
    
    Returns:
        Tuple of (ratings_df, items_df)
    """
    adapter = SQLAdapter(
        connection_string=connection_string,
        ratings_table=ratings_table,
        items_table=items_table
    )
    adapter.connect()
    
    ratings = adapter.load_ratings()
    items = adapter.load_items()
    
    adapter.disconnect()
    return ratings, items
