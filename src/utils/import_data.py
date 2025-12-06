"""
Import Module for CineMatch V2.1.6

Provides functionality to import data from various formats
including CSV, JSON, and other common data sources.

Phase 6 - Task 6.4: Import Capabilities
"""

import csv
import io
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ImportOptions:
    """Configuration options for imports."""
    encoding: str = "utf-8"
    date_format: str = "%Y-%m-%d"
    datetime_format: str = "%Y-%m-%d %H:%M:%S"
    null_values: List[str] = field(default_factory=lambda: ["", "null", "NULL", "None", "NA", "N/A"])
    skip_rows: int = 0
    max_rows: Optional[int] = None
    validate: bool = True
    skip_invalid: bool = False
    # CSV specific
    csv_delimiter: str = ","
    csv_quotechar: str = '"'
    csv_has_header: bool = True
    # JSON specific
    json_path: Optional[str] = None  # JSONPath to data array


@dataclass
class ImportResult:
    """Result of an import operation."""
    success: bool
    data: Optional[pd.DataFrame]
    total_rows: int = 0
    imported_rows: int = 0
    skipped_rows: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "total_rows": self.total_rows,
            "imported_rows": self.imported_rows,
            "skipped_rows": self.skipped_rows,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "duration_seconds": self.duration_seconds
        }


@dataclass
class ColumnMapping:
    """Mapping configuration for a column."""
    source_name: str
    target_name: str
    data_type: str = "string"  # string, int, float, datetime, bool
    required: bool = False
    default_value: Any = None
    transformer: Optional[Callable[[Any], Any]] = None
    validator: Optional[Callable[[Any], bool]] = None


@dataclass
class ImportSchema:
    """Schema definition for imports."""
    name: str
    columns: List[ColumnMapping]
    required_columns: Set[str] = field(default_factory=set)
    unique_columns: Set[str] = field(default_factory=set)
    
    def get_column_mapping(self) -> Dict[str, str]:
        """Get source to target column mapping."""
        return {c.source_name: c.target_name for c in self.columns}
    
    def get_required_source_columns(self) -> Set[str]:
        """Get required source column names."""
        return {c.source_name for c in self.columns if c.required}


class DataValidator:
    """Validates imported data."""
    
    @staticmethod
    def validate_column_type(value: Any, data_type: str) -> Tuple[bool, Any, Optional[str]]:
        """
        Validate and convert column value.
        
        Returns:
            Tuple of (is_valid, converted_value, error_message)
        """
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return True, None, None
        
        try:
            if data_type == "string":
                return True, str(value), None
            elif data_type == "int":
                return True, int(float(value)), None
            elif data_type == "float":
                return True, float(value), None
            elif data_type == "bool":
                if isinstance(value, bool):
                    return True, value, None
                if isinstance(value, str):
                    if value.lower() in ("true", "yes", "1", "t", "y"):
                        return True, True, None
                    if value.lower() in ("false", "no", "0", "f", "n"):
                        return True, False, None
                return False, None, f"Cannot convert '{value}' to boolean"
            elif data_type == "datetime":
                if isinstance(value, datetime):
                    return True, value, None
                return True, pd.to_datetime(value), None
            else:
                return True, value, None
        except Exception as e:
            return False, None, f"Cannot convert '{value}' to {data_type}: {e}"
    
    @staticmethod
    def validate_row(
        row: Dict[str, Any],
        schema: ImportSchema
    ) -> Tuple[bool, Dict[str, Any], List[str]]:
        """
        Validate and transform a row according to schema.
        
        Returns:
            Tuple of (is_valid, transformed_row, errors)
        """
        errors = []
        transformed = {}
        
        for mapping in schema.columns:
            source_value = row.get(mapping.source_name)
            
            # Check required
            if mapping.required and source_value is None:
                errors.append(f"Missing required column: {mapping.source_name}")
                continue
            
            # Apply default
            if source_value is None:
                source_value = mapping.default_value
            
            # Validate type
            is_valid, converted, error = DataValidator.validate_column_type(
                source_value, mapping.data_type
            )
            if not is_valid:
                errors.append(error)
                continue
            
            # Apply transformer
            if mapping.transformer and converted is not None:
                try:
                    converted = mapping.transformer(converted)
                except Exception as e:
                    errors.append(f"Transform error for {mapping.source_name}: {e}")
                    continue
            
            # Apply validator
            if mapping.validator and converted is not None:
                if not mapping.validator(converted):
                    errors.append(f"Validation failed for {mapping.source_name}: {converted}")
                    continue
            
            transformed[mapping.target_name] = converted
        
        return len(errors) == 0, transformed, errors


class ImportFormat(ABC):
    """Abstract base class for import formats."""
    
    @property
    @abstractmethod
    def format_name(self) -> str:
        """Get format name."""
        pass
    
    @property
    @abstractmethod
    def file_extensions(self) -> List[str]:
        """Get supported file extensions."""
        pass
    
    @abstractmethod
    def import_data(
        self,
        source: Union[str, bytes, Path, io.IOBase],
        options: Optional[ImportOptions] = None
    ) -> ImportResult:
        """
        Import data from source.
        
        Args:
            source: Data source (file path, bytes, or file-like object)
            options: Import options
            
        Returns:
            ImportResult with data and statistics
        """
        pass


class CSVImporter(ImportFormat):
    """CSV format importer."""
    
    @property
    def format_name(self) -> str:
        return "CSV"
    
    @property
    def file_extensions(self) -> List[str]:
        return [".csv", ".tsv", ".txt"]
    
    def import_data(
        self,
        source: Union[str, bytes, Path, io.IOBase],
        options: Optional[ImportOptions] = None
    ) -> ImportResult:
        """Import data from CSV source."""
        import time
        start_time = time.time()
        options = options or ImportOptions()
        
        try:
            # Handle different source types
            if isinstance(source, bytes):
                source = io.StringIO(source.decode(options.encoding))
            elif isinstance(source, (str, Path)):
                source = Path(source)
                if not source.exists():
                    return ImportResult(
                        success=False,
                        data=None,
                        errors=[{"message": f"File not found: {source}"}]
                    )
                source = open(source, 'r', encoding=options.encoding)
            
            # Read CSV
            df = pd.read_csv(
                source,
                delimiter=options.csv_delimiter,
                quotechar=options.csv_quotechar,
                header=0 if options.csv_has_header else None,
                skiprows=options.skip_rows,
                nrows=options.max_rows,
                na_values=options.null_values,
                encoding=options.encoding
            )
            
            total_rows = len(df)
            
            return ImportResult(
                success=True,
                data=df,
                total_rows=total_rows,
                imported_rows=total_rows,
                duration_seconds=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"CSV import error: {e}")
            return ImportResult(
                success=False,
                data=None,
                errors=[{"message": str(e)}],
                duration_seconds=time.time() - start_time
            )


class JSONImporter(ImportFormat):
    """JSON format importer."""
    
    @property
    def format_name(self) -> str:
        return "JSON"
    
    @property
    def file_extensions(self) -> List[str]:
        return [".json"]
    
    def import_data(
        self,
        source: Union[str, bytes, Path, io.IOBase],
        options: Optional[ImportOptions] = None
    ) -> ImportResult:
        """Import data from JSON source."""
        import time
        start_time = time.time()
        options = options or ImportOptions()
        
        try:
            # Read JSON
            if isinstance(source, bytes):
                data = json.loads(source.decode(options.encoding))
            elif isinstance(source, (str, Path)):
                source = Path(source)
                if not source.exists():
                    return ImportResult(
                        success=False,
                        data=None,
                        errors=[{"message": f"File not found: {source}"}]
                    )
                with open(source, 'r', encoding=options.encoding) as f:
                    data = json.load(f)
            else:
                data = json.load(source)
            
            # Navigate to data path if specified
            if options.json_path:
                for key in options.json_path.split('.'):
                    if isinstance(data, dict) and key in data:
                        data = data[key]
                    elif isinstance(data, list) and key.isdigit():
                        data = data[int(key)]
                    else:
                        return ImportResult(
                            success=False,
                            data=None,
                            errors=[{"message": f"JSON path not found: {options.json_path}"}]
                        )
            
            # Handle different data structures
            if isinstance(data, dict):
                if "data" in data:
                    data = data["data"]
                else:
                    data = [data]
            
            if not isinstance(data, list):
                data = [data]
            
            df = pd.DataFrame(data)
            total_rows = len(df)
            
            # Apply row limits
            if options.skip_rows > 0:
                df = df.iloc[options.skip_rows:]
            if options.max_rows:
                df = df.iloc[:options.max_rows]
            
            return ImportResult(
                success=True,
                data=df,
                total_rows=total_rows,
                imported_rows=len(df),
                duration_seconds=time.time() - start_time
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return ImportResult(
                success=False,
                data=None,
                errors=[{"message": f"JSON parse error: {e}"}],
                duration_seconds=time.time() - start_time
            )
        except Exception as e:
            logger.error(f"JSON import error: {e}")
            return ImportResult(
                success=False,
                data=None,
                errors=[{"message": str(e)}],
                duration_seconds=time.time() - start_time
            )


class ExcelImporter(ImportFormat):
    """Excel format importer."""
    
    @property
    def format_name(self) -> str:
        return "Excel"
    
    @property
    def file_extensions(self) -> List[str]:
        return [".xlsx", ".xls", ".xlsm"]
    
    def import_data(
        self,
        source: Union[str, bytes, Path, io.IOBase],
        options: Optional[ImportOptions] = None,
        sheet_name: Optional[str] = None
    ) -> ImportResult:
        """Import data from Excel source."""
        import time
        start_time = time.time()
        options = options or ImportOptions()
        
        try:
            if isinstance(source, bytes):
                source = io.BytesIO(source)
            elif isinstance(source, (str, Path)):
                source = Path(source)
                if not source.exists():
                    return ImportResult(
                        success=False,
                        data=None,
                        errors=[{"message": f"File not found: {source}"}]
                    )
            
            df = pd.read_excel(
                source,
                sheet_name=sheet_name or 0,
                skiprows=options.skip_rows,
                nrows=options.max_rows,
                na_values=options.null_values
            )
            
            total_rows = len(df)
            
            return ImportResult(
                success=True,
                data=df,
                total_rows=total_rows,
                imported_rows=total_rows,
                duration_seconds=time.time() - start_time
            )
            
        except ImportError:
            return ImportResult(
                success=False,
                data=None,
                errors=[{"message": "openpyxl package required for Excel import"}],
                duration_seconds=time.time() - start_time
            )
        except Exception as e:
            logger.error(f"Excel import error: {e}")
            return ImportResult(
                success=False,
                data=None,
                errors=[{"message": str(e)}],
                duration_seconds=time.time() - start_time
            )


class ImportManager:
    """Manages data imports across formats."""
    
    def __init__(self):
        self._importers: Dict[str, ImportFormat] = {}
        # Register default importers
        self.register_importer(CSVImporter())
        self.register_importer(JSONImporter())
        self.register_importer(ExcelImporter())
    
    def register_importer(self, importer: ImportFormat) -> None:
        """Register an importer."""
        for ext in importer.file_extensions:
            self._importers[ext.lower()] = importer
        self._importers[importer.format_name.lower()] = importer
    
    def get_importer(self, format_or_extension: str) -> Optional[ImportFormat]:
        """Get importer by format name or extension."""
        key = format_or_extension.lower()
        if not key.startswith('.'):
            key_with_dot = '.' + key
            if key_with_dot in self._importers:
                return self._importers[key_with_dot]
        return self._importers.get(key)
    
    def import_from_file(
        self,
        filepath: Union[str, Path],
        options: Optional[ImportOptions] = None,
        schema: Optional[ImportSchema] = None
    ) -> ImportResult:
        """Import data from file with automatic format detection."""
        filepath = Path(filepath)
        extension = filepath.suffix.lower()
        
        importer = self.get_importer(extension)
        if not importer:
            return ImportResult(
                success=False,
                data=None,
                errors=[{"message": f"No importer found for extension: {extension}"}]
            )
        
        result = importer.import_data(filepath, options)
        
        # Apply schema validation if provided
        if result.success and schema and options and options.validate:
            result = self._apply_schema(result, schema, options)
        
        return result
    
    def import_from_bytes(
        self,
        data: bytes,
        format: str,
        options: Optional[ImportOptions] = None,
        schema: Optional[ImportSchema] = None
    ) -> ImportResult:
        """Import data from bytes."""
        importer = self.get_importer(format)
        if not importer:
            return ImportResult(
                success=False,
                data=None,
                errors=[{"message": f"No importer found for format: {format}"}]
            )
        
        result = importer.import_data(data, options)
        
        if result.success and schema and options and options.validate:
            result = self._apply_schema(result, schema, options)
        
        return result
    
    def _apply_schema(
        self,
        result: ImportResult,
        schema: ImportSchema,
        options: ImportOptions
    ) -> ImportResult:
        """Apply schema validation to import result."""
        if result.data is None:
            return result
        
        df = result.data
        
        # Check required columns
        missing_cols = schema.get_required_source_columns() - set(df.columns)
        if missing_cols:
            if options.skip_invalid:
                result.warnings.append(f"Missing required columns: {missing_cols}")
            else:
                result.success = False
                result.errors.append({
                    "message": f"Missing required columns: {missing_cols}"
                })
                return result
        
        # Validate and transform rows
        transformed_rows = []
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            is_valid, transformed, errors = DataValidator.validate_row(row_dict, schema)
            
            if is_valid:
                transformed_rows.append(transformed)
            else:
                if options.skip_invalid:
                    result.skipped_rows += 1
                    result.warnings.append(f"Row {idx}: {errors}")
                else:
                    result.errors.append({
                        "row": idx,
                        "errors": errors
                    })
        
        if result.errors and not options.skip_invalid:
            result.success = False
        else:
            result.data = pd.DataFrame(transformed_rows)
            result.imported_rows = len(transformed_rows)
        
        return result


# Pre-defined schemas for CineMatch data types

class CineMatchSchemas:
    """Pre-defined import schemas for CineMatch."""
    
    RATINGS = ImportSchema(
        name="ratings",
        columns=[
            ColumnMapping("user_id", "user_id", "int", required=True),
            ColumnMapping("movie_id", "movie_id", "int", required=True),
            ColumnMapping("rating", "rating", "float", required=True,
                         validator=lambda x: 0.5 <= x <= 5.0),
            ColumnMapping("timestamp", "timestamp", "datetime"),
        ],
        required_columns={"user_id", "movie_id", "rating"},
        unique_columns={"user_id", "movie_id"}
    )
    
    MOVIES = ImportSchema(
        name="movies",
        columns=[
            ColumnMapping("movie_id", "movie_id", "int", required=True),
            ColumnMapping("title", "title", "string", required=True),
            ColumnMapping("genres", "genres", "string"),
            ColumnMapping("year", "year", "int"),
            ColumnMapping("imdb_id", "imdb_id", "string"),
            ColumnMapping("tmdb_id", "tmdb_id", "int"),
        ],
        required_columns={"movie_id", "title"},
        unique_columns={"movie_id"}
    )
    
    USERS = ImportSchema(
        name="users",
        columns=[
            ColumnMapping("user_id", "user_id", "int", required=True),
            ColumnMapping("username", "username", "string"),
            ColumnMapping("email", "email", "string"),
            ColumnMapping("created_at", "created_at", "datetime"),
        ],
        required_columns={"user_id"},
        unique_columns={"user_id"}
    )
    
    TAGS = ImportSchema(
        name="tags",
        columns=[
            ColumnMapping("user_id", "user_id", "int", required=True),
            ColumnMapping("movie_id", "movie_id", "int", required=True),
            ColumnMapping("tag", "tag", "string", required=True),
            ColumnMapping("timestamp", "timestamp", "datetime"),
        ],
        required_columns={"user_id", "movie_id", "tag"}
    )
    
    @classmethod
    def get_schema(cls, name: str) -> Optional[ImportSchema]:
        """Get schema by name."""
        return getattr(cls, name.upper(), None)


class RatingsImporter:
    """Specialized importer for user ratings."""
    
    def __init__(self, import_manager: Optional[ImportManager] = None):
        self._manager = import_manager or ImportManager()
    
    def import_ratings(
        self,
        source: Union[str, bytes, Path],
        format: Optional[str] = None,
        options: Optional[ImportOptions] = None
    ) -> ImportResult:
        """Import user ratings data."""
        options = options or ImportOptions()
        options.validate = True
        
        if isinstance(source, (str, Path)):
            result = self._manager.import_from_file(
                source, options, CineMatchSchemas.RATINGS
            )
        else:
            if not format:
                format = "csv"  # Default to CSV for bytes
            result = self._manager.import_from_bytes(
                source, format, options, CineMatchSchemas.RATINGS
            )
        
        return result
    
    def validate_ratings(self, df: pd.DataFrame) -> List[str]:
        """Validate ratings DataFrame."""
        errors = []
        
        # Check rating range
        if df['rating'].min() < 0.5 or df['rating'].max() > 5.0:
            errors.append("Ratings must be between 0.5 and 5.0")
        
        # Check for duplicates
        duplicates = df.duplicated(subset=['user_id', 'movie_id'], keep=False)
        if duplicates.any():
            dup_count = duplicates.sum()
            errors.append(f"Found {dup_count} duplicate user-movie ratings")
        
        return errors


class MovieImporter:
    """Specialized importer for movie data."""
    
    def __init__(self, import_manager: Optional[ImportManager] = None):
        self._manager = import_manager or ImportManager()
    
    def import_movies(
        self,
        source: Union[str, bytes, Path],
        format: Optional[str] = None,
        options: Optional[ImportOptions] = None
    ) -> ImportResult:
        """Import movie data."""
        options = options or ImportOptions()
        options.validate = True
        
        if isinstance(source, (str, Path)):
            result = self._manager.import_from_file(
                source, options, CineMatchSchemas.MOVIES
            )
        else:
            if not format:
                format = "csv"
            result = self._manager.import_from_bytes(
                source, format, options, CineMatchSchemas.MOVIES
            )
        
        # Parse year from title if not present
        if result.success and result.data is not None:
            df = result.data
            if 'year' not in df.columns or df['year'].isna().all():
                df['year'] = df['title'].apply(self._extract_year)
        
        return result
    
    def _extract_year(self, title: str) -> Optional[int]:
        """Extract year from movie title like 'Movie Name (1999)'."""
        if pd.isna(title):
            return None
        match = re.search(r'\((\d{4})\)\s*$', str(title))
        if match:
            return int(match.group(1))
        return None


# Global import manager instance
_import_manager: Optional[ImportManager] = None


def get_import_manager() -> ImportManager:
    """Get global import manager instance."""
    global _import_manager
    if _import_manager is None:
        _import_manager = ImportManager()
    return _import_manager


def import_from_file(
    filepath: Union[str, Path],
    options: Optional[ImportOptions] = None,
    schema: Optional[ImportSchema] = None
) -> ImportResult:
    """Convenience function for importing from file."""
    return get_import_manager().import_from_file(filepath, options, schema)


def import_ratings(
    source: Union[str, bytes, Path],
    format: Optional[str] = None,
    options: Optional[ImportOptions] = None
) -> ImportResult:
    """Convenience function for importing ratings."""
    return RatingsImporter(get_import_manager()).import_ratings(source, format, options)


def import_movies(
    source: Union[str, bytes, Path],
    format: Optional[str] = None,
    options: Optional[ImportOptions] = None
) -> ImportResult:
    """Convenience function for importing movies."""
    return MovieImporter(get_import_manager()).import_movies(source, format, options)
