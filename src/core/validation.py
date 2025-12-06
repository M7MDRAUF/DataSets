"""
CineMatch V2.1.6 - Validation Module

Comprehensive input validation framework with declarative rules,
custom validators, and detailed error messages.

Author: CineMatch Development Team
"""

from abc import ABC, abstractmethod
from typing import (
    Dict, Any, Optional, List, Callable, TypeVar, Generic,
    Union, Type, Tuple, Set
)
from dataclasses import dataclass, field
from enum import Enum
import re
from datetime import datetime
import logging


logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# Validation Results
# =============================================================================

@dataclass
class ValidationError:
    """Single validation error"""
    field: str
    message: str
    code: str
    value: Any = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'field': self.field,
            'message': self.message,
            'code': self.code
        }


@dataclass
class ValidationResult:
    """Validation result container"""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    
    def add_error(self, field: str, message: str, code: str, value: Any = None) -> None:
        self.errors.append(ValidationError(field, message, code, value))
        self.is_valid = False
    
    def add_warning(self, field: str, message: str, code: str) -> None:
        self.warnings.append(ValidationError(field, message, code))
    
    def merge(self, other: 'ValidationResult') -> None:
        """Merge another validation result"""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.is_valid:
            self.is_valid = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_valid': self.is_valid,
            'errors': [e.to_dict() for e in self.errors],
            'warnings': [w.to_dict() for w in self.warnings]
        }
    
    def raise_if_invalid(self) -> None:
        """Raise exception if validation failed"""
        if not self.is_valid:
            error_messages = [f"{e.field}: {e.message}" for e in self.errors]
            raise ValidationException(
                f"Validation failed: {'; '.join(error_messages)}",
                self.errors
            )


class ValidationException(Exception):
    """Exception raised when validation fails"""
    
    def __init__(self, message: str, errors: List[ValidationError]):
        super().__init__(message)
        self.errors = errors


# =============================================================================
# Validator Interface
# =============================================================================

class IValidator(ABC, Generic[T]):
    """Abstract validator interface"""
    
    @abstractmethod
    def validate(self, value: T, field_name: str = "value") -> ValidationResult:
        """Validate value and return result"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get human-readable description of validation rule"""
        pass


# =============================================================================
# Built-in Validators
# =============================================================================

class RequiredValidator(IValidator[Any]):
    """Validates that value is not None or empty"""
    
    def __init__(self, allow_empty: bool = False):
        self.allow_empty = allow_empty
    
    def validate(self, value: Any, field_name: str = "value") -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        if value is None:
            result.add_error(field_name, "Value is required", "required")
        elif not self.allow_empty:
            if isinstance(value, str) and value.strip() == "":
                result.add_error(field_name, "Value cannot be empty", "empty")
            elif isinstance(value, (list, dict)) and len(value) == 0:
                result.add_error(field_name, "Value cannot be empty", "empty")
        
        return result
    
    def get_description(self) -> str:
        return "Value is required" + ("" if self.allow_empty else " and cannot be empty")


class TypeValidator(IValidator[Any]):
    """Validates value type"""
    
    def __init__(self, expected_type: Union[Type, Tuple[Type, ...]]):
        self.expected_type = expected_type
    
    def validate(self, value: Any, field_name: str = "value") -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        if value is not None and not isinstance(value, self.expected_type):
            type_name = (
                self.expected_type.__name__ 
                if isinstance(self.expected_type, type) 
                else str(self.expected_type)
            )
            result.add_error(
                field_name,
                f"Expected type {type_name}, got {type(value).__name__}",
                "invalid_type",
                value
            )
        
        return result
    
    def get_description(self) -> str:
        type_name = (
            self.expected_type.__name__ 
            if isinstance(self.expected_type, type) 
            else str(self.expected_type)
        )
        return f"Must be of type {type_name}"


class RangeValidator(IValidator[Union[int, float]]):
    """Validates numeric value is within range"""
    
    def __init__(
        self,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        inclusive: bool = True
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.inclusive = inclusive
    
    def validate(self, value: Union[int, float], field_name: str = "value") -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        if value is None:
            return result
        
        if self.min_value is not None:
            if self.inclusive:
                if value < self.min_value:
                    result.add_error(
                        field_name,
                        f"Value must be >= {self.min_value}",
                        "min_value",
                        value
                    )
            else:
                if value <= self.min_value:
                    result.add_error(
                        field_name,
                        f"Value must be > {self.min_value}",
                        "min_value",
                        value
                    )
        
        if self.max_value is not None:
            if self.inclusive:
                if value > self.max_value:
                    result.add_error(
                        field_name,
                        f"Value must be <= {self.max_value}",
                        "max_value",
                        value
                    )
            else:
                if value >= self.max_value:
                    result.add_error(
                        field_name,
                        f"Value must be < {self.max_value}",
                        "max_value",
                        value
                    )
        
        return result
    
    def get_description(self) -> str:
        parts = []
        if self.min_value is not None:
            op = ">=" if self.inclusive else ">"
            parts.append(f"{op} {self.min_value}")
        if self.max_value is not None:
            op = "<=" if self.inclusive else "<"
            parts.append(f"{op} {self.max_value}")
        return f"Value must be {' and '.join(parts)}"


class LengthValidator(IValidator[Union[str, list, dict]]):
    """Validates length/size of value"""
    
    def __init__(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None
    ):
        self.min_length = min_length
        self.max_length = max_length
    
    def validate(self, value: Union[str, list, dict], field_name: str = "value") -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        if value is None:
            return result
        
        length = len(value)
        
        if self.min_length is not None and length < self.min_length:
            result.add_error(
                field_name,
                f"Length must be at least {self.min_length}",
                "min_length",
                value
            )
        
        if self.max_length is not None and length > self.max_length:
            result.add_error(
                field_name,
                f"Length must not exceed {self.max_length}",
                "max_length",
                value
            )
        
        return result
    
    def get_description(self) -> str:
        parts = []
        if self.min_length is not None:
            parts.append(f"at least {self.min_length}")
        if self.max_length is not None:
            parts.append(f"at most {self.max_length}")
        return f"Length must be {' and '.join(parts)}"


class RegexValidator(IValidator[str]):
    """Validates string matches regex pattern"""
    
    def __init__(self, pattern: str, message: Optional[str] = None):
        self.pattern = pattern
        self.compiled = re.compile(pattern)
        self.message = message
    
    def validate(self, value: str, field_name: str = "value") -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        if value is None:
            return result
        
        if not self.compiled.match(value):
            result.add_error(
                field_name,
                self.message or f"Value does not match pattern",
                "pattern",
                value
            )
        
        return result
    
    def get_description(self) -> str:
        return self.message or f"Must match pattern: {self.pattern}"


class ChoiceValidator(IValidator[Any]):
    """Validates value is one of allowed choices"""
    
    def __init__(self, choices: Union[List[Any], Set[Any], Type[Enum]]):
        if isinstance(choices, type) and issubclass(choices, Enum):
            self.choices = set(e.value for e in choices)
        else:
            self.choices = set(choices)
    
    def validate(self, value: Any, field_name: str = "value") -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        if value is None:
            return result
        
        if value not in self.choices:
            choices_str = ", ".join(str(c) for c in list(self.choices)[:5])
            result.add_error(
                field_name,
                f"Value must be one of: {choices_str}",
                "invalid_choice",
                value
            )
        
        return result
    
    def get_description(self) -> str:
        choices_str = ", ".join(str(c) for c in list(self.choices)[:5])
        return f"Must be one of: {choices_str}"


class EmailValidator(IValidator[str]):
    """Validates email format"""
    
    EMAIL_PATTERN = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    def __init__(self):
        self.regex = RegexValidator(self.EMAIL_PATTERN, "Invalid email format")
    
    def validate(self, value: str, field_name: str = "value") -> ValidationResult:
        return self.regex.validate(value, field_name)
    
    def get_description(self) -> str:
        return "Must be a valid email address"


class DateValidator(IValidator[Union[str, datetime]]):
    """Validates date/datetime value"""
    
    def __init__(
        self,
        min_date: Optional[datetime] = None,
        max_date: Optional[datetime] = None,
        format: str = "%Y-%m-%d"
    ):
        self.min_date = min_date
        self.max_date = max_date
        self.format = format
    
    def validate(self, value: Union[str, datetime], field_name: str = "value") -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        if value is None:
            return result
        
        # Parse if string
        if isinstance(value, str):
            try:
                value = datetime.strptime(value, self.format)
            except ValueError:
                result.add_error(
                    field_name,
                    f"Invalid date format. Expected: {self.format}",
                    "invalid_date"
                )
                return result
        
        if self.min_date and value < self.min_date:
            result.add_error(
                field_name,
                f"Date must be on or after {self.min_date.strftime(self.format)}",
                "min_date"
            )
        
        if self.max_date and value > self.max_date:
            result.add_error(
                field_name,
                f"Date must be on or before {self.max_date.strftime(self.format)}",
                "max_date"
            )
        
        return result
    
    def get_description(self) -> str:
        return f"Must be a valid date in format {self.format}"


class ListValidator(IValidator[List]):
    """Validates each item in list"""
    
    def __init__(self, item_validator: IValidator):
        self.item_validator = item_validator
    
    def validate(self, value: List, field_name: str = "value") -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        if value is None:
            return result
        
        for i, item in enumerate(value):
            item_result = self.item_validator.validate(item, f"{field_name}[{i}]")
            result.merge(item_result)
        
        return result
    
    def get_description(self) -> str:
        return f"Each item: {self.item_validator.get_description()}"


class CompositeValidator(IValidator[T]):
    """Combines multiple validators"""
    
    def __init__(self, validators: List[IValidator[T]], stop_on_first_error: bool = False):
        self.validators = validators
        self.stop_on_first_error = stop_on_first_error
    
    def validate(self, value: T, field_name: str = "value") -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        for validator in self.validators:
            v_result = validator.validate(value, field_name)
            result.merge(v_result)
            
            if self.stop_on_first_error and not result.is_valid:
                break
        
        return result
    
    def get_description(self) -> str:
        return "; ".join(v.get_description() for v in self.validators)


class CustomValidator(IValidator[T]):
    """Custom validation function"""
    
    def __init__(
        self,
        func: Callable[[T], bool],
        error_message: str,
        error_code: str = "custom"
    ):
        self.func = func
        self.error_message = error_message
        self.error_code = error_code
    
    def validate(self, value: T, field_name: str = "value") -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        if value is None:
            return result
        
        try:
            if not self.func(value):
                result.add_error(field_name, self.error_message, self.error_code, value)
        except Exception as e:
            result.add_error(field_name, str(e), "validation_error", value)
        
        return result
    
    def get_description(self) -> str:
        return self.error_message


# =============================================================================
# Domain-Specific Validators
# =============================================================================

class UserIdValidator(CompositeValidator[int]):
    """Validates user ID"""
    
    def __init__(self):
        super().__init__([
            RequiredValidator(),
            TypeValidator(int),
            RangeValidator(min_value=1)
        ])


class MovieIdValidator(CompositeValidator[int]):
    """Validates movie ID"""
    
    def __init__(self):
        super().__init__([
            RequiredValidator(),
            TypeValidator(int),
            RangeValidator(min_value=1)
        ])


class RatingValidator(CompositeValidator[float]):
    """Validates movie rating"""
    
    def __init__(self):
        super().__init__([
            RequiredValidator(),
            TypeValidator((int, float)),
            RangeValidator(min_value=0.5, max_value=5.0),
            CustomValidator(
                lambda x: x * 2 == int(x * 2),
                "Rating must be in 0.5 increments",
                "rating_increment"
            )
        ])


class SearchQueryValidator(CompositeValidator[str]):
    """Validates search query"""
    
    def __init__(self, min_length: int = 1, max_length: int = 200):
        super().__init__([
            RequiredValidator(),
            TypeValidator(str),
            LengthValidator(min_length=min_length, max_length=max_length),
            CustomValidator(
                lambda x: not x.strip().startswith('*'),
                "Query cannot start with wildcard",
                "invalid_query"
            )
        ])


class PaginationValidator:
    """Validates pagination parameters"""
    
    def __init__(self, max_page_size: int = 100):
        self.max_page_size = max_page_size
        self.page_validator = CompositeValidator([
            TypeValidator(int),
            RangeValidator(min_value=1)
        ])
        self.size_validator = CompositeValidator([
            TypeValidator(int),
            RangeValidator(min_value=1, max_value=max_page_size)
        ])
    
    def validate(self, page: int, page_size: int) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        result.merge(self.page_validator.validate(page, "page"))
        result.merge(self.size_validator.validate(page_size, "page_size"))
        return result


# =============================================================================
# Schema Validator
# =============================================================================

@dataclass
class FieldSchema:
    """Schema for a single field"""
    name: str
    validators: List[IValidator]
    required: bool = True
    default: Any = None


class SchemaValidator:
    """
    Validates objects against schema.
    
    Usage:
        schema = SchemaValidator({
            'user_id': FieldSchema('user_id', [UserIdValidator()]),
            'rating': FieldSchema('rating', [RatingValidator()])
        })
        result = schema.validate(data)
    """
    
    def __init__(self, fields: Dict[str, FieldSchema]):
        self.fields = fields
    
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate data against schema"""
        result = ValidationResult(is_valid=True)
        
        for field_name, schema in self.fields.items():
            value = data.get(field_name, schema.default)
            
            # Check required
            if schema.required and value is None:
                result.add_error(field_name, "Field is required", "required")
                continue
            
            # Run validators
            for validator in schema.validators:
                v_result = validator.validate(value, field_name)
                result.merge(v_result)
        
        return result
    
    def validate_or_raise(self, data: Dict[str, Any]) -> None:
        """Validate and raise on error"""
        result = self.validate(data)
        result.raise_if_invalid()


# =============================================================================
# Validation Decorator
# =============================================================================

def validate_args(**field_validators: IValidator):
    """
    Decorator to validate function arguments.
    
    Usage:
        @validate_args(
            user_id=UserIdValidator(),
            rating=RatingValidator()
        )
        def rate_movie(user_id: int, movie_id: int, rating: float):
            ...
    """
    def decorator(func: Callable) -> Callable:
        from functools import wraps
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            result = ValidationResult(is_valid=True)
            
            for param_name, validator in field_validators.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    v_result = validator.validate(value, param_name)
                    result.merge(v_result)
            
            result.raise_if_invalid()
            return func(*args, **kwargs)
        
        return wrapper
    return decorator
