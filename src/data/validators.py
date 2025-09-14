#!/usr/bin/env python3
"""
Phase 6: Data Validation Utilities
Comprehensive data validation with business rules and quality checks
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Set, Union, Tuple
from pathlib import Path
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of a validation check"""
    is_valid: bool
    message: str
    severity: str = "error"  # error, warning, info
    details: Optional[Dict[str, Any]] = None

@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    passed: int = 0
    failed: int = 0
    warnings: int = 0
    results: List[ValidationResult] = None

    def __post_init__(self):
        if self.results is None:
            self.results = []

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.warnings

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 1.0
        return self.passed / self.total

    def add_result(self, result: ValidationResult):
        """Add a validation result"""
        self.results.append(result)
        if result.severity == "error" and not result.is_valid:
            self.failed += 1
        elif result.severity == "warning":
            self.warnings += 1
        else:
            self.passed += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary"""
        return {
            'total_checks': self.total,
            'passed': self.passed,
            'failed': self.failed,
            'warnings': self.warnings,
            'success_rate': self.success_rate,
            'errors': [r.message for r in self.results if r.severity == "error" and not r.is_valid],
            'warnings_list': [r.message for r in self.results if r.severity == "warning"]
        }

class BaseValidator(ABC):
    """Abstract base class for data validators"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.strict_mode = self.config.get('strict_mode', False)
        self.report = ValidationReport()

    @abstractmethod
    def validate(self, data: pd.DataFrame) -> ValidationReport:
        """Validate the data and return report"""
        pass

    def _add_validation_result(self, is_valid: bool, message: str,
                             severity: str = "error", details: Optional[Dict] = None):
        """Helper to add validation result"""
        result = ValidationResult(is_valid, message, severity, details)
        self.report.add_result(result)

        if severity == "error" and not is_valid:
            logger.error(message)
        elif severity == "warning":
            logger.warning(message)
        else:
            logger.info(message)

class SchemaValidator(BaseValidator):
    """Validate data schema and structure"""

    def __init__(self, expected_schema: Dict[str, str], config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.expected_schema = expected_schema
        self.required_columns = self.config.get('required_columns', [])
        self.allow_extra_columns = self.config.get('allow_extra_columns', True)

    def validate(self, data: pd.DataFrame) -> ValidationReport:
        """Validate data schema"""
        self.report = ValidationReport()

        logger.info(f"Validating schema for {data.shape[0]} records")

        # Check required columns
        self._validate_required_columns(data)

        # Check data types
        self._validate_data_types(data)

        # Check for extra columns
        if not self.allow_extra_columns:
            self._validate_no_extra_columns(data)

        return self.report

    def _validate_required_columns(self, data: pd.DataFrame):
        """Check if all required columns are present"""
        missing_columns = set(self.required_columns) - set(data.columns)

        if missing_columns:
            self._add_validation_result(
                False,
                f"Missing required columns: {list(missing_columns)}",
                severity="error",
                details={'missing_columns': list(missing_columns)}
            )
        else:
            self._add_validation_result(
                True,
                "All required columns present",
                severity="info"
            )

    def _validate_data_types(self, data: pd.DataFrame):
        """Validate data types match expected schema"""
        type_mismatches = []

        for column, expected_type in self.expected_schema.items():
            if column in data.columns:
                actual_type = str(data[column].dtype)

                # Flexible type checking
                if not self._types_match(actual_type, expected_type):
                    type_mismatches.append({
                        'column': column,
                        'expected': expected_type,
                        'actual': actual_type
                    })

        if type_mismatches:
            self._add_validation_result(
                False,
                f"Data type mismatches found: {len(type_mismatches)} columns",
                severity="warning",
                details={'mismatches': type_mismatches}
            )
        else:
            self._add_validation_result(
                True,
                "All data types match schema",
                severity="info"
            )

    def _validate_no_extra_columns(self, data: pd.DataFrame):
        """Check for unexpected extra columns"""
        expected_columns = set(self.expected_schema.keys())
        actual_columns = set(data.columns)
        extra_columns = actual_columns - expected_columns

        if extra_columns:
            self._add_validation_result(
                False,
                f"Unexpected columns found: {list(extra_columns)}",
                severity="warning",
                details={'extra_columns': list(extra_columns)}
            )
        else:
            self._add_validation_result(
                True,
                "No unexpected columns",
                severity="info"
            )

    def _types_match(self, actual: str, expected: str) -> bool:
        """Check if data types match with flexibility"""
        type_mappings = {
            'int': ['int8', 'int16', 'int32', 'int64'],
            'float': ['float16', 'float32', 'float64'],
            'string': ['object', 'string'],
            'datetime': ['datetime64', '<M8[ns]'],
            'bool': ['bool'],
            'category': ['category']
        }

        for base_type, variants in type_mappings.items():
            if expected == base_type and any(variant in actual for variant in variants):
                return True
            if actual == expected:
                return True

        return False

class BusinessRulesValidator(BaseValidator):
    """Validate business rules and constraints"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.min_date = self.config.get('min_date')
        self.max_date = self.config.get('max_date')
        self.non_negative_columns = self.config.get('non_negative_columns', [])
        self.valid_ranges = self.config.get('valid_ranges', {})
        self.required_relationships = self.config.get('required_relationships', [])

    def validate(self, data: pd.DataFrame) -> ValidationReport:
        """Validate business rules"""
        self.report = ValidationReport()

        logger.info(f"Validating business rules for {data.shape[0]} records")

        # Date range validation
        if 'date' in data.columns:
            self._validate_date_range(data)

        # Non-negative values validation
        self._validate_non_negative_values(data)

        # Value ranges validation
        self._validate_value_ranges(data)

        # Relationship validation
        self._validate_relationships(data)

        return self.report

    def _validate_date_range(self, data: pd.DataFrame):
        """Validate date ranges"""
        date_col = 'date'

        if date_col not in data.columns:
            return

        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
            data[date_col] = pd.to_datetime(data[date_col], errors='coerce')

        # Check for null dates
        null_dates = data[date_col].isnull().sum()
        if null_dates > 0:
            self._add_validation_result(
                False,
                f"Found {null_dates} null/invalid dates",
                severity="error",
                details={'null_dates': null_dates}
            )

        # Check date range
        if self.min_date:
            min_date = pd.to_datetime(self.min_date)
            early_dates = (data[date_col] < min_date).sum()
            if early_dates > 0:
                self._add_validation_result(
                    False,
                    f"Found {early_dates} dates before minimum ({self.min_date})",
                    severity="error",
                    details={'early_dates': early_dates}
                )

        if self.max_date:
            max_date = pd.to_datetime(self.max_date)
            late_dates = (data[date_col] > max_date).sum()
            if late_dates > 0:
                self._add_validation_result(
                    False,
                    f"Found {late_dates} dates after maximum ({self.max_date})",
                    severity="warning",
                    details={'late_dates': late_dates}
                )

        # Check for future dates (suspicious)
        future_dates = (data[date_col] > datetime.now()).sum()
        if future_dates > 0:
            self._add_validation_result(
                True,
                f"Found {future_dates} future dates (may be intentional)",
                severity="warning",
                details={'future_dates': future_dates}
            )

    def _validate_non_negative_values(self, data: pd.DataFrame):
        """Validate non-negative constraints"""
        for column in self.non_negative_columns:
            if column in data.columns:
                negative_count = (data[column] < 0).sum()
                if negative_count > 0:
                    self._add_validation_result(
                        False,
                        f"Found {negative_count} negative values in {column}",
                        severity="error",
                        details={'column': column, 'negative_count': negative_count}
                    )
                else:
                    self._add_validation_result(
                        True,
                        f"No negative values in {column}",
                        severity="info"
                    )

    def _validate_value_ranges(self, data: pd.DataFrame):
        """Validate value ranges"""
        for column, range_config in self.valid_ranges.items():
            if column not in data.columns:
                continue

            min_val = range_config.get('min')
            max_val = range_config.get('max')

            if min_val is not None:
                below_min = (data[column] < min_val).sum()
                if below_min > 0:
                    self._add_validation_result(
                        False,
                        f"Found {below_min} values below minimum ({min_val}) in {column}",
                        severity="error"
                    )

            if max_val is not None:
                above_max = (data[column] > max_val).sum()
                if above_max > 0:
                    self._add_validation_result(
                        False,
                        f"Found {above_max} values above maximum ({max_val}) in {column}",
                        severity="warning"
                    )

    def _validate_relationships(self, data: pd.DataFrame):
        """Validate data relationships"""
        for relationship in self.required_relationships:
            parent_col = relationship.get('parent')
            child_col = relationship.get('child')

            if parent_col in data.columns and child_col in data.columns:
                # Check if all child values have corresponding parent values
                orphaned = ~data[child_col].isin(data[parent_col])
                orphaned_count = orphaned.sum()

                if orphaned_count > 0:
                    self._add_validation_result(
                        False,
                        f"Found {orphaned_count} orphaned records ({child_col} without {parent_col})",
                        severity="error"
                    )

class QualityValidator(BaseValidator):
    """Validate data quality metrics"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_null_percentage = self.config.get('max_null_percentage', 0.1)
        self.max_duplicate_percentage = self.config.get('max_duplicate_percentage', 0.05)
        self.min_unique_percentage = self.config.get('min_unique_percentage', {})

    def validate(self, data: pd.DataFrame) -> ValidationReport:
        """Validate data quality"""
        self.report = ValidationReport()

        logger.info(f"Validating data quality for {data.shape[0]} records")

        # Check missing values
        self._validate_missing_values(data)

        # Check duplicates
        self._validate_duplicates(data)

        # Check uniqueness
        self._validate_uniqueness(data)

        # Check statistical outliers
        self._validate_statistical_consistency(data)

        return self.report

    def _validate_missing_values(self, data: pd.DataFrame):
        """Check missing value percentages"""
        null_percentages = data.isnull().mean()

        high_null_columns = null_percentages[null_percentages > self.max_null_percentage]

        if not high_null_columns.empty:
            self._add_validation_result(
                False,
                f"High null percentages in columns: {dict(high_null_columns)}",
                severity="warning",
                details={'high_null_columns': dict(high_null_columns)}
            )
        else:
            self._add_validation_result(
                True,
                f"Null percentages acceptable (< {self.max_null_percentage:.1%})",
                severity="info"
            )

    def _validate_duplicates(self, data: pd.DataFrame):
        """Check for duplicate records"""
        duplicate_count = data.duplicated().sum()
        duplicate_percentage = duplicate_count / len(data)

        if duplicate_percentage > self.max_duplicate_percentage:
            self._add_validation_result(
                False,
                f"High duplicate percentage: {duplicate_percentage:.1%} ({duplicate_count} records)",
                severity="warning",
                details={'duplicate_count': duplicate_count, 'duplicate_percentage': duplicate_percentage}
            )
        else:
            self._add_validation_result(
                True,
                f"Duplicate percentage acceptable: {duplicate_percentage:.1%}",
                severity="info"
            )

    def _validate_uniqueness(self, data: pd.DataFrame):
        """Check uniqueness constraints"""
        for column, min_unique_pct in self.min_unique_percentage.items():
            if column in data.columns:
                unique_pct = data[column].nunique() / len(data)

                if unique_pct < min_unique_pct:
                    self._add_validation_result(
                        False,
                        f"Low uniqueness in {column}: {unique_pct:.1%} (expected > {min_unique_pct:.1%})",
                        severity="warning"
                    )
                else:
                    self._add_validation_result(
                        True,
                        f"Uniqueness acceptable in {column}: {unique_pct:.1%}",
                        severity="info"
                    )

    def _validate_statistical_consistency(self, data: pd.DataFrame):
        """Check statistical consistency"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns

        for column in numeric_columns:
            # Check for suspicious patterns
            series = data[column].dropna()

            if len(series) == 0:
                continue

            # Check for excessive zeros
            zero_percentage = (series == 0).mean()
            if zero_percentage > 0.9:
                self._add_validation_result(
                    True,
                    f"High zero percentage in {column}: {zero_percentage:.1%}",
                    severity="warning"
                )

            # Check for constant values
            if series.nunique() == 1:
                self._add_validation_result(
                    True,
                    f"Constant values detected in {column}",
                    severity="warning"
                )

class ComprehensiveValidator:
    """Comprehensive validator that combines multiple validation strategies"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.validators = []
        self._setup_validators()

    def _setup_validators(self):
        """Setup validation components"""
        # Schema validator
        if 'schema' in self.config:
            schema_validator = SchemaValidator(
                self.config['schema']['expected_schema'],
                self.config['schema']
            )
            self.validators.append(schema_validator)

        # Business rules validator
        if 'business_rules' in self.config:
            business_validator = BusinessRulesValidator(self.config['business_rules'])
            self.validators.append(business_validator)

        # Quality validator
        quality_validator = QualityValidator(self.config.get('quality', {}))
        self.validators.append(quality_validator)

    def validate(self, data: pd.DataFrame) -> ValidationReport:
        """Run comprehensive validation"""
        logger.info("Starting comprehensive data validation")

        combined_report = ValidationReport()

        for validator in self.validators:
            logger.info(f"Running {validator.__class__.__name__}")
            report = validator.validate(data)

            # Combine results
            combined_report.passed += report.passed
            combined_report.failed += report.failed
            combined_report.warnings += report.warnings
            combined_report.results.extend(report.results)

        logger.info(f"Validation complete: {combined_report.success_rate:.1%} success rate")
        return combined_report

def create_transaction_validator(config: Optional[Dict[str, Any]] = None) -> ComprehensiveValidator:
    """Create validator for transaction data"""
    default_config = {
        'schema': {
            'expected_schema': {
                'date': 'datetime',
                'store_id': 'int',
                'product_id': 'int',
                'quantity': 'int',
                'unit_price': 'float',
                'total_sales': 'float'
            },
            'required_columns': ['date', 'store_id', 'product_id']
        },
        'business_rules': {
            'non_negative_columns': ['quantity', 'unit_price', 'total_sales'],
            'valid_ranges': {
                'quantity': {'min': 0, 'max': 10000},
                'unit_price': {'min': 0, 'max': 100000}
            }
        },
        'quality': {
            'max_null_percentage': 0.05,
            'max_duplicate_percentage': 0.01
        }
    }

    if config:
        # Merge with user config
        for key, value in config.items():
            if key in default_config:
                default_config[key].update(value)
            else:
                default_config[key] = value

    return ComprehensiveValidator(default_config)

if __name__ == "__main__":
    # Demo usage
    print("✅ Data Validators Demo")
    print("=" * 50)

    # Create transaction validator
    validator = create_transaction_validator()
    print("✅ Created transaction validator")

    print("\n🔍 Validator components:")
    for i, val in enumerate(validator.validators):
        print(f"  {i+1}. {val.__class__.__name__}")

    print("\n🏭 Validation system ready!")
    print("Ready to validate transaction, product, and store data with comprehensive checks.")