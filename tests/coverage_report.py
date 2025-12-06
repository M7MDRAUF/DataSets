"""
CineMatch V2.1.6 - Coverage Configuration and Report Generator

Configuration and utilities for test coverage reporting.
Task 3.15: Generate coverage report.

Author: CineMatch Development Team
Date: December 5, 2025
"""

import subprocess
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


# =============================================================================
# COVERAGE CONFIGURATION
# =============================================================================

COVERAGE_CONFIG = """
# .coveragerc - Coverage configuration for CineMatch
# Run: pytest --cov=src --cov-report=html --cov-report=xml

[run]
source = src
branch = True
parallel = True
omit = 
    */tests/*
    */__pycache__/*
    */venv/*
    */.venv/*
    setup.py
    conftest.py

[report]
# Regexes for lines to exclude from consideration
exclude_lines =
    # Have to re-enable the standard pragma
    pragma: no cover
    
    # Don't complain about missing debug-only code:
    def __repr__
    def __str__
    
    # Don't complain if tests don't hit defensive assertion code:
    raise AssertionError
    raise NotImplementedError
    
    # Don't complain if non-runnable code isn't run:
    if __name__ == .__main__.:
    if TYPE_CHECKING:
    
    # Don't complain about abstract methods
    @abstractmethod

ignore_errors = True
skip_empty = True

# Fail if coverage is below threshold
fail_under = 70

precision = 2
show_missing = True

[html]
directory = coverage_html
title = CineMatch Test Coverage Report

[xml]
output = coverage.xml

[json]
output = coverage.json
pretty_print = True
"""


# =============================================================================
# COVERAGE DATA STRUCTURES
# =============================================================================

@dataclass
class FileCoverage:
    """Coverage data for a single file"""
    filename: str
    statements: int = 0
    missing: int = 0
    excluded: int = 0
    branches: int = 0
    partial_branches: int = 0
    coverage_percent: float = 0.0
    missing_lines: List[int] = field(default_factory=list)


@dataclass  
class CoverageReport:
    """Complete coverage report"""
    timestamp: str
    total_statements: int = 0
    total_missing: int = 0
    total_branches: int = 0
    total_partial_branches: int = 0
    coverage_percent: float = 0.0
    files: List[FileCoverage] = field(default_factory=list)
    uncovered_modules: List[str] = field(default_factory=list)


# =============================================================================
# COVERAGE UTILITIES
# =============================================================================

class CoverageAnalyzer:
    """Analyze and report test coverage"""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.src_dir = self.project_root / "src"
        self.tests_dir = self.project_root / "tests"
    
    def run_coverage(self, output_format: str = "all") -> bool:
        """Run pytest with coverage"""
        cmd = [
            sys.executable, "-m", "pytest",
            "--cov=src",
            "--cov-branch",
            f"--cov-report={output_format}",
            str(self.tests_dir)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except Exception as e:
            print(f"Error running coverage: {e}")
            return False
    
    def get_source_files(self) -> List[Path]:
        """Get all Python source files"""
        return list(self.src_dir.rglob("*.py"))
    
    def get_test_files(self) -> List[Path]:
        """Get all test files"""
        return list(self.tests_dir.glob("test_*.py"))
    
    def analyze_coverage_json(self, json_path: Path) -> Optional[CoverageReport]:
        """Analyze coverage from JSON report"""
        if not json_path.exists():
            return None
        
        with open(json_path) as f:
            data = json.load(f)
        
        report = CoverageReport(
            timestamp=datetime.now().isoformat(),
            total_statements=data.get("totals", {}).get("num_statements", 0),
            total_missing=data.get("totals", {}).get("missing_lines", 0),
            total_branches=data.get("totals", {}).get("num_branches", 0),
            total_partial_branches=data.get("totals", {}).get("num_partial_branches", 0),
            coverage_percent=data.get("totals", {}).get("percent_covered", 0.0)
        )
        
        for filename, file_data in data.get("files", {}).items():
            file_cov = FileCoverage(
                filename=filename,
                statements=file_data.get("summary", {}).get("num_statements", 0),
                missing=file_data.get("summary", {}).get("missing_lines", 0),
                excluded=file_data.get("summary", {}).get("excluded_lines", 0),
                branches=file_data.get("summary", {}).get("num_branches", 0),
                partial_branches=file_data.get("summary", {}).get("num_partial_branches", 0),
                coverage_percent=file_data.get("summary", {}).get("percent_covered", 0.0),
                missing_lines=file_data.get("missing_lines", [])
            )
            report.files.append(file_cov)
        
        return report
    
    def identify_uncovered_modules(self) -> List[str]:
        """Identify modules without any tests"""
        source_files = self.get_source_files()
        test_files = self.get_test_files()
        
        # Extract module names from test files
        tested_modules = set()
        for test_file in test_files:
            # test_foo.py -> foo
            name = test_file.stem.replace("test_", "")
            tested_modules.add(name)
        
        # Check which source modules have no tests
        uncovered = []
        for source_file in source_files:
            if source_file.name.startswith("_"):
                continue
            
            module_name = source_file.stem
            if module_name not in tested_modules:
                uncovered.append(str(source_file.relative_to(self.project_root)))
        
        return uncovered
    
    def generate_markdown_report(self, report: CoverageReport) -> str:
        """Generate Markdown coverage report"""
        lines = [
            "# CineMatch Test Coverage Report",
            "",
            f"**Generated:** {report.timestamp}",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Statements | {report.total_statements} |",
            f"| Covered Statements | {report.total_statements - report.total_missing} |",
            f"| Missing Statements | {report.total_missing} |",
            f"| Coverage | **{report.coverage_percent:.1f}%** |",
            f"| Branches | {report.total_branches} |",
            f"| Partial Branches | {report.total_partial_branches} |",
            "",
            "## File Coverage",
            "",
            "| File | Statements | Missing | Coverage |",
            "|------|------------|---------|----------|"
        ]
        
        # Sort by coverage (lowest first)
        sorted_files = sorted(report.files, key=lambda f: f.coverage_percent)
        
        for file_cov in sorted_files:
            status = "✅" if file_cov.coverage_percent >= 80 else "⚠️" if file_cov.coverage_percent >= 60 else "❌"
            lines.append(
                f"| {status} {file_cov.filename} | {file_cov.statements} | "
                f"{file_cov.missing} | {file_cov.coverage_percent:.1f}% |"
            )
        
        # Add uncovered modules section
        if report.uncovered_modules:
            lines.extend([
                "",
                "## Uncovered Modules",
                "",
                "The following modules have no dedicated tests:",
                ""
            ])
            for module in report.uncovered_modules:
                lines.append(f"- `{module}`")
        
        # Add recommendations
        lines.extend([
            "",
            "## Recommendations",
            "",
            "### High Priority (< 60% coverage)",
            ""
        ])
        
        low_coverage = [f for f in report.files if f.coverage_percent < 60]
        if low_coverage:
            for f in low_coverage[:5]:
                lines.append(f"- **{f.filename}** ({f.coverage_percent:.1f}%)")
                if f.missing_lines:
                    lines.append(f"  - Missing lines: {f.missing_lines[:10]}...")
        else:
            lines.append("No files below 60% coverage! 🎉")
        
        return "\n".join(lines)


# =============================================================================
# PYTEST COVERAGE MARKERS
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers for coverage"""
    config.addinivalue_line(
        "markers", "coverage: mark test as coverage-related"
    )


# =============================================================================
# COVERAGE SCRIPT
# =============================================================================

def main():
    """Main coverage script"""
    print("="*60)
    print("CineMatch Coverage Analysis")
    print("="*60)
    
    analyzer = CoverageAnalyzer()
    
    # Get source file count
    source_files = analyzer.get_source_files()
    test_files = analyzer.get_test_files()
    
    print(f"\n📁 Source Files: {len(source_files)}")
    print(f"🧪 Test Files: {len(test_files)}")
    
    # Identify uncovered modules
    uncovered = analyzer.identify_uncovered_modules()
    if uncovered:
        print(f"\n⚠️  Modules without dedicated tests:")
        for module in uncovered[:10]:
            print(f"   - {module}")
        if len(uncovered) > 10:
            print(f"   ... and {len(uncovered) - 10} more")
    
    print("\n" + "="*60)
    print("To generate full coverage report, run:")
    print("  pytest --cov=src --cov-report=html --cov-report=json")
    print("="*60)
    
    # Write .coveragerc if it doesn't exist
    coveragerc_path = Path(__file__).parent.parent / ".coveragerc"
    if not coveragerc_path.exists():
        with open(coveragerc_path, 'w') as f:
            f.write(COVERAGE_CONFIG)
        print(f"\n✅ Created {coveragerc_path}")


if __name__ == "__main__":
    main()
