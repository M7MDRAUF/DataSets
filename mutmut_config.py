"""
CineMatch V2.1.6 - Mutation Testing Configuration

Configuration for mutmut mutation testing to verify test quality.
Task 3.6: Mutation testing setup

Author: CineMatch Development Team
Date: December 2025
"""


def pre_mutation(context):
    """
    Pre-mutation hook to skip certain files or mutations.
    
    Return True to skip a mutation, False to allow it.
    """
    # Skip test files themselves
    if context.filename.startswith("tests/"):
        return True
    
    # Skip __pycache__ directories
    if "__pycache__" in context.filename:
        return True
    
    # Skip migration files
    if "migration" in context.filename.lower():
        return True
    
    # Skip generated files
    if context.filename.endswith("_generated.py"):
        return True
    
    # Skip specific files that are configuration-only
    skip_files = [
        "setup.py",
        "conftest.py",
        "__init__.py",
    ]
    
    for skip_file in skip_files:
        if context.filename.endswith(skip_file):
            return True
    
    return False


def pre_mutation_ast(context):
    """
    Pre-mutation AST hook for fine-grained control.
    
    Return True to skip the mutation.
    """
    # Skip logging statements - mutating these doesn't test logic
    if context.current_source_line:
        line = context.current_source_line.strip()
        
        # Skip logger calls
        if line.startswith("logger.") or line.startswith("logging."):
            return True
        
        # Skip comments
        if line.startswith("#"):
            return True
        
        # Skip docstrings
        if line.startswith('"""') or line.startswith("'''"):
            return True
        
        # Skip print statements (should be removed anyway)
        if line.startswith("print("):
            return True
        
        # Skip import statements
        if line.startswith("import ") or line.startswith("from "):
            return True
        
        # Skip type hints only lines
        if " -> " in line and "def " not in line:
            return True
    
    return False


# Mutmut configuration dictionary
# This can be used with mutmut run --use-config mutmut_config.py

MUTMUT_CONFIG = {
    # Paths to mutate
    "paths_to_mutate": [
        "src/algorithms/",
        "src/data_processing.py",
        "src/recommendation_engine.py",
        "src/search_engine.py",
        "src/utils.py",
    ],
    
    # Test runner configuration
    "runner": "pytest",
    "tests_dir": "tests/",
    
    # Timeout per test in seconds
    "test_timeout": 60,
    
    # Number of parallel workers
    "parallel": True,
    "n_jobs": 4,
    
    # Mutation operators to enable
    "operators": [
        "AOD",  # Arithmetic operator deletion
        "AOR",  # Arithmetic operator replacement
        "COD",  # Conditional operator deletion
        "COI",  # Conditional operator insertion
        "CRP",  # Constant replacement
        "DDL",  # Decorator deletion
        "EHD",  # Exception handler deletion
        "EXS",  # Exception swallowing
        "IHD",  # Hiding variable deletion
        "IOD",  # Overriding method deletion
        "IOP",  # Overridden method calling position
        "LCR",  # Logical connector replacement
        "LOD",  # Logical operator deletion
        "LOR",  # Logical operator replacement
        "ROR",  # Relational operator replacement
        "SCD",  # Super calling deletion
        "SCI",  # Super calling insertion
        "SIR",  # Slice index removal
    ],
    
    # Directories to exclude
    "exclude_dirs": [
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "build",
        "dist",
        ".eggs",
        "*.egg-info",
    ],
    
    # Files to exclude
    "exclude_files": [
        "setup.py",
        "conftest.py",
        "*_test.py",
        "test_*.py",
    ],
}


# Quick commands reference
COMMANDS = """
# Mutation Testing Commands for CineMatch

# Run full mutation testing (can take a long time)
mutmut run --paths-to-mutate src/algorithms/ --tests-dir tests/

# Run with specific test file
mutmut run --paths-to-mutate src/algorithms/svd_recommender.py --tests-dir tests/test_algorithms.py

# Run with timeout
mutmut run --paths-to-mutate src/ --runner "pytest -x --timeout=30"

# Show results
mutmut results

# Show surviving mutants (potential test gaps)
mutmut show <mutant_id>

# Generate HTML report
mutmut html

# Run specific mutation
mutmut run --mutant <mutant_id>

# Apply a mutant to see the change
mutmut apply <mutant_id>

# Reset to original
mutmut revert

# Calculate mutation score
# Mutation Score = (Killed Mutants / Total Mutants) * 100
# Target: > 70% mutation score
"""


if __name__ == "__main__":
    print("Mutation Testing Configuration for CineMatch V2.1.6")
    print("=" * 50)
    print("\nConfiguration:")
    for key, value in MUTMUT_CONFIG.items():
        print(f"  {key}: {value}")
    print("\n" + COMMANDS)
