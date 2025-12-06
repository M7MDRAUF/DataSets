# Contributing to CineMatch

Thank you for your interest in contributing to CineMatch! This document provides guidelines and best practices for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Commit Conventions](#commit-conventions)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)

---

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. We expect all contributors to:

- Be respectful and considerate
- Focus on constructive feedback
- Accept responsibility for mistakes
- Prioritize the community's best interests

---

## Getting Started

### Prerequisites

- Python 3.11+
- Git
- Docker (optional, for containerized development)
- 8GB+ RAM (recommended for ML models)

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/CineMatch.git
cd CineMatch
```

---

## Development Setup

### 1. Create Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

### 3. Download Data

```bash
# Download MovieLens 32M dataset
# Place in data/ml-32m/ directory
```

### 4. Train Models (Optional)

```bash
python train_all_models.py
```

### 5. Run Application

```bash
streamlit run app/main.py
```

---

## Code Style

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with the following specifications:

#### General Rules

- **Line length**: Maximum 100 characters
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Double quotes for strings, single quotes for dict keys
- **Trailing commas**: Required for multi-line collections

#### Import Ordering

Imports should be organized in the following order:

```python
# 1. Standard library imports
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# 2. Third-party imports
import numpy as np
import pandas as pd
import streamlit as st

# 3. Local application imports
from src.data_processing import load_data
from src.recommendation_engine import RecommendationEngine
```

#### Type Hints

All public functions must include type hints:

```python
def recommend_movies(
    user_id: int,
    n_recommendations: int = 10,
    exclude_watched: bool = True
) -> pd.DataFrame:
    """
    Generate movie recommendations for a user.
    
    Args:
        user_id: The ID of the user to recommend movies for.
        n_recommendations: Number of recommendations to return.
        exclude_watched: Whether to exclude already-watched movies.
    
    Returns:
        DataFrame containing movie recommendations with columns:
        movieId, title, predicted_rating, genres.
    
    Raises:
        ValueError: If user_id is invalid.
        ModelNotTrainedError: If recommendation model is not trained.
    """
    pass
```

#### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Variables | snake_case | `user_ratings` |
| Functions | snake_case | `calculate_similarity()` |
| Classes | PascalCase | `RecommendationEngine` |
| Constants | UPPER_SNAKE_CASE | `MAX_RECOMMENDATIONS` |
| Private | Leading underscore | `_internal_method()` |
| Protected | Leading underscore | `_protected_attr` |

#### Docstrings

Use Google-style docstrings for all public modules, classes, and functions:

```python
"""
Module description.

This module provides functionality for...

Example:
    >>> from module import function
    >>> result = function(arg)

Attributes:
    MODULE_CONSTANT: Description of the constant.
"""
```

### Linting and Formatting

Before submitting, ensure your code passes:

```bash
# Run linter
flake8 src/ tests/

# Check type hints
mypy src/

# Format code
black src/ tests/
isort src/ tests/
```

---

## Testing Requirements

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_recommendation_engine.py

# Run tests by marker
pytest -m "not slow"  # Skip slow tests
pytest -m unit        # Only unit tests
pytest -m integration # Only integration tests
```

### Test Categories

| Marker | Description | Expected Duration |
|--------|-------------|-------------------|
| `unit` | Isolated unit tests | < 1 second |
| `integration` | Tests with dependencies | < 10 seconds |
| `slow` | Long-running tests | > 10 seconds |
| `e2e` | End-to-end tests | Variable |

### Coverage Requirements

- **Minimum coverage**: 80% for new code
- **Critical paths**: 100% coverage required for:
  - Algorithm implementations
  - API endpoints
  - Data processing functions

### Writing Tests

```python
import pytest
from src.algorithms.svd import SVDAlgorithm


class TestSVDAlgorithm:
    """Tests for SVD recommendation algorithm."""
    
    @pytest.fixture
    def algorithm(self, sample_ratings_df):
        """Create trained algorithm instance."""
        algo = SVDAlgorithm()
        algo.train(sample_ratings_df)
        return algo
    
    def test_recommend_returns_expected_count(self, algorithm):
        """Verify recommend returns correct number of items."""
        recommendations = algorithm.recommend(user_id=1, n=5)
        assert len(recommendations) == 5
    
    def test_recommend_excludes_watched_movies(self, algorithm, sample_ratings_df):
        """Verify watched movies are excluded by default."""
        user_watched = sample_ratings_df[
            sample_ratings_df['userId'] == 1
        ]['movieId'].tolist()
        
        recommendations = algorithm.recommend(user_id=1, n=5)
        
        for movie_id in recommendations['movieId']:
            assert movie_id not in user_watched
    
    @pytest.mark.parametrize("n", [1, 5, 10, 20])
    def test_recommend_various_counts(self, algorithm, n):
        """Test recommendation with various count parameters."""
        recommendations = algorithm.recommend(user_id=1, n=n)
        assert len(recommendations) <= n
```

---

## Pull Request Process

### Before Creating a PR

1. ✅ Ensure all tests pass locally
2. ✅ Update documentation if needed
3. ✅ Add tests for new functionality
4. ✅ Run linting and formatting
5. ✅ Rebase on latest `main`

### PR Template

When creating a PR, use this template:

```markdown
## Description
Brief description of the changes.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings generated
```

### Review Process

1. **Create PR** targeting `main` branch
2. **Automated checks** run (tests, linting)
3. **Code review** by maintainer
4. **Address feedback** through additional commits
5. **Squash and merge** after approval

### PR Size Guidelines

- **Small PRs preferred**: < 500 lines of changes
- **Split large features** into smaller, reviewable chunks
- **One concern per PR**: Don't mix refactoring with features

---

## Commit Conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/):

### Format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation changes |
| `style` | Formatting, missing semicolons, etc. |
| `refactor` | Code restructuring without behavior change |
| `test` | Adding or updating tests |
| `chore` | Maintenance tasks |
| `perf` | Performance improvements |

### Examples

```bash
# Feature
feat(algorithms): add hybrid recommendation algorithm

# Bug fix
fix(api): handle missing user ID in recommendations endpoint

# Documentation
docs(readme): update installation instructions

# Refactor
refactor(engine): extract similarity calculation to separate module

# Test
test(svd): add edge case tests for sparse matrices

# Breaking change
feat(api)!: change recommendation response format

BREAKING CHANGE: The `recommendations` field is now an array of objects
instead of an array of movie IDs.
```

### Commit Best Practices

- **Atomic commits**: Each commit should be a single logical change
- **Present tense**: "Add feature" not "Added feature"
- **Imperative mood**: "Move cursor to..." not "Moves cursor to..."
- **No period** at the end of the subject line
- **50/72 rule**: Subject ≤ 50 chars, body wrapped at 72 chars

---

## Documentation

### When to Update Documentation

- Adding new features
- Changing API endpoints
- Modifying configuration options
- Updating dependencies
- Fixing bugs that affect user behavior

### Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | Project overview, quick start |
| `CONTRIBUTING.md` | Contribution guidelines (this file) |
| `doc/API_DOCUMENTATION.md` | API reference |
| `doc/ARCHITECTURE.md` | System architecture |
| `doc/DEPLOYMENT.md` | Deployment instructions |
| `doc/TROUBLESHOOTING.md` | Common issues and solutions |

### Docstring Requirements

All public functions must have docstrings with:

- Brief description
- Args (with types)
- Returns (with type)
- Raises (if applicable)
- Example (for complex functions)

---

## Issue Reporting

### Bug Reports

Use the bug report template:

```markdown
## Bug Description
Clear description of the bug.

## Steps to Reproduce
1. Go to '...'
2. Click on '...'
3. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- OS: [e.g., Windows 11]
- Python version: [e.g., 3.11.4]
- CineMatch version: [e.g., 2.1.6]

## Additional Context
Any other relevant information.
```

### Feature Requests

```markdown
## Feature Description
Clear description of the proposed feature.

## Use Case
Why is this feature needed? What problem does it solve?

## Proposed Solution
How do you envision this feature working?

## Alternatives Considered
Other solutions you've considered.
```

---

## Questions?

If you have questions about contributing:

1. Check existing [documentation](./doc/)
2. Search existing [issues](../../issues)
3. Open a new issue with the `question` label

Thank you for contributing to CineMatch! 🎬
