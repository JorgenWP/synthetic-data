# Contributing to Synthetic Graph Data Generation

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/synthetic-data.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Set up your development environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements-dev.txt
   ```

## Development Workflow

### Code Style

We follow PEP 8 guidelines with some modifications:
- Line length: 100 characters
- Use Black for formatting
- Use isort for import sorting

Format your code before committing:
```bash
black src tests
isort src tests
```

### Testing

- Write tests for all new functionality
- Ensure all tests pass before submitting PR
- Aim for high test coverage (>80%)

Run tests:
```bash
pytest tests/ -v
```

Check coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

### Documentation

- Add docstrings to all functions, classes, and modules
- Use Google-style docstrings
- Update README.md if adding new features
- Add examples for new functionality

Example docstring:
```python
def my_function(param1: str, param2: int) -> bool:
    """
    Brief description of the function.
    
    More detailed description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param2 is negative
    """
    pass
```

### Commit Messages

- Use clear and descriptive commit messages
- Start with a verb in present tense (Add, Fix, Update, Remove)
- Reference issue numbers when applicable

Examples:
```
Add graph attention network model
Fix bug in data preprocessing
Update documentation for training script
Remove deprecated visualization function
```

## Pull Request Process

1. Update documentation with details of changes
2. Add or update tests as needed
3. Ensure all tests pass and code is formatted
4. Update the CHANGELOG.md (if exists)
5. Create a pull request with a clear description of changes

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe the tests you ran

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Commented on complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
```

## Code Review Process

- At least one maintainer review is required
- Address all review comments
- Keep discussions professional and constructive
- Be patient and respectful

## Reporting Bugs

Use GitHub Issues to report bugs. Include:
- Clear and descriptive title
- Steps to reproduce
- Expected behavior
- Actual behavior
- System information (OS, Python version, etc.)
- Code samples or error messages

## Suggesting Features

Feature requests are welcome! Open an issue with:
- Clear and descriptive title
- Detailed description of the proposed feature
- Use cases and benefits
- Possible implementation approach (optional)

## Questions?

If you have questions, feel free to:
- Open an issue for discussion
- Reach out to maintainers

Thank you for contributing! 🙏
