# Contributing to PerfectFrameAI (Enhanced)

Thank you for your interest in contributing to PerfectFrameAI Enhanced! This document provides guidelines and instructions for contributing to this project.

## Development Setup

### Prerequisites
- Python 3.10 or higher
- Docker
- Git
- NVIDIA GPU with CUDA support (optional, but recommended)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/esN2k/PerfectFrameAI.git
cd PerfectFrameAI
```

2. **Install dependencies using Poetry:**
```bash
pip install poetry
poetry install
```

Alternatively, using pip:
```bash
pip install -r extractor_service/requirements.txt
```

3. **Run tests to verify setup:**
```bash
pytest tests/
```

## Running Tests

We use pytest for testing. Run the full test suite:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_person_mode_e2e.py -v
```

## Code Style

Please follow these guidelines when contributing:

### Python Style Guide
- Follow **PEP 8** conventions
- Add **type hints** to all function signatures
- Write **docstrings** using Google style format
- Keep line length to **100 characters** maximum
- Use meaningful variable and function names

### Example:
```python
def score_frame(frame: np.ndarray, config: dict) -> float:
    """Calculate composite quality score for a video frame.
    
    Args:
        frame: Input frame as numpy array (RGB format).
        config: Configuration dictionary with scoring weights.
        
    Returns:
        Composite score between 0-10.
    """
    # Implementation here
    pass
```

### Code Quality Tools

Before submitting a pull request, run:

```bash
# Format code with black
black --line-length 100 .

# Check code style
flake8 src/ --max-line-length=100

# Type checking (if mypy is configured)
mypy src/
```

## Adding New Scoring Models

To add a new quality assessment model:

1. **Create a new module** in `extractor_service/app/modules/`:
```python
# your_model.py
import numpy as np

class YourModel:
    def __init__(self, config: dict):
        """Initialize your model."""
        self.config = config
        
    def score_frame(self, frame: np.ndarray) -> float:
        """Score a frame.
        
        Args:
            frame: Input frame as numpy array.
            
        Returns:
            Score between 0-10.
        """
        # Your scoring logic
        return score
```

2. **Integrate into composite scoring** in the appropriate evaluator module.

3. **Add configuration** to `config.yaml`:
```yaml
scoring_weights:
  aesthetic: 0.4
  face_quality: 0.3
  sharpness: 0.15
  your_model: 0.15  # Add your model weight
```

4. **Write tests** for your model:
```python
# tests/test_your_model.py
def test_your_model_scoring():
    model = YourModel(config={})
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    score = model.score_frame(frame)
    assert 0 <= score <= 10
```

5. **Update documentation** in README.md to mention the new scoring feature.

## Submitting Pull Requests

1. **Fork the repository** and create a new branch:
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes** following the code style guidelines.

3. **Write or update tests** to cover your changes.

4. **Run the test suite** to ensure everything passes:
```bash
pytest tests/ -v
```

5. **Commit your changes** with descriptive commit messages:
```bash
git commit -m "Add feature: description of your changes"
```

6. **Push to your fork:**
```bash
git push origin feature/your-feature-name
```

7. **Open a Pull Request** on GitHub with:
   - Clear description of changes
   - Reference to any related issues
   - Screenshots (if applicable)
   - Test results

## Pull Request Guidelines

- **Keep PRs focused** - one feature or fix per PR
- **Write clear descriptions** - explain what and why, not just how
- **Maintain test coverage** - aim for >70% coverage
- **Update documentation** - if you change behavior, update README.md
- **Follow existing patterns** - stay consistent with the codebase

## Reporting Bugs

When reporting bugs, please include:

1. **Description** of the bug
2. **Steps to reproduce** the issue
3. **Expected behavior** vs actual behavior
4. **Environment details**:
   - OS and version
   - Python version
   - GPU model (if applicable)
   - Docker version
5. **Error messages or logs**
6. **Sample data** if possible (or description)

## Feature Requests

We welcome feature requests! Please:

1. **Check existing issues** to avoid duplicates
2. **Describe the feature** clearly
3. **Explain the use case** and benefits
4. **Provide examples** if applicable

## Code Review Process

1. Maintainers will review your PR within a few days
2. Address any requested changes
3. Once approved, your PR will be merged
4. Your contribution will be acknowledged in the next release

## Questions?

If you have questions about contributing:
- Open an issue with the `question` label
- Check existing documentation
- Review closed issues for similar questions

## License

By contributing, you agree that your contributions will be licensed under the GNU General Public License v3.0.

---

Thank you for contributing to PerfectFrameAI Enhanced! 🎉
