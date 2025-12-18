# Development Guidelines for WorkZone

## Code Style and Standards

This project follows PEP 8 and uses automated tools to ensure consistency.

### Tools

- **black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing

### Running Quality Checks

```bash
# Format all code
make format

# Or manually:
black src/ tests/ --line-length 88
isort src/ tests/

# Lint
flake8 src/ tests/

# Type check
mypy src/

# Run tests
pytest tests/ -v
```

## Project Structure Guidelines

### Module Organization

- **models/**: AI model wrappers (YOLO, VLM, CLIP)
- **pipelines/**: Complete processing workflows (training, inference)
- **cli/**: Command-line interfaces
- **utils/**: Shared utilities and helpers
- **config.py**: Configuration management

### Naming Conventions

- Classes: `PascalCase` (e.g., `YOLODetector`)
- Functions: `snake_case` (e.g., `process_frame`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_BATCH_SIZE`)
- Private: Leading underscore (e.g., `_internal_method`)

### Documentation

All public functions and classes must have:

```python
def my_function(param1: str, param2: int = 10) -> dict:
    """
    Brief description of what function does.

    Longer description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2 (default: 10)

    Returns:
        Description of return value
        
    Raises:
        ValueError: When validation fails
        
    Examples:
        >>> result = my_function("test", 20)
        >>> print(result)
    """
```

## Type Hints

All new code must include type hints:

```python
from typing import Optional, List, Dict, Tuple

def process_data(
    images: List[np.ndarray],
    model: str,
    confidence: float = 0.5,
) -> Dict[str, List]:
    """Process multiple images."""
    pass
```

## Adding New Features

1. **Create feature branch**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Write tests first** (TDD approach)
   ```bash
   # Add tests in tests/test_*.py
   pytest tests/test_my_feature.py -v
   ```

3. **Implement feature**
   - Follow PEP 8
   - Add type hints
   - Add docstrings

4. **Run quality checks**
   ```bash
   black src/ tests/
   isort src/ tests/
   flake8 src/ tests/
   mypy src/
   pytest tests/ -v
   ```

5. **Commit and push**
   ```bash
   git commit -m "feat: add my feature"
   git push origin feature/my-feature
   ```

6. **Open pull request**

## Configuration

### Adding New Configuration Options

1. Update `config.py` dataclass:
   ```python
   @dataclass
   class MyNewConfig:
       param1: str = "default"
       param2: int = 42
   ```

2. Add to `ProjectConfig`:
   ```python
   @dataclass
   class ProjectConfig:
       my_new_config: MyNewConfig = field(default_factory=MyNewConfig)
   ```

3. Update `configs/config.yaml`:
   ```yaml
   my_new_config:
     param1: "default"
     param2: 42
   ```

4. Add environment variable support in `from_env()`:
   ```python
   @classmethod
   def from_env(cls) -> "MyNewConfig":
       return cls(
           param1=os.getenv("MY_PARAM1", "default"),
           param2=int(os.getenv("MY_PARAM2", "42")),
       )
   ```

## Performance Profiling

### Profile inference speed

```python
import time
from src.workzone.models.yolo_detector import YOLODetector

detector = YOLODetector("weights/best.pt")

# Warm up
detector.detect(frame)

# Time inference
start = time.time()
for _ in range(100):
    detector.detect(frame)
elapsed = time.time() - start

print(f"FPS: {100 / elapsed}")
```

### Memory profiling

```bash
pip install memory_profiler

python -m memory_profiler script.py
```

## Debugging

### Enable debug logging

```python
from src.workzone.utils.logging_config import setup_logger

logger = setup_logger(__name__, verbose=True)
logger.debug("Debug message")
```

### Use Python debugger

```python
import pdb

pdb.set_trace()  # Breakpoint
```

## CI/CD Integration

### GitHub Actions

Example workflow file `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    
    - name: Run tests
      run: pytest tests/ -v
```

## Common Issues

### Import Errors

If you get import errors, ensure:
1. You've installed in editable mode: `pip install -e .`
2. `src` is in Python path
3. `__init__.py` files exist in all packages

### CUDA Out of Memory

Reduce batch size or image size:
```bash
python -m src.workzone.cli.train_yolo --batch 16 --imgsz 640
```

### W&B Login Required

```bash
wandb login
# Paste API key from https://wandb.ai/settings/keys
```

## Resources

- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Type Hints Guide](https://docs.python.org/3/library/typing.html)
- [Pytest Documentation](https://docs.pytest.org/)
- [Black Code Formatter](https://black.readthedocs.io/)

---

For questions or issues, please open a GitHub issue or contact the maintainers.
