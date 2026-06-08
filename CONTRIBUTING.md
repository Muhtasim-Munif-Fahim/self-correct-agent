# Contributing to Self-Correct Agent

Thank you for your interest in contributing! This project aims to make LLM outputs more reliable by reducing hallucinations.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/self-correct-agent.git`
3. Install in development mode:
   ```bash
   pip install -e ".[dev,search]"
   ```
4. Run the tests:
   ```bash
   python -m pytest -q
   ```
5. Run the demo script:
   ```bash
   python examples/demo.py
   ```

## Development Workflow

1. Create a branch: `git checkout -b fix/your-fix-description`
2. Make changes
3. Add tests for your changes
4. Ensure all tests pass: `python -m pytest -q`
5. Submit a Pull Request

## Code Style

- Type hints are required on all public functions
- Docstrings follow NumPy/SciPy format
- Keep functions under 50 lines
- Keep files under 300 lines

## Adding a New Verification Tool

1. Create a class that inherits from `Tool` in `tools.py`
2. Implement the `name` property and `search()` method
3. Add tests in `tests/test_tools.py`
4. Update `__init__.py` exports
5. Add a short example or notebook if the tool changes the public workflow

## Reporting Issues

When reporting bugs, please include:
- Python version
- `self-correct` version
- Minimal reproduction code
- Full traceback
