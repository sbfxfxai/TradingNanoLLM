# Contributing to TradingNanoLLM

Thank you for your interest in contributing to TradingNanoLLM! This project aims to create a lightweight, efficient LLM specifically optimized for trading and financial analysis tasks.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- CUDA-compatible GPU (recommended for training and inference)

### Development Setup
1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/TradingNanoLLM.git
   cd TradingNanoLLM
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -e .[dev]
   ```

## 🛠️ Development Workflow

### Branching Strategy
- `main`: Stable release branch
- `develop`: Integration branch for features
- `feature/your-feature-name`: Feature development branches
- `fix/issue-description`: Bug fix branches

### Making Changes
1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Add tests for new functionality
4. Run tests and ensure they pass:
   ```bash
   pytest tests/
   ```
5. Format code:
   ```bash
   black trading_nanovllm/
   flake8 trading_nanovllm/
   ```
6. Commit with descriptive messages:
   ```bash
   git commit -m "Add prefix caching optimization for faster inference"
   ```
7. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
8. Create a Pull Request

## 📋 Areas for Contribution

### High Priority
- **Performance Optimizations**
  - CUDA graph implementation
  - Tensor parallelism support
  - Memory-efficient attention mechanisms
  - Quantization techniques (INT8, INT4)

- **Trading Utilities**
  - Advanced technical indicators
  - Portfolio optimization algorithms
  - Risk management tools
  - Real-time data integration

- **Model Improvements**
  - Fine-tuning on larger financial datasets
  - Multi-modal support (charts, news, data)
  - Specialized financial embeddings

### Medium Priority
- **Infrastructure**
  - Docker containerization
  - CI/CD pipeline improvements
  - Distributed training support
  - Model versioning and registry

- **Documentation**
  - API documentation
  - Trading strategy examples
  - Performance benchmarking guides
  - Deployment tutorials

### Low Priority
- **Integrations**
  - Trading platform APIs
  - Financial data providers
  - Visualization tools
  - Backtesting frameworks

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_llm.py

# Run with coverage
pytest --cov=trading_nanovllm
```

### Test Categories
- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **Performance Tests**: Benchmark inference speed
- **End-to-End Tests**: Full workflow testing

### Writing Tests
- Place tests in the `tests/` directory
- Use descriptive test names: `test_prefix_cache_improves_throughput`
- Include edge cases and error conditions
- Mock external dependencies (APIs, file systems)

## 📝 Code Style

### Python Style Guide
- Follow PEP 8 guidelines
- Use type hints for all functions
- Maximum line length: 88 characters
- Use descriptive variable and function names

### Code Formatting
We use `black` for code formatting and `flake8` for linting:
```bash
# Format code
black trading_nanovllm/ tests/ examples/

# Check linting
flake8 trading_nanovllm/ tests/
```

### Documentation
- Use Google-style docstrings
- Include type hints in function signatures
- Add examples for complex functions
- Update README.md for user-facing changes

Example docstring:
```python
def generate_trade_signal(
    market_data: Dict[str, Any],
    risk_tolerance: str = "moderate"
) -> Dict[str, Union[str, float]]:
    """Generate trading signal based on market data.
    
    Args:
        market_data: Dictionary containing price, volume, indicators
        risk_tolerance: Risk level (conservative/moderate/aggressive)
        
    Returns:
        Dictionary with signal, confidence, and reasoning
        
    Example:
        >>> data = {"price": 100, "rsi": 70, "volume": 1000000}
        >>> signal = generate_trade_signal(data, "conservative")
        >>> print(signal["signal"])  # "Hold"
    """
```

## 🐛 Bug Reports

### Before Reporting
- Check existing issues for duplicates
- Ensure you're using the latest version
- Test with minimal reproduction case

### Bug Report Template
```markdown
## Bug Description
Brief description of the issue

## Steps to Reproduce
1. Load model with: `TradingLLM("model-name")`
2. Run inference with: `llm.generate(prompts, params)`
3. Observe error/unexpected behavior

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- TradingNanoLLM version:
- Python version:
- OS:
- GPU (if applicable):
- CUDA version:

## Additional Context
- Error messages
- Logs
- Screenshots
```

## 💡 Feature Requests

### Feature Request Template
```markdown
## Feature Description
Clear description of the proposed feature

## Use Case
Why is this feature needed? What problem does it solve?

## Proposed Implementation
High-level approach or specific suggestions

## Alternatives Considered
Other solutions you've thought about

## Additional Context
Examples, mockups, or related issues
```

## 📊 Performance Guidelines

### Benchmarking
- Use `bench.py` for performance testing
- Test on standardized hardware when possible
- Include baseline comparisons
- Document performance improvements

### Optimization Principles
- Minimize memory allocations in hot paths
- Leverage batching for better throughput
- Use appropriate data types (float16 vs float32)
- Profile code before optimizing

## 📄 Documentation

### API Documentation
- Document all public functions and classes
- Include usage examples
- Specify parameter types and return values
- Note any side effects or state changes

### User Documentation
- Keep README.md up to date
- Add examples for new features
- Update installation instructions
- Include troubleshooting guides

## 🔒 Security

### Reporting Vulnerabilities
- Do not create public issues for security vulnerabilities
- Email maintainers directly
- Provide detailed reproduction steps
- Allow time for fixes before disclosure

### Security Guidelines
- Validate all user inputs
- Sanitize file paths and model names
- Avoid executing arbitrary code
- Use secure defaults

## 📦 Release Process

### Version Numbering
We follow Semantic Versioning (SemVer):
- `MAJOR.MINOR.PATCH`
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### Release Checklist
- [ ] Update version in `pyproject.toml`
- [ ] Update CHANGELOG.md
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create release notes
- [ ] Tag release in Git

## 🤝 Community

### Communication
- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: General questions and ideas
- Pull Request reviews: Code feedback

### Code of Conduct
Be respectful, inclusive, and constructive in all interactions. We welcome contributors from all backgrounds and experience levels.

## 📚 Resources

### Learning Resources
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Financial ML Papers](https://github.com/firmai/financial-machine-learning)

### Development Tools
- [VS Code Extensions](https://code.visualstudio.com/docs/python/python-tutorial)
- [Git Best Practices](https://www.atlassian.com/git/tutorials/comparing-workflows)
- [Python Testing](https://docs.pytest.org/en/stable/)

## ❓ Questions?

If you have questions about contributing:
1. Check existing issues and discussions
2. Review this contributing guide
3. Create a new discussion or issue
4. Tag maintainers if needed

Thank you for contributing to TradingNanoLLM! 🎉
