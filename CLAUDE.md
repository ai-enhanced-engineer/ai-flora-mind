# Development Guide - AI Flora Mind

Complete development guide consolidating essential information for efficient development.

## Quick Start

### Environment Setup
- Install and setup: `make environment-create` (installs `uv`, virtual environment, and pre-commit)
- Sync dependencies: `make environment-sync` (after editing `pyproject.toml`)
- Python version: defined in `.python-version`

### Development Workflow
- **Format code**: `make format` (Ruff auto-formatting)
- **Lint**: `make lint` (Ruff) + `make type-check` (MyPy)
- **Validate**: `make validate-branch` (format + lint + type check)
- **Test**: `make test-validate-branch` (format + lint + test)
- **Line length**: 120 characters

### Testing
- `make unit-test` - Unit tests
- `make functional-test` - Functional tests 
- `make integration-test` - Integration tests (requires external dependencies)
- `make all-test` - All tests + coverage

## Architecture Overview

### Design Principles
**Modular, workflow-based architecture** with clean separation of concerns:

1. **Configuration-Driven** - Components instantiated via config files + dependency injection
2. **Plugin-Based** - Multiple implementations through factory pattern
3. **Extensibility** - Components registered via enums and factories
4. **Abstraction** - Pluggable implementations via abstract interfaces
5. **Type Safety** - Pydantic models + strict MyPy
6. **Observability** - Structured logging and tracing

### Core Components
- **Application Layer** - HTTP API or CLI interface
- **Business Logic** - Core functionality and workflows
- **Configuration** - Pydantic models with validation
- **Registry** - Factory functions for dependency injection
- **Bootstrap** - Application initialization logic

## Core Engineering Principles

### 1. Clarity and Maintainability
- Function and test names must be descriptive and self-explanatory
- Use inline comments only when necessary, and only to explain non-obvious logic
- Avoid docstrings unless documentation is essential or clarifies edge cases
- Follow consistent naming conventions:
  - For tests: `test__{function_name}__{what_is_being_tested}`

### 2. Strong Typing and Static Guarantees
- All code must be fully MyPy-compliant
- Use precise type annotations for all functions and data structures
- Never remove `# type: ignore` comments; these are intentional and must be preserved

### 3. Structured and Observable Code
- Use structured logging with semantic key-value pairs:
  - `logger.debug("event", key1=value1, key2=value2)`
- Avoid unstructured logs or print statements
- Log relevant inputs, outputs, and failure details to support debugging and observability

### 4. Explicit and Minimal Interfaces
- Prefer explicit arguments and flat configuration structures
- Avoid nested or dynamic magic in configurations unless strictly necessary
- Ensure all functions and classes have a clear, single responsibility

### 5. Separation of Concerns
- Maintain a clean modular structure across services, pipelines, or libraries
- Each module or class should encapsulate a distinct, well-defined purpose
- Do not mix unrelated responsibilities within the same abstraction

### 6. Production Readiness by Default
- Assume all code is production-bound:
  - Handle errors defensively
  - Fail loudly and early when assumptions are violated
  - Validate inputs and sanitize outputs as needed
- Include tests for all logic paths, including edge cases

### 7. Tool-Agnostic Logic
- Core logic should not be tightly coupled to third-party tools
- External dependencies must be abstracted behind clear interfaces where possible
- Design systems to allow easy substitution or mocking of dependencies

## Coding Conventions

### Naming & Style
- **Classes**: `PascalCase` (`ServiceConfig`, `DataProcessor`)
- **Functions/Variables**: `snake_case` (`get_config`, `user_id`)
- **Constants**: `UPPER_SNAKE_CASE` (`DEFAULT_TIMEOUT`)
- **Enums**: `PascalCase` class, `UPPER_SNAKE_CASE` values
- **Files**: `snake_case` (`data_processor.py`)

### Configuration Patterns
- **String enums** for registration (`RegisteredComponents`)
- **Pydantic validation** with `Field(description="...")`
- **Custom validators** with `@field_validator`
- **Environment mapping** via `alias` parameter

### Component Development
1. Add to relevant enum for registration
2. Implement with proper interface/signature
3. Add to factory function in registry
4. Create config subclass if needed

## Testing Guidelines

### Test Architecture & Organization

#### Test Classification
1. **Unit Tests** (`@pytest.mark.unit`) - Test individual components in isolation
2. **Functional Tests** (`@pytest.mark.functional`) - Test complete workflows
3. **Integration Tests** (`@pytest.mark.integration`) - Test with external dependencies

### Testing Standards
- **Markers**: `@pytest.mark.{unit,functional,integration}`
- **Fixtures** in `conftest.py`
- **Custom Abstractions**: Avoid mock libraries, use project-specific implementations
- **Async testing** with `pytest-asyncio`
- **Mirror source structure** in test organization

### Custom Abstractions Over Mocks

**CRITICAL: Use Custom Abstractions, Not Mock Libraries**

This project **AVOIDS** explicit mock libraries like `unittest.mock.MagicMock` and `AsyncMock`. Instead, it prioritizes custom abstractions and real implementations designed for testing.

**Preferred Approaches (in order of preference):**

1. **Monkeypatch** (preferred method):
```python
def test_function(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_method(self, user_input: str) -> tuple[str, str]:
        return "EXPECTED_RESULT", "EXPECTED_MESSAGE"
    
    monkeypatch.setattr(TargetClass, "method_name", mock_method)
```

2. **Fake implementations**:
```python
# Use project-specific fake classes
fake_processor = FakeDataProcessor()  # Instead of MagicMock
```

3. **Local implementations with controlled data** (most preferred):
```python
# Use project-specific Local* implementations
local_service = LocalDataService(sample_response={"data": [...]})
```

4. **Custom test classes extending local implementations**:
```python
class FailingService(LocalDataService):
    def process_data(self, data: str) -> str:
        raise RuntimeError("Service failure")
```

### Test Function Naming
```python
def test__component_or_method__specific_behavior_or_scenario() -> None:
    # Use double underscores to clearly separate:
    # 1. Component/method being tested
    # 2. Specific behavior being validated
```

## Error Handling and Patterns

### Error Handling
- We do not declare custom errors in this project, lets raise the error directly where it is needed
```python
try:
    result = await dangerous_operation()
except SpecificException as e:
    logger.error("Operation failed", exc_info=e, extra={"context": "value"})
    raise ServiceException("User-friendly message") from e
```

### Configuration Validation
Always validate configuration at startup:
```python
def validate_config(config: AppConfig) -> None:
    if not config.required_field:
        raise ValueError("Required field must be provided")
```

### Resource Management
Use context managers for all resources:
```python
async with get_resource() as resource:
    await resource.process()
    # Automatic cleanup
```

## Security Considerations

### Input Validation
- All user inputs validated through Pydantic models
- Content filtering and sanitization
- Maximum input length enforcement

### API Key Management
- Environment variable-based configuration
- No hardcoded secrets in code
- Key rotation support

### Data Protection
- Session-based data isolation
- Cross-tenant data protection
- Secure data handling practices

## Deployment and Environment

### Environment Variables
- Use descriptive names with project prefix
- Document all required variables
- Provide sensible defaults where possible
- Use environment aliasing in Pydantic configs

### Monitoring and Observability
- Structured logging with correlation IDs
- Health check endpoints
- Performance monitoring
- Error tracking and alerting

## Communication and Collaboration

### Commit Message Structure
```
type: brief description

- Detailed bullet points of changes
- Performance improvements with metrics
- Quality assurance notes (tests, linting, type safety)
- Impact on architecture or future development
```

**Commit Types:**
- `feat`: New features or functionality
- `fix`: Bug fixes
- `refactor`: Code restructuring without functional changes
- `test`: Testing improvements or additions
- `docs`: Documentation updates

### PR Description Template
- **Overview**: High-level summary of changes
- **Phase-based organization**: Break large changes into logical phases
- **Quality metrics**: Test counts, performance improvements, validation status
- **Benefits**: For developers, architecture, and CI/CD
- **Future enhancements**: Planned next steps

### Validation Checklist
- [ ] All tests pass (existing + new)
- [ ] Linter clean (`make lint`)
- [ ] Type checker clean (`make type-check`)
- [ ] Pre-commit hooks pass
- [ ] Performance benchmarks (for testing changes)
- [ ] Documentation updated

## Extension Points

### Adding New Components
- Create enum entry for registration
- Implement interface or abstract base class
- Add factory function to registry
- Create configuration model
- Add comprehensive tests

### Configuration Management
- Use Pydantic models with validation
- Environment variable mapping
- Hierarchical configuration support
- Runtime configuration updates

This guide provides comprehensive information needed for effective development, combining architectural understanding, implementation details, testing strategies, and operational considerations.