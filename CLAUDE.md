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

### Service Management
- **Build service**: `make service-build` (builds Docker image)
- **Start service**: `make service-start` (starts API at http://localhost:8000)
- **Stop service**: `make service-stop` (stops running service)
- **Quick start**: `make service-quick-start` (build + start in one command)
- **Model selection**: `FLORA_MODEL_TYPE=<type> make service-start`
  - Available types: `heuristic`, `decision_tree`, `random_forest`, `xgboost`

### API Development
- **Start API locally**: `make api-dev` (with auto-reload)
  - `FLORA_MODEL_TYPE=heuristic make api-dev` - Rule-based classifier
  - `FLORA_MODEL_TYPE=decision_tree make api-dev` - Decision tree (96% accuracy)
  - `FLORA_MODEL_TYPE=random_forest make api-dev` - Random forest (96% accuracy)
  - `FLORA_MODEL_TYPE=xgboost make api-dev` - XGBoost (not implemented)
  - `make api-dev ARGS='--model-type decision_tree --log-level debug'` - CLI arguments
  - `make api-dev ARGS='--port 8001 --host localhost'` - Custom port/host
- **Validate API**: `make api-validate` (run comprehensive tests)
- **API with docs**: `make api-docs` (opens Swagger UI)

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
- **Inline comments**: Only add when explaining non-obvious logic or highlighting important business logic steps
  - Never explain what the code does (redundant with readable code)
  - Only explain why the code does something unexpected or complex
  - Use for important algorithm steps (e.g., `# Rule 1: Perfect Setosa separation`)
  - Avoid obvious comments like `# Initialize variables` or `# Return result`
- **Docstrings**: Follow strict guidelines to avoid redundancy and maintain value
  - **NEVER add docstrings to trivial functions** like `__init__`, simple getters/setters, or self-explanatory methods
  - **NEVER include Args/Returns sections** unless absolutely crucial for understanding complex behavior
  - **ONLY add docstrings** when they provide clear, non-obvious content that supports understanding
  - **Focus on domain knowledge** like business rules, EDA insights, architectural patterns, or complex algorithms
  - **Examples of good docstrings**: Document specific business logic rules, explain domain-specific thresholds, describe architectural patterns
  - **Examples to avoid**: Restating what the function name conveys, obvious parameter descriptions, redundant return value descriptions
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

#### General Structure
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

#### Feature Integration Template
For major feature additions like new predictors, use this concrete structure:

```
feat: integrate {FeatureName} with {MainBenefit}

- Implement {ComponentName} class with {TechnicalDetails}
- Add comprehensive test suite with {TestCount} test cases covering {Coverage}%
- Register {feature_name} in configuration and factory with {Integration}
- Promote {artifact} to {location} for {purpose}
- Refactor {system} with {improvement} for {benefit}
- Fix {issue} for {resolution}
- Update {component} to {change} for {reason}

{FeatureName} Performance: {metrics} on {dataset}
{SystemImprovement}: {description}
All Tests Pass: {coverage} test coverage maintained, {validation} verified

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Example (Decision Tree Integration):**
```
feat: integrate decision tree predictor with service management tooling

- Implement DecisionTreePredictor with 5-feature engineering (4 original + petal_area)
- Add comprehensive test suite with 8 test cases covering initialization, prediction, and error handling
- Register decision_tree model type in configuration and factory with proper model path resolution
- Promote production model to registry/prd/decision_tree.joblib for Docker packaging
- Refactor Makefile with tool-agnostic service-* targets (service-build, service-start, service-stop, service-validate)
- Replace docker-* commands with service-* commands for cleaner developer experience
- Fix Docker working directory to /app for consistent model path resolution
- Update API integration tests to validate all three predictors (heuristic, decision_tree, random_forest)

Model Performance: 96.0% accuracy on full iris dataset with 5 engineered features
Service Management: New service-quick-start command for one-step build and deployment
All Tests Pass: 95% test coverage maintained, Docker deployment verified
```

### PR Description Template
**Important**: Always create the PR description in a .md file first (e.g., `pr_description.md`) so the user can review it. Delete this file after the PR is created.

- **Overview**: High-level summary of changes that focuses on the main feature, fix or experiment being pushed. Ask the user for clarification if needed
- **Key changes**: Main introduced abstractions and supported changes, communicated in a feature-focused way with changes grouped by functionality. Mention the file where each key change happened in each bullet point
- **Tests**: Describe the tests added and their focus, mention the coverage if new tests are added
- **Next steps**: Future work or enhancements planned

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

---

# 🌸 PROJECT-SPECIFIC: AI Flora Mind Integration Patterns

**Note**: This section contains project-specific patterns and processes for the AI Flora Mind iris classification system.

## Adding New Predictors - Complete Integration Process

This documents the exact process we followed to integrate the Random Forest predictor, which should be repeated for future predictors (Decision Tree, XGBoost, etc.).

### Phase 1: Predictor Implementation

#### 1.1 Create Predictor Class
- **Location**: `ai_flora_mind/predictors/{algorithm_name}.py`
- **Pattern**: Inherit from `BasePredictor` abstract class
- **Requirements**:
  - Implement `predict(measurements: IrisMeasurements) -> str` method
  - Handle model loading in `__init__` if file-based
  - Use structured logging for observability
  - Include comprehensive error handling

**Example Structure**:
```python
# ai_flora_mind/predictors/random_forest.py
class RandomForestPredictor(BasePredictor):
    def __init__(self, model_path: str) -> None:
        # Model loading with error handling
        
    def predict(self, measurements: IrisMeasurements) -> str:
        # Feature engineering + prediction
```

#### 1.2 Update Predictor Module
- **File**: `ai_flora_mind/predictors/__init__.py`
- **Action**: Export new predictor class
```python
from .random_forest import RandomForestPredictor
```

#### 1.3 Add Comprehensive Unit Tests
- **Location**: `tests/predictors/test_{algorithm_name}.py`
- **Required Test Coverage**:
  - Initialization (both valid and invalid model paths)
  - Prediction accuracy with known test cases
  - Edge case handling (extreme values)
  - Feature preparation (if applicable)
  - Error scenarios (missing files, invalid models)
  - Interface compliance with BasePredictor

### Phase 2: Configuration Integration

#### 2.1 Add Model Type Enum
- **File**: `ai_flora_mind/configs.py`
- **Pattern**: Add to `ModelType` enum
```python
class ModelType(Enum):
    HEURISTIC = "heuristic"
    RANDOM_FOREST = "random_forest"
    NEW_ALGORITHM = "new_algorithm"  # Add here
```

#### 2.2 Update Model Path Configuration
- **File**: `ai_flora_mind/configs.py`
- **Method**: `ServiceConfig.get_model_path()`
- **Pattern**: Add new case to match statement
```python
match self.model_type:
    case ModelType.NEW_ALGORITHM:
        return f"{base_path}/new_algorithm.joblib"
```

### Phase 3: Factory Integration

#### 3.1 Register Predictor in Factory
- **File**: `ai_flora_mind/factory.py`
- **Pattern**: Add elif clause in `get_predictor()` function
```python
elif config.model_type == ModelType.NEW_ALGORITHM:
    model_path = config.get_model_path()
    if not model_path:
        raise ValueError("New Algorithm model requires a file path")
    predictor = NewAlgorithmPredictor(model_path=model_path)
    logger.info("New Algorithm predictor created successfully", model_path=model_path)
    return predictor
```

### Phase 4: Production Model Registry

#### 4.1 Model Promotion Process
1. **Research Phase**: Models trained and saved in `research/models/` with timestamps
2. **Selection**: Choose best performing model based on evaluation metrics
3. **Promotion**: Copy selected model to production registry

**Commands**:
```bash
# Copy model from research to production registry
cp research/models/new_algorithm_optimized_2025_XX_XX_XXXXXX.joblib registry/prd/new_algorithm.joblib

# Verify model file
ls -la registry/prd/
```

#### 4.2 Production Registry Standards
- **Location**: `registry/prd/`
- **Naming Convention**: `{algorithm_name}.joblib` (no timestamps)
- **Content**: Only production-ready, tested models
- **Size Consideration**: Keep Docker images lean (include only implemented predictors)

### Phase 5: Testing Integration

#### 5.1 Update API Integration Tests
- **File**: `tests/test_api_layer.py`
- **Action**: Add new model type to parametrized fixtures
```python
@pytest_asyncio.fixture(scope="function", params=[
    ModelType.HEURISTIC, 
    ModelType.RANDOM_FOREST,
    ModelType.NEW_ALGORITHM  # Add here
])
```

#### 5.2 Run Comprehensive Test Suite
```bash
make all-test-validate-branch  # Must pass 90% coverage
```

### Phase 6: Docker Integration

#### 6.1 Update Docker Configuration
- **Files**: `Dockerfile`, `.dockerignore`
- **Action**: No changes needed (registry/prd/ already copied)
- **Verification**: Build and test Docker image
```bash
docker build -t ai-flora-mind:test .
FLORA_MODEL_TYPE=new_algorithm docker run --rm ai-flora-mind:test
```

### Phase 7: Documentation and Deployment

#### 7.1 Update Environment Configuration
- **File**: `docker-compose.yml`
- **Action**: Update model type options in comments
```yaml
environment:
  # Options: heuristic, random_forest, new_algorithm
  - FLORA_MODEL_TYPE=${FLORA_MODEL_TYPE:-heuristic}
```

#### 7.2 Update CLI Tools
- **File**: `scripts/isolation/api_layer.py`
- **Action**: Add to choices list
```python
choices=["heuristic", "random_forest", "new_algorithm"]
```

### Phase 8: Validation Checklist

Before committing new predictor integration:

- [ ] **Unit Tests**: New predictor has comprehensive test coverage
- [ ] **Integration Tests**: API layer tests pass with all model types
- [ ] **Configuration**: Model type enum and path configuration updated
- [ ] **Factory**: Predictor registered in factory with proper error handling
- [ ] **Registry**: Production model copied to `registry/prd/` with clean name
- [ ] **Service**: Build succeeds and service runs with new model type (`make service-quick-start`)
- [ ] **Documentation**: CLI tools and deployment docs updated
- [ ] **Validation**: `make all-test-validate-branch` passes
- [ ] **Manual Testing**: API responds correctly with new predictor

### Phase 9: Commit Structure

Follow this commit pattern for predictor integration:

```
feat: integrate {AlgorithmName} predictor with production registry

- Implement {AlgorithmName}Predictor class with {feature_count} feature engineering
- Add comprehensive unit tests with {test_count} test cases covering {coverage}%
- Register {algorithm_name} model type in configuration and factory
- Promote trained model to registry/prd/{algorithm_name}.joblib
- Update API integration tests to include {algorithm_name} model type
- Verify service deployment with `FLORA_MODEL_TYPE={algorithm_name} make service-start`

Model Performance: {accuracy}% accuracy on test set
Production Ready: All tests pass, Docker verified, documentation updated

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

## Model Lifecycle Management

### Research to Production Pipeline
1. **Research Phase** (`research/models/`): Experimentation with timestamped models
2. **Evaluation Phase** (`research/results/`): Performance metrics and model selection
3. **Promotion Phase** (`registry/prd/`): Copy best model with clean naming
4. **Integration Phase**: Code integration following above process
5. **Deployment Phase**: Docker build and environment configuration

### Model Versioning Strategy
- **Research Models**: Keep timestamps for experiment tracking
- **Production Models**: Use semantic algorithm names only
- **Rollback Strategy**: Maintain previous model versions in registry subdirectories
- **A/B Testing**: Support multiple model variants through configuration

### Performance Monitoring
- **Prediction Logging**: Structured logs with input/output pairs
- **Model Metrics**: Track prediction latency and accuracy
- **Error Tracking**: Monitor prediction failures and model loading issues
- **Resource Usage**: Memory and CPU consumption per model type

This integration process ensures consistent, testable, and maintainable predictor additions to the AI Flora Mind system.