# Contributing to Human3D

Thank you for your interest in contributing to Human3D! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Set up the development environment (see [Development Setup](#development-setup))
4. Create a new branch for your feature or bugfix

## How to Contribute

### Reporting Bugs

- Check existing issues to avoid duplicates
- Use the bug report template
- Include steps to reproduce, expected behavior, and actual behavior
- Include system information (OS, Python version, GPU, etc.)

### Suggesting Features

- Use the feature request template
- Describe the use case and expected behavior
- Explain why this feature would be useful

### Submitting Code

1. Ensure your code follows the style guidelines
2. Write or update tests as needed
3. Update documentation if necessary
4. Submit a pull request

## Development Setup

```bash
# Clone the repository
git clone https://github.com/Ashritha-sai/Human3D---Reconstruction.git
cd Human3D

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Run tests
pytest tests/
```

### Downloading Model Checkpoints

```bash
# SAM checkpoint
mkdir -p checkpoints
wget -P checkpoints/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
mv checkpoints/sam_vit_b_01ec64.pth checkpoints/sam_vit_b.pth

# YOLOv8 pose models (auto-downloaded on first run, or manual)
# See https://docs.ultralytics.com/models/yolov8/
```

## Pull Request Process

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and commit with clear messages:
   ```bash
   git commit -m "Add feature: description of changes"
   ```

3. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Open a Pull Request** against the `main` branch

5. **Address review feedback** if any

### PR Requirements

- All tests must pass
- Code must follow style guidelines
- New features should include tests
- Documentation should be updated if needed

## Style Guidelines

### Python Code Style

- Follow [PEP 8](https://pep8.org/) conventions
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use meaningful variable and function names

### Code Organization

```
src/human3d/
├── models/      # Deep learning model wrappers
├── reconstruct/ # 3D reconstruction modules
├── utils/       # Utility functions
└── viz/         # Visualization helpers
```

### Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Keep the first line under 72 characters
- Reference issues when applicable

### Documentation

- Add docstrings to new functions and classes
- Update README.md for user-facing changes
- Include inline comments for complex logic

## Questions?

Feel free to open an issue for any questions about contributing.

---

Thank you for contributing to Human3D!
