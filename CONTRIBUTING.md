# Contributing to Neural Image Caption Studio

Thank you for your interest in contributing to **Neural Image Caption Studio**! We welcome contributions ranging from bug fixes and documentation enhancements to new neural architectures and evaluation metrics.

---

## 🛠️ Development Setup

1. **Fork and Clone the Repository**:
   ```bash
   git clone https://github.com/abhishek947kumar/image-caption-generator.git
   cd image-caption-generator
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Verify the Installation**:
   ```bash
   pytest tests/
   ```

5. **Start the Development Server**:
   ```bash
   python run.py
   ```

---

## 🌿 Branching Strategy

- **`main`**: Production-ready, fully tested codebase.
- **Feature Branches**: `feat/<feature-name>` (e.g. `feat/attention-mechanism`)
- **Bug Fix Branches**: `fix/<bug-description>` (e.g. `fix/beam-search-eos`)
- **Documentation**: `docs/<topic>` (e.g. `docs/flickr8k-guide`)

---

## 🧪 Testing Guidelines

Before opening a pull request, ensure all tests pass:
```bash
pytest tests/ -v
```

If you add new modules or decoding strategies, please include corresponding unit tests in the `tests/` directory.

---

## 📜 Commit Conventions

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:
- `feat`: A new feature or capability
- `fix`: A bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring without behavioral changes
- `test`: Adding or correcting tests
- `chore`: Maintenance tasks, dependencies, tooling

---

## 📬 Submitting a Pull Request

1. Push your branch to your GitHub fork:
   ```bash
   git push origin feat/your-feature-name
   ```
2. Open a Pull Request against the `main` branch.
3. Fill out the PR template describing your changes, motivation, and any test results.
4. Ensure continuous integration checks pass.
