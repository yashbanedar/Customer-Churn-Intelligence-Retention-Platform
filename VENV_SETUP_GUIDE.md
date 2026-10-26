# Python Virtual Environment Setup & Installation Guide

This guide details how to configure, activate, and manage the Python 3.13 isolated runtime environment for the **Customer Churn Intelligence & Retention Optimization Platform** on Windows, Linux, and macOS.

---

## 1. System Requirements & Runtime Prerequisites

- **Python Version**: Python 3.13.x (Recommended: 3.13.7)
- **Package Manager**: `pip` (version 24.0 or higher)
- **Version Control**: `git` (version 2.40 or higher)
- **Relational Database**: PostgreSQL 16+ (or local PostgreSQL 18 service, or SQLite fallback)

---

## 2. Virtual Environment Initialization

From the project root directory:

### Windows (PowerShell / Command Prompt)
```powershell
# 1. Create isolated virtual environment using Python 3.13 launcher
py -3.13 -m venv .venv

# 2. Allow execution of PowerShell activation scripts (if restricted)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 3. Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

*For standard Windows CMD:*
```cmd
.\.venv\Scripts\activate.bat
```

### macOS / Linux (Bash / Zsh)
```bash
# 1. Create virtual environment
python3.13 -m venv .venv

# 2. Activate virtual environment
source .venv/bin/activate
```

---

## 3. Dependency Installation & Verification

Once activated, your terminal prompt should display `(.venv)`.

```bash
# 1. Upgrade pip, setuptools, and wheel
python -m pip install --upgrade pip setuptools wheel

# 2. Install all pinned platform dependencies
python -m pip install -r requirements.txt

# 3. Verify installed Python and pip location
python -c "import sys; print('Python executable:', sys.executable)"
python -c "import pydantic, sqlalchemy, sklearn; print('Core dependencies successfully loaded!')"
```

---

## 4. Environment Variables Configuration

Copy the example configuration to create your local `.env` file:

```powershell
Copy-Item .env.example .env
```

Ensure your PostgreSQL credentials in `.env` match your local PostgreSQL service:
```ini
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=telco_churn_db
```
*(If PostgreSQL is not running or credentials are not yet configured, set `USE_SQLITE_FALLBACK=True` to run with zero external dependencies).*

---

## 5. Verification: Running Smoke Tests

Run the configuration test suite to confirm the environment is healthy:

```powershell
pytest -v tests/test_config.py
```

---

## 6. Troubleshooting Common Issues

1. **PowerShell Script Execution Blocked**:
   - Error: `...cannot be loaded because running scripts is disabled on this system.`
   - Solution: Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in your PowerShell session.
2. **PostgreSQL Binary (`psycopg2`) Installation Issues**:
   - We specify `psycopg2-binary` in `requirements.txt` to use pre-compiled binaries and eliminate the need for C compiler tools.
3. **Deactivating the Environment**:
   - Run `deactivate` in your terminal when you finish working.
