# Python Environment Setup

This project follows [PEP 621](https://peps.python.org/pep-0621/) packaging and lists its dependencies in **`pyproject.toml`**.  
You can work with either the built-in `venv` module (recommended) or with **conda**.

---

## Prerequisites

- **Python 3.11**  
  Download from <https://www.python.org/downloads/release/python-3110/>
- **Git** – <https://git-scm.com/>  
- *(Optional)* **Miniconda** – <https://docs.conda.io/en/latest/miniconda.html>

---

## Setup Instructions

### 1  Clone the repository

```bash
git clone https://github.com/your-org/your-repo.git
cd your-repo
```

### 2 Create a python environment and install dependencies
The idea of this approach is to ensure that:
1. no python environment is precious
2. everybody can run the code and get the same results
3. everybody can use their favorite platform and package manager

In this example we're specifying python 3.11. We should decide as a team what python version we will do our development in. Google colab is currently (release notes)[https://colab.research.google.com/notebooks/relnotes.ipynb] at 3.11. It's not the most cutting edge (python is already on 3.13), but it is modern and well supported.

Option A: use `venv`, the "pure python" virtual environment appraoch:

**macOS / Linux**
```bash:
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

**Windows (PowerShell or CMD) ***
```powershell:
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```

Option B: use conda for environment management
**macOS / Linux**
```bash:
conda update conda
conda create --prefix .conda python=3.11
conda activate ./.conda
pip install --upgrade pip
pip install -e .
```

**Windows (PowerShell or CMD) ***
```powershell:
conda create --prefix .conda python=3.11 pip
conda activate ./.conda
pip install --upgrade pip
pip install -e .
```

### Python editor
You can use whatever editor you like:

**VS Code** 

VScode automatically detects a folder-level `.venv` or `.conda` environment which is handy. If it does not, press Ctrl + Shift + P, choose “Python: Select Interpreter”, and pick the interpreter inside .venv or .conda. Has very nice integration for various AI helpers and formatting pacakges (ruff, black).

**Spyder** 

Spyder is very good for doing analysis, but is a little clunkier to change environments, and has less handy features for working in large code bases than VScode. Easily installed in a conda environment, or can be downloaded separately. Note that it will have issues in an environment that also has PySide6 in it, since it is written in PyQt5, and will have conflicts on which backend to use.

## Package Management

### How are the packages managed?
The repository has a single `pyproject.toml` file, which specifies the versions of packages which are required to run the project. You should be as specific as needed, but not overly restrictive on required versions. Being looser can be helpful to avoid conflicts, but we should be as restrictive as needed to ensure reliable installation/operation. The packages are listed under `[project.dependencies]`.

### Adding a new dependency
Edit `pyproject.toml`, and add the required package under `[project.dependencies]`:

```toml:
[project]
dependencies = [
  "numpy >=1.26",
  "scikit-learn",        # ← added
]
```

Then, re-install the necessary packages. We will let pip handle this:
```bash:
pip install -e .
```

If there are required formatting tools needed (eg all new files should be formatted with black). 

### Developer Tools (optional)

Option A:

These can be manually installed:
```bash:
pip install pytest ruff black
```

Option B:

These can be listed in the `pyproject.toml`, eg:

```toml:
[project.optional-dependencies]
dev = ["pytest", "ruff", "black"]
```

And then installed with the rest of the package:
```bash:
pip install -e ".[dev]"
```

### Well you blew it. You ruined the environment. Now what?
Reset it by deleting it:

**macOS / Linux**
```bash:
rm -rf .venv/ .conda/ __pycache__/ *.egg-info

```

**Windows / PowerShell**
```powershell:
Remove-Item -Recurse -Force .venv, .conda, __pycache__
```
Then reinstall again following the prior instructions.


