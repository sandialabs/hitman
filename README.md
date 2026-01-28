# HITMAN

Hermite Interpolation of Trajectories (HIT) and Measurement synthesis for Analysis of Navigators (MAN) is a python libary (HITMAN) for interpolating flight trajectories and generating synthetic IMU measurements.

## Getting started

A virtual environment is recommended. For example, using [conda](https://www.anaconda.com/download)
```
conda create -n hitman  python=3.12 pip
conda activate hitman
```
Or if python is already installed, [venv](https://docs.python.org/3/library/venv.html)
```
python -m venv .venv
.venv\Scripts\activate
```

Then, install requirements
```
pip install -r requirements.txt
```

Run unit tests
```
python -m unittest discover -s tests -p "*test.py"
```

and check out Jupyter notebooks

```.\docs\source\notebooks```

## For developers
Additional installation steps for developers.

### Pre-Commit Hooks
[Pre-commit](https://pre-commit.com/) hooks are automatically evaluated in the CI/CD pipeline according to the configuration file `.pre-commit-config.yaml`. These checks can be run locally, once `pre-commit` is installed (e.g. `pip` above), by installing the git-hook scripts
```
pre-commit install
```
Only files tracked (staged) by git will be checked when committing. To run against all files
```
pre-commit run --all-files
```

### Compile documentation
```
cd docs
make html
```
Note, you can exclude notebooks by listing all extensions except myst_nb or intersphinx
```
sphinx-build -b html -D extensions=sphinx.ext.autodoc,sphinx.ext.mathjax,sphinx.ext.viewcode,sphinx.ext.autosummary,sphinx.ext.napoleon,sphinx_autodoc_typehints,sphinx.ext.doctest,sphinx.ext.inheritance_diagram,sphinxcontrib.bibtex source/ build/html
```
