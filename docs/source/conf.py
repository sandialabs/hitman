# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os

sys.path.insert(0, os.path.abspath("../../"))

print(sys.path)

project = "HITMAN"
copyright = (
    "2024-2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS)"
)
author = "HITMAN Developers"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx.ext.doctest",
    "sphinx.ext.inheritance_diagram",
    "myst_nb",
    "sphinxcontrib.bibtex",
    "sphinx.ext.autosectionlabel",
]
bibtex_bibfiles = ["references.bib"]
# bibtex_default_style = "unsrt" # This is a problem for notebooks, etc.


templates_path = ["_templates"]
exclude_patterns = []

# napoleon_use_param = False
# napoleon_use_rtype = True
# autodoc_typehints = "description"

# Don't expand named default values
autodoc_preserve_defaults = True

# Include __init__ docstring for classes
autoclass_content = "both"  # or 'init'

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# Enable latex-style equations $ and $$ in ipynb
myst_enable_extensions = ["dollarmath", "amsmath"]
myst_heading_anchors = 3
# Some notebooks take a while to run...
nb_execution_timeout = 180
nb_number_source_lines = True

# Error if notebooks not successful
# nb_execution_allow_errors = False
# nb_execution_raise_on_error = True

# typehints_fully_qualified = True


# latex_elements = {'preamble': r'\input{latex_macros.tex.txt}'}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/hitman_logo.png"  # Path to your logo image relative to conf.py

html_theme_options = {
    "logo_only": True,
    "display_version": False,  # Optional: hide the version number
}
