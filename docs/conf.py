"""Sphinx configuration for the QBP documentation."""

from __future__ import annotations

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "QBP"
author = "WISER"
copyright = f"{datetime.now().year}, {author}"

try:
    from importlib.metadata import version as _pkg_version
    release = _pkg_version("qbp")
except Exception:
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_parser",
    "jupyter_sphinx",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".venv", "venv"]

# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = False
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

todo_include_todos = True

# -- MyST --------------------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
    "smartquotes",
    "tasklist",
]
myst_heading_anchors = 3

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "shibuya"
html_title = "QBP"
html_static_path = ["_static"]
html_favicon = "_static/favicon-dark.svg"
html_css_files = ["custom.css"]

html_theme_options = {
    "accent_color": "violet",
    "github_url": "https://github.com/Quantum-Solutions-Launchpad/QSL-NNL-P7",
    "nav_links": [
        {"title": "Getting Started", "url": "getting-started/quickstart"},
        {"title": "User Guide", "url": "user-guide/workflow"},
        {"title": "Models", "url": "models/catalog"},
        {"title": "API", "url": "api/index"},
    ],
}
