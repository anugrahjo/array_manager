"""Sphinx configuration for array_manager."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from array_manager import __version__

project = 'array_manager'
author = 'Anugrah Jo Joshy'
copyright = '2021, LSDO Lab, UCSD'
version = release = __version__
extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.doctest',
    'sphinx.ext.mathjax',
]
source_suffix = {'.md': 'markdown'}
root_doc = 'index'
language = 'en'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
html_theme = 'sphinx_rtd_theme'
