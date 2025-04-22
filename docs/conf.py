import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'chemotaxis-sim'
author = 'Le Chen and Ian Ruau'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = []
html_static_path = []
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
