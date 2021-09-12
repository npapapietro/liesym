# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'liesym'
copyright = '2021, Nathan Papapietro'
author = 'Nathan Papapietro'

# The full version, including alpha/beta/rc tags
release = '0.6.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx_math_dollar',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinx.ext.viewcode',
    'numpydoc',
    'sphinx.ext.inheritance_diagram'

]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

pygments_style = 'sphinx'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

autodoc_inherit_docstrings = True
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False

# MathJax file, which is free to use.  See https://www.mathjax.org/#gettingstarted
# As explained in the link using latest.js will get the latest version even
# though it says 2.7.5.
mathjax_path = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-AMS_HTML-full'


mathjax_config = {
    'tex2jax': {
        'inlineMath': [["\\(", "\\)"]],
        'displayMath': [["\\[", "\\]"]],
    },
}

# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

autodoc_mock_imports = ["liesym.liesym"]

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_type_aliases = {
    "Matrix": "sympy.Matrix",
    "Symbol": "sympy.Symbol",
    "Basic": "sympy.Basic"
    # "NumericSymbol": "NumericSymbol"
}

# latex_engine = 'pdflatex'

# latex_elements = {

#     'preamble': r'''
# \usepackage[titles]{tocloft}
# \usepackage{etoolbox}
# \usepackage{dynkin-diagrams}
# \cftsetpnumwidth {1.25cm}\cftsetrmarg{1.5cm}
# \setlength{\cftchapnumwidth}{0.75cm}
# \setlength{\cftsecindent}{\cftchapnumwidth}
# \setlength{\cftsecnumwidth}{1.25cm}
# ''',
#     'printindex': r'\footnotesize\raggedright\printindex',
# }
# latex_show_urls = 'footnote'
