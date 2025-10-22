#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ANTsPy documentation build configuration

import os
import sys
import shutil

# -----------------------------------------------------------------------------
# Paths & environment
# -----------------------------------------------------------------------------

# Are we on Read the Docs?
on_rtd = os.environ.get('READTHEDOCS') == 'True'

# Add package root to sys.path
# (docs/ is usually at repo/docs/, package at repo/ants/)
sys.path.insert(0, os.path.abspath('../../'))

# Avoid importing heavy/GUI deps during doc builds
autodoc_mock_imports = [
    '_tkinter', 'matplotlib',
    'nibabel', 'skimage', 'scipy',
    'torch', 'tensorflow', 'keras',
    'vtk', 'itk'  # add/remove as needed
]

if on_rtd:
    # RTD can't load our C++ extension; provide an empty lib/__init__.py
    os.makedirs('../../ants/lib', exist_ok=True)
    shutil.copyfile('emptyinit.py', '../../ants/lib/__init__.py')

# Import after path fixes (safe even when mocked on RTD)
import ants  # noqa: F401
import numpy as np  # noqa: F401
import sphinx_rtd_theme  # noqa: F401

# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
]

# Generate autosummary stub files (required for the API Reference section)
autosummary_generate = True

# Autodoc defaults so module pages list members by default
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# Make cross-ref errors fail the build (helps catch stale links in curated pages)
nitpicky = True
# Silence known, intentional missing targets by adding tuples like:
# nitpick_ignore = [('py:class', 'ants.ANTsImage')]
nitpick_ignore = []

napoleon_preprocess_types = True  # be explicit
# Allows linkage of type aliases in docstrings
# Developers: please try to use actual types rather than aliases
napoleon_type_aliases = {
    'boolean': 'bool',
    'integer': 'int',
    'string': 'str',
    'optional': 'typing.Optional',
    'ndarray': 'numpy.ndarray',
    'array': 'numpy.ndarray',
    'ANTsImage': 'ants.core.ANTsImage',
    '3-tuple': 'typing.Tuple',
    '2-tuple': 'typing.Tuple',
    'n-D tuple': 'typing.Tuple',
}

# Napoleon / docstring parsing
napoleon_use_ivar = True
napoleon_numpy_docstring = True
napoleon_google_docstring = True
napoleon_attr_annotations = True

# Keep object names shorter in headings (omit full module prefix in titles)
add_module_names = False

# Templates & source
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'

# Project info
project = 'ANTsPy'
copyright = '2017-2025, ANTs Contributors'
author = 'ANTs Contributors'

# Versioning (derived from RTD context when available)
version = 'local'
release = 'local'

if on_rtd:
    rtd_version = os.environ.get('READTHEDOCS_VERSION', 'local')
    rtd_version_type = os.environ.get('READTHEDOCS_VERSION_TYPE', 'branch')
    if rtd_version_type == 'tag':
        version = rtd_version.lstrip('v')
        release = f"release {version}"
    elif rtd_version_type == 'branch' and rtd_version == 'latest':
        version = 'dev'
        release = f"dev ({rtd_version})"
    else:
        version = rtd_version
        release = rtd_version

# Language
language = 'en'

# Patterns to ignore
exclude_patterns = []

# Pygments style
pygments_style = 'sphinx'

# TODOs
todo_include_todos = True

# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------

html_theme = 'sphinx_rtd_theme'
# If you keep a custom theme later, set html_theme_path accordingly.

html_logo = '_static/img/antspy-logo.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -----------------------------------------------------------------------------
# HTML Help / Man / Texinfo
# -----------------------------------------------------------------------------

htmlhelp_basename = 'ANTsPydoc'

man_pages = [
    (master_doc, 'ANTsPy', 'ANTsPy Documentation', [author], 1),
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'ANTsPy', 'ANTsPy Documentation', author, 'ANTsPy',
     'Medical imaging analysis library in Python with algorithms for registration, segmentation, and more', 'Miscellaneous'),
]

# -----------------------------------------------------------------------------
# Intersphinx
# -----------------------------------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# -----------------------------------------------------------------------------
# Patch: prevent Sphinx from cross-referencing ivar tags
# (kept from your original config)
# -----------------------------------------------------------------------------

from docutils import nodes  # noqa: E402
from sphinx.util.docfields import TypedField  # noqa: E402
from sphinx import addnodes  # noqa: E402


def patched_make_field(self, types, domain, items, **kw):
    # `kw` catches `env=None` needed for newer sphinx while maintaining
    # backwards compatibility when passed along further down!
    def handle_item(fieldarg, content):
        par = nodes.paragraph()
        par += addnodes.literal_strong('', fieldarg)  # Patch: this line added
        if fieldarg in types:
            par += nodes.Text(' (')
            # NOTE: using .pop() here to prevent a single type node to be
            # inserted twice into the doctree, which leads to
            # inconsistencies later when references are resolved
            fieldtype = types.pop(fieldarg)
            if len(fieldtype) == 1 and isinstance(fieldtype[0], nodes.Text):
                typename = u''.join(n.astext() for n in fieldtype)
                typename = typename.replace('int', 'python:int')
                typename = typename.replace('long', 'python:long')
                typename = typename.replace('float', 'python:float')
                typename = typename.replace('type', 'python:type')
                par.extend(self.make_xrefs(self.typerolename, domain, typename,
                                           addnodes.literal_emphasis, **kw))
            else:
                par += fieldtype
            par += nodes.Text(')')
        par += nodes.Text(' -- ')
        par += content
        return par

    fieldname = nodes.field_name('', self.label)
    if len(items) == 1 and self.can_collapse:
        fieldarg, content = items[0]
        bodynode = handle_item(fieldarg, content)
    else:
        bodynode = self.list_type()
        for fieldarg, content in items:
            bodynode += nodes.list_item('', handle_item(fieldarg, content))
    fieldbody = nodes.field_body('', bodynode)
    return nodes.field('', fieldname, fieldbody)


TypedField.make_field = patched_make_field
