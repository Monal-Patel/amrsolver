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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'amrsolver'
copyright = '2023, Monal Patel'
author = 'Monal Patel'

# The full version, including alpha/beta/rc tags
release = 'v0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
  # all the other extension
  "sphinxcontrib.doxylink", "sphinx.ext.mathjax" 
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['./sphinx/_build/templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['website', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['./']


#doxygen_root = "docs/doxygen" # this is just a convenience variable
doxylink = {
    "amrsolver": (  # "demo" is the role name that you can later use in sphinx to reference this doxygen documentation (see below)
        f"./build_doxygen/html/tagfile.xml", # the first parameter of this tuple is the tagfile
        f"./build_doxygen/html", # the second parameter of this tuple is a relative path pointing from
                                     # sphinx output directory to the doxygen output folder inside the output
                                     # directory tree.
                                     # Doxylink will use the tagfile to get the html file name of the symbol you want
                                     # to link and then prefix it with this path to generate html links (<a>-tags).
    ),
}
