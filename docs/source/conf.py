# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------

project = "RAPIDS CLI"
html_title = "RAPIDS CLI"
copyright = f"{datetime.date.today().year}, NVIDIA"
author = "NVIDIA"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = []

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "header_links_before_dropdown": 7,
    "icon_links": [],
    "logo": {
        "link": "https://docs.rapids.ai/",
    },
    "github_url": "https://github.com/rapidsai/rapids-cli",
    "show_toc_level": 1,
    "navbar_align": "right",
}

html_sidebars = {
    "**": ["sidebar-nav-bs", "sidebar-ethical-ads"],
}

html_static_path = ["_static"]


def setup(app):
    app.add_css_file("https://docs.rapids.ai/assets/css/custom.css")
    app.add_css_file(
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.9.0/css/all.min.css"
    )
    app.add_js_file(
        "https://docs.rapids.ai/assets/js/custom.js", loading_method="defer"
    )
