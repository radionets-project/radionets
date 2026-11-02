#!/usr/bin/env python3
import datetime
import os
import tomllib
from pathlib import Path

import radionets

pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
pyproject = tomllib.loads(pyproject_path.read_text())

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    "IPython.sphinxext.ipython_console_highlighting",
    "matplotlib.sphinxext.plot_directive",
    "myst_parser",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_tippy",
    "sphinx_togglebutton",
]

autodoc_mock_imports = [
    "numba",
    "numba.np",
    "numba.np.ufunc",
    "numba.np.ufunc.decorators",
]

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
]

myst_url_schemes = {
    "http": None,
    "https": None,
    "wiki": "https://en.wikipedia.org/wiki/{{path}}#{{fragment}}",
    "doi": "https://doi.org/{{path}}",
    "gh-issue": {
        "url": "https://github.com/radionets-project/radionets/issue/{{path}}#{{fragment}}",
        "title": "Issue #{{path}}",
        "classes": ["github"],
    },
}

bibtex_bibfiles = ["references.bib"]
bibtex_encoding = "utf8"

copybutton_exclude = ".linenos, .gp"
copybutton_selector = "div:not(.no-copybutton) > div.highlight > pre"
copybutton_image_svg = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 640"><!--!Font Awesome Free v7.0.0 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2025 Fonticons, Inc.-->
    <path d="M480 400L288 400C279.2 400 272 392.8 272 384L272 128C272 119.2 279.2 112 288 112L421.5 112C425.7 112 429.8 113.7 432.8 116.7L491.3 175.2C494.3 178.2 496 182.3 496 186.5L496 384C496 392.8 488.8 400 480 400zM288 448L480 448C515.3 448 544 419.3 544 384L544 186.5C544 169.5 537.3 153.2 525.3 141.2L466.7 82.7C454.7 70.7 438.5 64 421.5 64L288 64C252.7 64 224 92.7 224 128L224 384C224 419.3 252.7 448 288 448zM160 192C124.7 192 96 220.7 96 256L96 512C96 547.3 124.7 576 160 576L352 576C387.3 576 416 547.3 416 512L416 496L368 496L368 512C368 520.8 360.8 528 352 528L160 528C151.2 528 144 520.8 144 512L144 256C144 247.2 151.2 240 160 240L176 240L176 192L160 192z"/>
</svg>
"""  # noqa: E501

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "changes",
    "*.log",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"


project = pyproject["project"]["name"]
author = pyproject["project"]["authors"][0]["name"]
copyright = "{}.  Last updated {}".format(
    author, datetime.datetime.now().strftime("%d %b %Y %H:%M")
)
python_requires = pyproject["project"]["requires-python"]

# make some variables available to each page
rst_epilog = f"""
.. |python_requires| replace:: {python_requires}
"""


version = radionets.__version__
# The full version, including alpha/beta/rc tags.
release = version


# -- Version switcher -----------------------------------------------------

# Define the json_url for our version switcher.
json_url = "https://radionets.readthedocs.io/en/latest/_static/switcher.json"

# Define the version we use for matching in the version switcher.,
version_match = os.getenv("READTHEDOCS_VERSION")
# If READTHEDOCS_VERSION doesn't exist, we're not on RTD
# If it is an integer, we're in a PR build and the version isn't correct.
if not version_match or version_match.isdigit():
    # For local development, infer the version to match from the package.
    version_match = "latest" if "dev" in release or "rc" in release else release

    # We want to keep the relative reference when on a pull request or locally
    json_url = "_static/switcher.json"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_file_suffix = ".html"

html_css_files = [
    "css/radionets.css",
    "css/sphinx-design.css",
    "css/tippy.css",
]

html_favicon = "_static/favicon/favicon.ico"

html_theme_options = {
    "github_url": "https://github.com/radionets-project/radionets",
    "header_links_before_dropdown": 5,
    "navbar_start": ["navbar-logo", "version-switcher"],
    "switcher": {
        "version_match": version_match,
        "json_url": json_url,
    },
    "navigation_with_keys": False,
    "show_version_warning_banner": True,
    "icon_links_label": "Quick Links",
    "icon_links": [
        {
            "name": "Radionets Project",
            "url": "https://github.com/radionets-project",
            "type": "url",
            "icon": "https://avatars.githubusercontent.com/u/77392854?s=200&v=4",
        },
    ],
    "logo": {
        "image_light": "../assets/radionets_large_subtitle.png",
        "image_dark": "../assets/radionets_large_subtitle_dark.png",
        "alt_text": "radionets",
    },
    "announcement": """
        <p>radionets is not stable yet, so expect large and rapid
        changes to structure and functionality as we explore various
        design choices before the 1.0 release.</p>
    """,
}

html_title = f"{project}: Visibility Simulations in Python"
htmlhelp_basename = project + "docs"


# Configuration for intersphinx
intersphinx_mapping = {
    "astropy": ("https://docs.astropy.org/en/stable", None),
    "ipywidgets": ("https://ipywidgets.readthedocs.io/en/stable", None),
    "joblib": ("https://joblib.readthedocs.io/en/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numba": ("https://numba.readthedocs.io/en/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "pytest": ("https://docs.pytest.org/en/stable", None),
    "python": ("https://docs.python.org/3", None),
    "rich": ("https://rich.readthedocs.io/en/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}


suppress_warnings = [
    "intersphinx.external",
]


# warnings
nitpick_ignore = [
    # needed for building the docs locally.
    ("py:attr", "assign"),
    ("py:attr", "device"),
    ("py:attr", "dst_type"),
    ("py:attr", "dtype"),
    ("py:attr", "grad_input"),
    ("py:attr", "grad_output"),
    ("py:attr", "non_blocking"),
    ("py:attr", "persistent"),
    ("py:attr", "requires_grad"),
    ("py:attr", "strict"),
    ("py:class", "BatchNorm"),
    ("py:class", "Dropout"),
    ("py:class", "Module"),
    ("py:class", "fastai.callback.core.Callback"),
    ("py:class", "torch._VariableFunctionsClass.tensor"),
    ("py:class", "torch.nn.Parameter"),
    ("py:class", "torch.nn.modules.module.T"),
    ("py:class", "torch.utils.hooks.RemovableHandle"),
    ("py:func", "register_module_forward_hook"),
    ("py:func", "register_module_forward_pre_hook"),
    ("py:func", "register_module_full_backward_hook"),
    ("py:func", "register_module_full_backward_pre_hook"),
    ("py:meth", "nn.Module.load_state_dict"),
]
