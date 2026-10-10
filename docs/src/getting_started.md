# Getting started

## Installation

To install the latest commit from the main branch, run the following command in the terminal:
```sh
pip install git+https://github.com/anugrahjo/array_manager.git
```

To uninstall, run:
```sh
pip uninstall array_manager
```

To upgrade to the latest commit, uninstall and then reinstall using:
```sh
pip uninstall array_manager
pip install git+https://github.com/anugrahjo/array_manager.git
```

To install array_manager in development mode, clone the repository and install it using:
```sh
git clone https://github.com/anugrahjo/array_manager.git
pip install -e ./array_manager
```
The `-e` flag installs the package in editable mode, 
allowing you to modify the source code without needing to reinstall it.

To upgrade to the latest commit in development mode, navigate to the repository root and run:
```sh
git pull
```

## Create a vector

Define named components with tuple shapes. Enable views when allocating to access
components by name. Initial values default to zero.

```{doctest}

>>> from array_manager.api import Vector, VectorComponentsDict
>>> components = VectorComponentsDict()
>>> components["position"] = {"shape": (3,)}
>>> components["velocity"] = {"shape": (3,)}
>>> vector = Vector(components)
>>> vector.allocate(setup_views=True)
>>> vector["position"] = [1.0, 2.0, 3.0]
>>> vector.data
array([1., 2., 3., 0., 0., 0.])
```

## Building the documentation

From the repository root, install the documentation dependencies and build:

```sh
rm -rf docs/_build
pip install -e . -r docs/requirements.txt
python -m sphinx -E -a -b html -W --keep-going docs docs/_build/html
python -m sphinx -E -a -b doctest -W --keep-going docs docs/_build/doctest
```

The HTML entry point is `docs/_build/html/index.html`. These commands treat
warnings as errors and run the documentation examples with Sphinx doctest.

Documentation source files use MyST Markdown (`.md`). Fenced directives preserve
Sphinx features such as API references, navigation, and executable examples.
