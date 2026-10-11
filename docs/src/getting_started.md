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

## Simplest use case

Create a vector composed of two subvectors or vector components.

```{doctest}
>>> from array_manager.api import Vector, VectorComponentsDict

>>> # Define vector components with their names and tuple shapes
>>> components = VectorComponentsDict()

>>> # Components are initialized to zero unless initial values are supplied
>>> components["position"] = {"shape": (3,)}
>>> components["velocity"] = {"shape": (3,)}

>>> # Create a vector defined by the components
>>> vector = Vector(components)

>>> # Allocate memory and enable views to access components later by name
>>> vector.allocate(setup_views=True)

>>> # Set the values for a vector component to overwrite its initial value
>>> vector["position"] = [1.0, 2.0, 3.0]

>>> vector.data
array([1., 2., 3., 0., 0., 0.])
```

## Building the documentation

From the repository root, remove any previous build artifacts,
install the documentation dependencies, build the HTML documentation,
and run the documentation tests:

```sh
rm -rf docs/_build
pip install -e . -r docs/requirements.txt
python -m sphinx -E -a -b html -W --keep-going docs docs/_build/html
python -m sphinx -E -a -b doctest -W --keep-going docs docs/_build/doctest
```

The HTML entry point is `docs/_build/html/index.html`.

The first Sphinx command rebuilds all documentation, treats warnings as errors,
and continues processing after errors to report as many issues as possible.
The doctest command executes Python examples in the
documentation recognized by Sphinx's doctest builder.

Documentation source files use MyST Markdown (`.md`). Fenced directives support
Sphinx features such as API references, navigation, and executable examples.
