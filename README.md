
## Build Docs

```sh
make -C docs/ html
```

## View Generated Docs

Open `docs/_build/html/index.html` in your browser.

## What's in this package?

- `docs` stores source files and files required by Sphinx and GitHub for generating docs
  - `_exts` stores extensions for generating docs with extra features
    like including an n2 diagram
  - `_build` contains the built docs
  - `_src_docs` contains docs source files for generating docs
  - `_static` contains a single file, `custom.css` used to make
    `jupyter-sphinx` output match the readthedocs theme
- `atomics` is where all Python code for the project goes
- `.github/workflows/` contains a file that tells GitHub how to build
  the docs

Take a look around inside `docs/index.rst` and `docs/_src_docs/` to get
an idea of how to organize your source files.

## Setting up Docs for your Project

To set up docs:

- Make sure to set `SPHINXPROJ` correctly in `docs/Makefile` and `docs/make.bat`
- I'm not sure if you need `docs/requirements.txt`, but leave it in just in
  case
- Use `docs/.embedrc` to use custom directives (may be necesssary only for
  sphinx_auto_embed, which we no longer use)
- Change README on `docs/build-docs.sh` to match project name.
  - NOTE: `docs/build-docs.sh` works only on Linux.
  - To test docs, run `make -C docs/ html` from the project root.
- Make sure the project name is correct in `docs/conf.py`.
- Make sure the author is correct in `docs/conf.py`.
- The file `.github/workflows/doc_pages_workflow.yml` is used for GitHub
  to automatically generate docs whenever someone pushes to the `master` branch
- To generate example scripts for use in docs and tests (must be done
  manually before running Sphinx), use
  `atomics/utils/generate_example_scripts.py`.
  - NOTE: You will need to `pip install docstirng_parser` in order to
    use `generate_example_scripts.py`. `docstring_parser` is not listed
    as a project dependency in `setup.py`.
  - NOTE: You will need to make sure that you are importing the correct
    project within `generate_example_scripts.py`.
  - NOTE: Follow the examples in `omtools/examples/` to make example
    classes. `generate_example_scripts.py` will generate the scripts in
    `valid/` and `invalid/` directories.

Use the [Sphinx webpage](https://www.sphinx-doc.org/en/master/) as
reference.

## API Docs

For the API docs (under `docs/_src_docs/api/`), use
[autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html).

Use `autodoc` to populate `.rst files` in `_src_docs/api/` and add their
paths to `docs/index.rst` toctree to include API docs for your classes,
functions, etc.
