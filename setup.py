from distutils.core import setup

setup(
    name='array_manager',
    version='0.1',
    packages=[
        'array_manager',
    ],
    install_requires=[
        'numpy',
        'scipy',
        'pint',
        'sphinx-rtd-theme',
        'sphinx-code-include',
        'jupyter-sphinx',
        'numpydoc',
    ],
)

