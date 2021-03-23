from distutils.core import setup

setup(
    name='array_manager',
    version='0.1',
    packages=[
        'array_manager',
    ],
    install_requires=[
        'numpy',
        'dash==1.2.0',
        'dash-daq==0.1.0',
        'openmdao',
        'pint',
        'sphinx-rtd-theme',
        'sphinx-code-include',
        'jupyter-sphinx',
        'numpydoc',
    ],
)

