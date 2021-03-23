from distutils.core import setup

setup(
    name='array assembly and format manager',
    version='0.1',
    packages=[
        'array assembly and format manager',
    ],
    install_requires=[
        'numpy',
        'dash==1.2.0',
        'dash-daq==0.1.0',
        'pint',
        'sphinx-rtd-theme',
        'sphinx-code-include',
        'jupyter-sphinx',
        'numpydoc',
    ],
)

