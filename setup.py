from setuptools import setup, find_packages

setup(
    name='array_manager',
    version='0.1',
    packages=find_packages(),
    #packages=['array_manager'],
    install_requires=[
        'numpy',
        'scipy',
        'pint',
#         'sphinx-rtd-theme',
#         'sphinx-code-include',
#         'jupyter-sphinx',
#         'numpydoc',
    ],
)

