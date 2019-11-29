#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "numpy>=1.17.0",
    "networkx>=2.0",
    "pandas>=0.2",
    "scikit-learn>=0.2",
    "scipy>=1.3",
    "matplotlib>=3.0"
]

setup_requirements = ["pytest-runner"]

test_requirements = ["pytest"]

extras_requirements = {"plot": ["pygraphviz"], "test": ["pytest"]}

author = "Michael Aichmueller"

setup(
    author=author,
    author_email="m.aichmueller@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Informatics",
    ],
    description="Causality and Inference Methods",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="causalpy causality invariant causal prediction scm sem",
    name="CausalPy",
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite="test",
    tests_require=test_requirements,
    extras_require=extras_requirements,
    url="https://github.com/maichmueller/scm",
    version="0.0.1",
    zip_safe=False,
)
