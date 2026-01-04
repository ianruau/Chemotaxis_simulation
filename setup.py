#!/usr/bin/env python3
from setuptools import setup

setup(
    name="chemotaxis-sim",
    version="0.1.0",
    description="Chemotaxis simulation CLI tool",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Le Chen and Ian Ruau",
    author_email="chenle02@gmail.com",
    python_requires=">=3.7",
    py_modules=["chemotaxis", "simulation", "paper2_constants", "plot_from_npz", "implied_constants"],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "sympy",
        "argcomplete",
        "tabulate",
        "tqdm",
        "questionary",
        "termplotlib",
        "PyYAML",
    ],
    entry_points={
        "console_scripts": [
            "chemotaxis=chemotaxis:main",
            "chemotaxis-sim=simulation:main",
            "chemotaxis-plot=plot_from_npz:main",
            "chemotaxis-constants=implied_constants:main",
        ]
    },
    extras_require={
        "docs": [
            "sphinx>=3.0",
            "sphinx-rtd-theme",
        ],
    },
)
