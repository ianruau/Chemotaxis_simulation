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
    py_modules=["simulation"],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "tabulate",
        "tqdm",
        "questionary",
        "termplotlib",
    ],
    entry_points={
        "console_scripts": [
            "chemotaxis-sim=simulation:main",
        ]
    },
    extras_require={
        "docs": [
            "sphinx>=3.0",
            "sphinx-rtd-theme",
        ],
    },
)

