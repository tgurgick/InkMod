#!/usr/bin/env python3
"""Setup script for InkMod."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = requirements_path.read_text().splitlines() if requirements_path.exists() else []

setup(
    name="inkmod",
    version="0.1.0",
    description="A CLI tool that mirrors writing styles using OpenAI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/inkmod",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    py_modules=["inkmod"],
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "inkmod=inkmod:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    keywords="cli, writing, style, openai, ai",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/inkmod/issues",
        "Source": "https://github.com/yourusername/inkmod",
        "Documentation": "https://github.com/yourusername/inkmod#readme",
    },
) 