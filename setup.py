"""Minimal setup.py — install with:  pip install -e ."""

from setuptools import setup, find_packages

setup(
    name="so33-activation",
    version="1.0.0",
    description="SO33 Activation: Parallel Transport in Pseudo-Euclidean Space R^{3,3}",
    author="Panchenko Alexander",
    author_email="sascha.panchenko2018@yandex.ru",
    license="MIT",
    packages=find_packages(exclude=["tests*", "examples*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchdiffeq>=0.2.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": ["pytest", "matplotlib", "jupyter"],
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
