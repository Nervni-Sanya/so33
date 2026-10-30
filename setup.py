"""Minimal setup.py — install with:  pip install -e ."""

from setuptools import setup, find_packages

setup(
    name="so33-activation",
    version="1.0.0-beta.3",
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
        # Everything the benchmark harness needs, incl. the top-tagging
        # download:  pip install -e ".[bench]"
        "bench": ["scikit-learn>=1.0", "scipy", "matplotlib", "huggingface_hub",
                  "pandas", "pyarrow", "tables"],
        # Only for the linear-EFP baseline (benchmarks.run_efp_baseline).
        # energyflow pulls in wasserstein, which pins numpy<2; the EFP code
        # works with numpy 2, so re-install numpy>=2 afterwards if pip
        # downgraded it.
        "efp": ["energyflow>=1.3"],
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
