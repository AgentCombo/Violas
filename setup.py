from pathlib import Path

from setuptools import setup


README = Path(__file__).parent / "PYPI_README.md"


setup(
    name="violas",
    version="0.0.1",
    description="Core grouped vector retrieval primitives from the Violas project.",
    long_description=README.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Violas Authors",
    license="Apache-2.0",
    url="https://github.com/DoubleNorth/Violas",
    project_urls={
        "Documentation": "https://github.com/DoubleNorth/Violas#readme",
        "Issues": "https://github.com/DoubleNorth/Violas/issues",
    },
    python_requires=">=3.9",
    packages=["violas", "violas.storage", "violas.core"],
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "tqdm",
    ],
    extras_require={
        "faiss": ["faiss-cpu"],
    },
    keywords=[
        "vector-database",
        "vector-search",
        "retrieval",
        "semantic-search",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
