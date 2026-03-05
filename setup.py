from setuptools import setup, find_packages
from pathlib import Path

long_description = Path("README.md").read_text(encoding="utf-8")

setup(
    name="self-correct",
    version="0.1.0",
    description="A lightweight anti-hallucination wrapper for LLMs using Chain-of-Verification.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Muhtasim Munif Fahim",
    url="https://github.com/Muhtasim-Munif-Fahim/self-correct-agent",
    project_urls={
        "Bug Tracker": "https://github.com/Muhtasim-Munif-Fahim/self-correct-agent/issues",
        "Source Code": "https://github.com/Muhtasim-Munif-Fahim/self-correct-agent",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
    extras_require={
        "search": [
            "duckduckgo-search",
        ],
        "dev": [
            "pytest",
        ],
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
