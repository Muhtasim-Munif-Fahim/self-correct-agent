from setuptools import setup, find_packages

setup(
    name="self-correct",
    version="0.1.0",
    description="A lightweight anti-hallucination wrapper for LLMs using Chain-of-Verification.",
    author="Open Source Developer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Requires the user to bring their own LLM client, but we interface assuming standard messages block
    ],
    extras_require={
        "search": [
            "duckduckgo-search",
        ],
        "dev": [
            "pytest",
        ],
    },
    python_requires=">=3.8",
)
