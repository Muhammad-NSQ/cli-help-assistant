from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cli-help-assistant",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Natural language help for command-line tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "click>=8.0.0",
        "sentence-transformers>=2.2.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "faiss-cpu>=1.7.0",
        "pyyaml>=6.0",
        "rich>=13.0.0",
        "jinja2>=3.1.0",
        "numpy>=1.24.0",
    ],
    entry_points={
        "console_scripts": [
            "cli-help=cli_help.main:cli",
        ],
    },
)
