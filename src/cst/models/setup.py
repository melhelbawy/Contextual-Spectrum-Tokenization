SETUP_PY = '''
"""
Setup script for Contextual Spectrum Tokenization (CST)
"""

from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="contextual-spectrum-tokenization",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A production-ready dynamic tokenization architecture for transformer models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cst-implementation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
            "pre-commit>=3.3.0",
        ],
        "gpu": [
            "faiss-gpu>=1.7.4",
        ],
        "vision": [
            "opencv-python>=4.7.0",
            "timm>=0.9.0",
        ],
        "audio": [
            "librosa>=0.10.0",
            "torchaudio>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cst-train=cst.training.train:main",
            "cst-evaluate=cst.evaluation.evaluate:main",
            "cst-demo=cst.demo.demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "cst": [
            "configs/*.yaml",
            "data/*.json",
        ],
    },
)
'''
