"""Setup script for the Insider Threat Detection System."""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="insider-threat-detection",
    version="1.0.0",
    author="AI Security Team",
    author_email="security@example.com",
    description="Advanced machine learning system for detecting insider threats using LSTM/GRU neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/insider-threat-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.800",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "ipykernel>=6.0.0",
            "ipywidgets>=7.6.0",
        ],
        "gpu": [
            "tensorflow-gpu>=2.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "insider-threat-train=scripts.train:main",
            "insider-threat-evaluate=scripts.evaluate:main",
            "insider-threat-predict=scripts.predict:main",
            "insider-threat=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml"],
        "config": ["*.py"],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/insider-threat-detection/issues",
        "Source": "https://github.com/yourusername/insider-threat-detection",
        "Documentation": "https://github.com/yourusername/insider-threat-detection/wiki",
    },
    keywords="machine-learning, cybersecurity, insider-threat, lstm, neural-networks, anomaly-detection",
    zip_safe=False,
)
