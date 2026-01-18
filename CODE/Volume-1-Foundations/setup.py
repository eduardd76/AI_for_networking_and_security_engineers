"""
Setup script for AI for Networking Engineers - Volume 1 code examples.

Install with: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().splitlines()

# Filter out comments and empty lines
requirements = [
    req.strip() for req in requirements
    if req.strip() and not req.strip().startswith('#')
]

setup(
    name="ai-networking-volume1",
    version="1.0.0",
    author="Eduard Dulharu (Ed Harmoosh)",
    author_email="ed@vexpertai.com",
    description="Code examples from AI for Networking Engineers - Volume 1: Foundations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eduardd76/AI_for_networking_and_security_engineers",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "Topic :: System :: Networking",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
        ]
    },
    entry_points={
        "console_scripts": [
            # Add CLI tools here if needed
            # "ai-networking=module:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
