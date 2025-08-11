"""
Setup configuration for SLAM Simulation System.
"""

from setuptools import setup, find_packages

setup(
    name="slam-simulation",
    version="0.1.0",
    description="SLAM Simulation System with EKF, SWBA, and SRIF estimators",
    author="SLAM Sim Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.6.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "pyyaml>=6.0",
        "typer>=0.9.0",
        "rich>=13.0.0",
        "plotly>=5.14.0",
        "pandas>=2.0.0",
        "transforms3d>=0.4.1",
        "sympy>=1.12",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.3.0",
        ],
        "performance": [
            "numba>=0.57.0",
            "joblib>=1.3.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "slam-sim=tools.cli:main",
        ],
    },
)