"""
Instant Policy - In-Context Imitation Learning via Graph Diffusion

Based on the paper: "Instant Policy: In-Context Imitation Learning via Graph Diffusion" (ICLR 2025)
Authors: Vitalis Vosylius and Edward Johns, Imperial College London
"""

from setuptools import setup, find_packages

setup(
    name="instant_policy",
    version="1.0.0",
    description="In-Context Imitation Learning via Graph Diffusion",
    author="Vitalis Vosylius, Edward Johns",
    author_email="vitalis.vosylius19@imperial.ac.uk",
    url="https://www.robot-learning.uk/instant-policy",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "torch-geometric>=2.5.0",
        "pytorch-lightning>=2.4.0",
        "numpy>=1.26.0",
        "scipy>=1.14.0",
        "open3d>=0.18.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "sim": [
            "rlbench",
        ],
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
