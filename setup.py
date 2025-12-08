"""
Setup script for MicroVLM-E.
"""

from setuptools import setup, find_packages

setup(
    name="microvlm-e",
    version="0.1.0",
    author="MicroVLM Team",
    description="MicroVLM-E: Micro Vision-Language Model - Efficient",
    long_description=open("PROJECT_STRUCTURE.txt").read() if __import__("os").path.exists("PROJECT_STRUCTURE.txt") else "",
    long_description_content_type="text/plain",
    url="https://github.com/microvlm/microvlm-e",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "transformers>=4.36.0",
        "peft>=0.7.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",
        "timm>=0.9.12",
        "einops>=0.7.0",
        "omegaconf>=2.3.0",
        "Pillow>=10.0.0",
        "tqdm>=4.66.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "requests>=2.31.0",
        "huggingface_hub>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "training": [
            "wandb>=0.16.0",
            "tensorboard>=2.15.0",
            "datasets>=2.15.0",
        ],
        "data": [
            "webdataset>=0.2.77",
            "img2dataset>=1.45.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "microvlm-train=train:main",
            "microvlm-inference=inference:main",
            "microvlm-export=export_model:main",
        ],
    },
)

