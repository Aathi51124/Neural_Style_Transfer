[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[tool.poetry]
name = "stylegan3"
version = "0.1.0"
description = "StyleGAN3: Official PyTorch Implementation"
authors = ["NVIDIA <info@nvidia.com>"]

[tool.poetry.dependencies]
python = ">=3.6"
torch = ">=1.7"
numpy = "*"
scipy = "*"
pillow = "*"

[tool.poetry.dev-dependencies]
# Add any development dependencies here

[build]
includes = ["**/*"]
