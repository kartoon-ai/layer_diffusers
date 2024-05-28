from setuptools import setup, find_packages
from pathlib import Path

script_directory = Path(__file__).resolve().parent

with open(script_directory / 'requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="layer_diffusers",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=required,  # Use the list of requirements as dependencies
    author="Joël Seytre",
    author_email="joel@kartoon.ai",
    description="Diffusers implementation of LayerDiffuse",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/kartoon-ai/layer_diffusers",
)