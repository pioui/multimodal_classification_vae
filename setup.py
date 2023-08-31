#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

requirements = [
    "numpy",
    "torch>=1.0.1",
    "matplotlib>=3.0.3",
    "scikit-learn>=0.20.3",
    "pandas>=0.24.2",
    "tqdm>=4.31.1",
    "statsmodels",
    "arviz",
    "tifffile"
]

setup_requirements = [
    "pytest-runner",
]
test_requirements = [
    "pytest",
]

setup(
    description="Multimodal Classification with Variational Autoencoder",
    version="0.1",
    url = 'https://github.com/pioui/multimodal_classification_vae',
    author='Pigi Lozou',
    author_email='piyilozou@gmail.com',
    license='MIT',
    install_requires=requirements,
    include_package_data=True,
    keywords="mcvae",
    name="mcvae",
    packages=[
        'mcvae', 
        'mcvae.architectures', 
        'mcvae.dataset', 
        'mcvae.inference',
        'mcvae.utils',
        ],
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    zip_safe=False,
)
