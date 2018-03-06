#!/usr/bin/env python

import os
import pkg_resources
import sys

from setuptools import setup

with open('README.md', 'r') as f:
    readme = f.read()

setup_requires = []
install_requires = [
    'cached_property',
    'torch>=0.3.0',
    'joblib>=0.11',
    'gym>=0.9.2',
    'numpy>=1.13.3',
    'terminaltables',
]

setup(
    name='machina',
    version='0.0.1',
    description='A PyTorch Library for Reinforcement Learning',
    long_description=readme,
    author='rarilurelo',
    author_email='rarilurelo@gamil.com',
    url='https://github.com/DeepX-inc/machina',
    license='MIT License',
    packages=['machina',
              ],
    zip_safe=False,
    install_requires=install_requires,
    setup_requires=setup_requires,
)

