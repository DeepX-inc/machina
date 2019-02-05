#!/usr/bin/env python

import os
import pkg_resources
import sys

from setuptools import setup
from setuptools import find_packages

with open('README.md', 'r') as f:
    readme = f.read()

install_requires = [
    'cached_property',
    'torch>=0.4.1',
    'joblib>=0.11',
    'gym>=0.10.5',
    'numpy>=1.13.3',
    'terminaltables',
    'pandas',
]

setup(
    name='machina',
    version='0.2.0',
    description='Machina is a library for a deep reinforcement learning.',
    long_description=readme,
    author='Reiji Hatsugai',
    author_email='reiji.hatsugai@deepx.co.jp',
    url='https://github.com/DeepX-inc/machina',
    license='Apache License',
    packages=find_packages(),
    zip_safe=False,
    install_requires=install_requires,
    test_suite='tests'
)
