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
    'torch>=1.0.1',
    'joblib>=0.11',
    'cloudpickle',
    'redis',
    'gym>=0.10.5',
    'numpy>=1.13.3',
    'terminaltables',
    'pandas',
]

setup(
    name='machina-rl',
    version='0.2.1',
    description='machina is a library for a deep reinforcement learning.',
    long_description=readme,
    author='Reiji Hatsugai',
    author_email='reiji.hatsugai@deepx.co.jp',
    url='https://github.com/DeepX-inc/machina',
    license='MIT License',
    packages=find_packages(),
    zip_safe=False,
    install_requires=install_requires,
    test_suite='tests'
)
