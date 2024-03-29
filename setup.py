from setuptools import setup, find_packages
import os
from os import path

# Model package name
NAME = 'neoway_nlp'
# Current Version
VERSION = os.environ.get('APP_VERSION', 'latest')

# Dependecies for the package
with open('requirements.txt') as r:
    DEPENDENCIES = [
        dep for dep in map(str.strip, r.readlines())
        if all([not dep.startswith("#"),
                not dep.endswith("#dev"),
                len(dep) > 0])
    ]

# Project descrpition
with open("README.md") as f:
    LONG_DESCRIPTION = f.read()


setup(
    name=NAME,
    version=VERSION,
    description='<@description>',
    long_description=LONG_DESCRIPTION,
    author='TimKartawijaya',
    author_email='tim.kartawijaya@gmail.com',
    license='MIT',
    packages=find_packages(exclude=("tests", "docs")),
    entry_points={
        'console_scripts': [
            '{name}={name}.main:cli'.format(name=NAME)
        ],
    },
    # external packages as dependencies
    install_requires=DEPENDENCIES
)
