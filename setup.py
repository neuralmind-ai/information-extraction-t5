"""Install information_extraction_t5"""
import os
from distutils.core import setup
from setuptools import find_packages

pkg_dir = os.path.dirname(__name__)

with open(os.path.join(pkg_dir, 'requirements.txt'), 'r', encoding='utf-8') as fd:
    requirements = fd.read().splitlines()

setup(
    name='information_extraction_t5',
    version='1.0',
    packages=find_packages('.', exclude=['data*',
                                         'lightning_logs*',
                                         'models*']),
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    install_requires=requirements,
)
