import os

from setuptools import setup

base_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(base_dir, 'requirements.txt')) as f:
    base_requirements = f.readlines()

setup(
    name='solvian-ml',
    description='Infrastructure for solvian ML services.',
    keywords=['machine-learning'],
    version='0.1',
    packages=['solvian.ml.infrastructure'],
    scripts=[],

    author='Lucas David',
    author_email='lucasdavid@solvian.com',

    setup_requires=base_requirements,
    install_requires=base_requirements,
    zip_safe=False,
)
