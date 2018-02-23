import os

from setuptools import setup

base_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(base_dir, 'requirements.txt')) as f:
    base_requirements = f.readlines()

setup(name='mlswarm-infrastructure',
      description='Basic infrastructure necessary for ML Swarm usage.',
      keywords=['machine-learning'],
      version='0.1',
      packages=[
          'mlswarm.infrastructure',
          'mlswarm.infrastructure.services',
          'mlswarm.infrastructure.services.estimators',
          'mlswarm.infrastructure.services.parsers'
      ],
      scripts=[],

      author='Lucas David',
      author_email='lucasolivdavid@gmail.com',

      setup_requires=base_requirements,
      install_requires=base_requirements,
      zip_safe=False)
