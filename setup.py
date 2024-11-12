from setuptools import setup, find_packages
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='Spike-Encoding',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
    # Optional metadata
    author='FZI',
    author_email='nitzsche@fzi.de',
    description='Spike Encoding and Data Utilities',
    url='https://essgitlab.fzi.de/ecs-neuromorphic-demo/spike-encoding'
)
