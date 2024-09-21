from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="src",
    version="0.1.0",
    description="A project for training models to generate Kilterboard climbs.",
    author="Ilya Senatorov",
    author_email="il.senatorov@protonmail.com",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.11",
)
