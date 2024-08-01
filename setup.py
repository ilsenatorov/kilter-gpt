from setuptools import find_packages, setup

setup(
    name="kiltergpt",
    version="0.1.0",
    description="A project for training models to generate Kilterboard climbs.",
    author="Ilya Senatorov",
    author_email="il.senatorov@protonmail.com",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "torchaudio",
        "wandb",
        "pytorch-lightning",
        "plotly",
        "pandas",
        "numpy",
        "matplotlib",
        "ipykernel",
        "jupyter",
        "beartype",
        # Add any other dependencies you might have
    ],
    python_requires=">=3.10",
)
