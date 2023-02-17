from setuptools import find_packages, setup

requirements = ["boto3", "click", "Flask", "Flask-API", "PyYAML"]

def_requirements = [
    "black",
    "bumpversion",
    "coverage",
    "flake8",
    "isort",
    "pytest",
    "pytest-cov",
    "radon[flake8]",
    "tox",
    "moto",
]

setup(
    name="ml2p",
    version="0.3.3",
    url="http://github.com/prodigyfinance/ml2p",
    license="ISCL",
    description=(
        "A minimum-lovable machine-learning pipeline, built on top of AWS SageMaker."
    ),
    long_description=open("README.rst", "r").read(),
    author="Prodigy Finance",
    author_email="devops@prodigyfinance.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    extras_require={"dev": def_requirements},
    entry_points={
        "console_scripts": ["ml2p=ml2p.cli:ml2p", "ml2p-docker=ml2p.docker:ml2p_docker"]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: ISC License (ISCL)",
        "Operating System :: POSIX",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
