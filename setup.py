from setuptools import setup, find_packages

setup(
    name="ml2p",
    version="0.0.1",
    url='http://github.com/prodigyfinance/ml2p',
    license='Unknown',
    description=(
        "A minimum-lovable machine-learning pipeline, built on top of"
        " AWS SageMaker."
    ),
    long_description=open('README.rst', 'r').read(),
    author='Prodigy Finance',
    author_email='devops@prodigyfinance.com',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'boto3',
        'click',
        'flask',
        'flask-API',
        'PyYAML',
    ],
    extras_require={
        'dev': [
            'bumpversion',
            'coverage',
            'flake8',
            'pytest',
            'pytest-cov',
            'tox',
        ],
    },
    entry_points={
        'console_scripts': [
            'ml2p=ml2p.cli:ml2p',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: ISC License (ISCL)',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
