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
    ],
    extras_require={
        'dev': ['flake8', 'pytest==3.6.0', 'pytest-flake8==1.0.1'],
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
