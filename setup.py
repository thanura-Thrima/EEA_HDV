from setuptools import find_packages, setup

setup(
    name='RTlib',
    packages=find_packages(include=['RTlib']),
    version='0.0.0.1',
    description='RT Python library',
    author='Raveen Thrimawithana',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)