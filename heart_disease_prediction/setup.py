#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = [ ]

setup(
    author="Guadalupe López",
    author_email='a01688491@tec.mx',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Detecting and preventing the factors that have the greatest impact on heart",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='heart_disease_prediction',
    name='heart_disease_prediction',
    packages=find_packages(include=['heart_disease_prediction', 'heart_disease_prediction.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/Glverdugo/heart_disease_prediction',
    version='0.1.0',
    zip_safe=False,
)
