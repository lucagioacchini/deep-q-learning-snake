from setuptools import setup, find_packages

setup(
    name='deepqsnake',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'keras==3.6.0',
        'tensorflow==2.17.0',
        'matplotlib==3.9.2',
        'pygame==2.6.1',
        'numpy==1.26.4'
    ],
    include_package_data=True,
)
