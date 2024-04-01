from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
#with open(path.join(here, 'README.md'), encoding='utf-8') as f:
#    long_description = f.read()

setup(
    name='SNeSCOPE',
    version='0.0.1',
    description='development version',
    #long_description=long_description,
    #long_description_content_type='text/markdown',    
    author='Ido Irani',
    author_email='idoirani@gmail.com',  # Optional
    keywords='astronomy',  # Optional
    packages=["SNeSCOPE"],
    #packages=find_packages(),
    install_requires=['numpy','matplotlib','dynesty','scipy','numba','astropy'],  
    python_requires='>=3',
)

