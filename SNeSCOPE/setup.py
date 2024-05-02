from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

description='SNeSCOPE is a python package for modeling supernova light curves using the analytic models of Morag et al 2023 and Morag et al 2024.',
long_description=open('README.md').read(),
long_description_content_type='text/markdown',

setup(
    name='SNeSCOPE',
    version='1.0.3',
    description='Published version (April 2024)',  
    author='Ido Irani',
    author_email='idoirani@gmail.com', 
    url = 'https://github.com/idoirani/shock_cooling',
    keywords='astronomy',  
    packages=["SNeSCOPE"],
    #packages=find_packages(),
    install_requires=['numpy','matplotlib','dynesty','scipy','numba','astropy','pandas','tqdm','ipdb','extinction'],  
    python_requires='>=3.6',
)

