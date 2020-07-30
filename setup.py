from setuptools import setup, find_packages

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='asgl',
    version='v0.0.1',
    author='Alvaro Mendez Civieta',
    author_email='almendez@est-econ.uc3m.es',
    license='GNU General Public License',
    zip_safe=False,
    url='https://github.com/alvaromc317/ASGL',
    description='A regression solver for linear and quantile regression models and lasso based penalizations',
    long_description=long_description,
    python_requires='>=3.5',
    install_requires=["cvxpy >= 1.1.0",
                      "numpy >= 1.15",
                      "scikit-learn >= 0.23.1"],
    packages=find_packages()
)
