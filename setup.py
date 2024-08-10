from setuptools import setup, find_packages
from os import path

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='asgl',
    version='1.0.6',
    author='Alvaro Mendez Civieta',
    author_email='alvaromc317@gmail.com',
    license='GNU General Public License',
    zip_safe=False,
    url='https://github.com/alvaromc317/asgl',
    dowload_url='https://github.com/alvaromc317/asgl/archive/v1.0.5.tar.gz',
    description='A regression solver for high dimensional penalized linear and quantile regression models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['variable-selection', 'regression', 'penalization', 'lasso', 'adaptive-lasso', 'group-lasso',
              'sparse-group-lasso', 'high-dimension', 'quantile-regression'],
    python_requires='>=3.8',
    install_requires=["cvxpy >= 1.2.0",
                      "numpy >= 1.20.0",
                      "scikit-learn >= 1.0"],
    packages=find_packages()
)
