from setuptools import setup, find_packages
from os import path

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='asgl',
    version='v1.0.1',
    author='Alvaro Mendez Civieta',
    author_email='almendez@est-econ.uc3m.es',
    license='GNU General Public License',
    zip_safe=False,
    url='https://github.com/alvaromc317/asgl',
    dowload_url='https://github.com/alvaromc317/asgl/archive/v1.0.1.tar.gz',
    description='A regression solver for linear and quantile regression models and lasso based penalizations',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['variable-selection', 'regression', 'penalization', 'lasso', 'adaptive-lasso', 'group-lasso',
              'sparse-group-lasso'],
    python_requires='>=3.5',
    install_requires=["cvxpy >= 1.1.0",
                      "numpy >= 1.15",
                      "scikit-learn >= 0.23.1"],
    packages=find_packages()
)
