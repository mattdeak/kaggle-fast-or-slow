from setuptools import find_packages, setup

setup(
    name="kaggle-fs",
    packages=find_packages(),
    version="0.1.0",
    description="Kaggle Fast and Slow | Model RunTime Prediction",
    console_scripts=["setup-parquet=kaggle_fs.scripts:main"],
)
