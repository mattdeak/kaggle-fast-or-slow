from setuptools import find_packages, setup

setup(
    name="kaggle-fs",
    packages=find_packages(),
    version="0.1.0",
    description="Kaggle Fast and Slow | Model RunTime Prediction",
    entry_points={
        "console_scripts": ["setup-parquet=scripts.setup_parquet:main"],
    },
)
