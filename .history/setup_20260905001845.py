from setuptools import setup, find_packages


setup(
    name='AI-Powered Credit Scoring using alternative data',
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.10, <3.13",
    install_requires=[
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.29.0",
        "pydantic>=2.10.0",
        "pydantic-settings>=2.5.0",
        "python-dotenv>=1.0.0",
        "loguru>=0.7.2",
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.5.0",
        "xgboost>=2.0.0",
        "shap>=0"
    ]
)