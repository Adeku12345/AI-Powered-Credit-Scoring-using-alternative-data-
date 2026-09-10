from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the List of requirements
    '''
    
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPRN_E_DOR)
    
setup(
    name='AI-Powered Credit Scoring using alternative data',
    version="1.0.0",
    author='Josh',
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
        "shap>=0.44.0",
        "pymongo>=4.8.0",
        "python-jose[cryptography]>=3.3.0",
        "passlib[bcrypt]>=1.7.4",
        
        
    ],
    extras_require={"dev": ["pytest>=8.0.0", "black>=24.3.0"]},
)