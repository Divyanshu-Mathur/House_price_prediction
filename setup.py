from setuptools import find_packages,setup
from typing import List


Hypen_e_dot ="-e ."

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file:
        requirements = file.readlines() 
        requirements = [i.replace("\n", "") for i in requirements]
        
        if Hypen_e_dot in requirements:
            requirements.remove(Hypen_e_dot)
    return requirements
        

setup(
    name="House_Price_Prediction",
    version='0.0.0.1',
    author="Divyanshu",
    author_email="divyanshumathur2004@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)