# ML Project Road Map

This README provides a step-by-step guide for setting up and organizing your Machine Learning (ML) project using Python. Follow these instructions to create a structured project environment.

## Step 1: Project Initialization

1. **Create Project Folder:** Begin by creating a new folder for your ML project.

2. **Open Visual Studio Code (VSCode):** Open VSCode and navigate to your project folder using the integrated terminal.

`code .`

3. **Create Virtual Environment:** Set up a virtual environment for your project using conda.
`conda create -p venv python==3.8 -y`
`conda activate venv`

4. **Initialize Git:** Initialize a Git repository in your project folder.
`git init`

5. **Set Up GitHub Repository:** Create a new GitHub repository for your project. Follow GitHub's instructions to create a remote repository.


6. **Create `setup.py`:** Create a setup.py file for your project. This file will define project metadata and dependencies.
```python
from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    This function will return the list of requirements.
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name='CUSTOMER CHURN HUBPAY',
    version='0.0.1',
    author='Frank',
    author_email='enendufrankc@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
```

7. **Create `requirements.txt`:** Generate a requirements.txt file listing all the project dependencies.
```python
pandas
numpy
matplotlib
seaborn
scikit-learn
-e .

8. **Create `src` Folder:** Create a "src" folder in your project directory. This folder will contain your project's source code.

9. **Create `__init__.py`:** Inside the "src" folder, create an empty `__init__.py` file to indicate that it's a Python package.

10. **Push to Git Repository:** Commit your initial project files to the Git repository you created on GitHub.
`git add .`
`git commit -m "Initial commit"`
`git remote add origin <repository_url>`
`git branch -M main`
`git push -u origin main`