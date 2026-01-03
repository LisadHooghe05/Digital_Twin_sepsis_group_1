# Project digital twins sepsis AKI
To be able to run the code, you must meet the requirements listed in the sections below. After following the steps in the README, the packages listed under Python Package Requirements should be installed in your environment. 

### 1. System Requirements
- Operating System: Windows
- Python version: Python 3.12.x(tested with Python 3.12.10)
- Git==2.51.0.windows.1 (for version control)
- Visual Studio Code version 1.107.1

### 2. Python Package Requirements
> **Note:**  After doing all the steps in the README you should see the following list when running pip list in the terminal.

```powershell
contourpy==1.3.2
cycler==0.12.1
fonttools==4.61.1
hdbscan==0.8.41
joblib==1.5.3
kiwisolver==1.4.9
matplotlib==3.10.8
numpy==2.2.6
packaging==25.0
pandas==2.3.3
patsy==1.0.2
pillow==12.1.0
pyparsing==3.3.1
python-dateutil==2.9.0.post0
pytz==2025.2
scikit-learn==1.7.2
scikit-posthocs==0.11.4
scipy==1.15.3
seaborn==0.13.2
six==1.17.0
threadpoolctl==3.6.0
tzdata==2025.3
```
### 3. Installation Requirements
> **Note:**  Detailed instructions for the following steps are provided in the README

1. Install Python 3.12.x (tested with Python 3.12.10)
2. Clone the project repository
3. Create a virtual environment
4. Install the required packages

### 4. Functional Requirements
- The code needs to have access to new made folder that contains the csv files from the zip folder, which contains the MIMIC-IV data. Otherwise, the code will not run.
- The system must load and preprocess datasets using NumPy and Pandas.
- The system must perform clustering using HDBSCAN.
- The system must support statistical analysis using SciPy and scikit-learn.
- The system must support post-hoc statistical testing using scikit-posthocs.

### 5. Non-Functional Requirements
- The system must run on Python 3.12.x
- The system must complete the main code, where there will be clustered to get the wanted results
- The system must produce the same result for everyone who has the same input data

### 6. Constraints
- The programming language is Python
- Only the packages listed above are required

### 7. Notes
- The project is not guaranteed to be able to work with python 3.13 or higher or lower than 3.12
- Package versions are pinned to ensure reproducibility