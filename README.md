# Coursework 1

Coursework 1 contains two parts: Regression and Clustering.

The README.txt contains details of the source code. This file is purely for set up and installation and running the code. 

## Installation

A virtualenv with Python3 is used for running all the sourcecode. 

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.

```bash
pip install -r requirements.txt
```

The following packages can be seen on the requirements.txt file:
```bash
matplotlib==3.3.4
numpy==1.19.5
pandas==1.2.2
scikit-learn==0.24.1
scipy==1.6.0
sklearn==0.0
yellowbrick==1.3.post1
```

## Directory Structure

```
Coursework
├── code
│    └── regression.py
│    └── clustering.py
├── data
│    └── adult.csv
│    └── wholesale_customers.csv
│── plots 
│── README.md
│── README.txt
│── Report.pdf
│__ requirements.txt 
```

The .csv files need to be put in the data folder. 
## Running regression.py

Set the following constants, and vary them if you wish to experiment. 

```python
DATA_FOLDER = "../data"
FILE_NAME = "adult.csv"
USE_SPLIT = 0.25
```
Run the command line argument:

```
python regression.py
```

## Running clustering.py

Set the following constants, and vary them if you wish to experiment. 

```python
DATA_FOLDER = "../data"
FILE_NAME = "wholesale_customers.csv"
PLOT_FOLDER = "../plots"
```
Run the command line argument:

```
python clustering.py
```

The plots can be found in the plots folder. 
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.