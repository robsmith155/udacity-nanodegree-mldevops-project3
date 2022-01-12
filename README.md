# udacity-nanodegree-mldevops-project3
Third project of [Udacity ML DevOps Engineer nanodegree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821): Deploying a Machine Learning Model on Heroku with FastAPI

## Overview
The main goal of this project is to develop skills in the following tools:
- [DVC](https://dvc.org/): Open source version control system for ML projects. Ability to track data and models.
- [FastAPI](https://fastapi.tiangolo.com/): Fast web framework for building APIs with Python.
- [Heroku](https://id.heroku.com/login): Platform as a service (PaaS) that allows free deployment of applications in the cloud.

### Dataset
Here we will be working with the [US census dataset](https://archive.ics.uci.edu/ml/datasets/census+income). The goal is to predict whether a person earns more than $50k based on certain census information (such as age, gender, education and race).

### EDA
Some basic analysis of the raw data is contained in the `EDA` notebook in the `notebooks` folder. Note that most of the output is not visible from `pandas-profiling`. You will need to open the notebook and rerun it to see the output.

### Data cleaning and processing
The code for the taking he raw data and performing basic data cleaning was developed in the `clean-data.ipynb` notebook. The code was then productionized in the `mlp` package under the processing submodule. The code for preparing the data for training was provided by Udacity and is included in the same module.

### Modeling
Finding the absolute best model was not the main goal of the project. However, I ran some analysis and a hyperparameter search for a Random Forest classifier. This can be found in the `notebooks` folder.

## Instructions
Below you will find instructions on how you can rerun things in the repository.

### Clone repo
Run the following to clone this repository to your local directory:

```
git clone https://github.com/robsmith155/udacity-nanodegree-mldevops-project3.git
```

### Python environment
To create the Python virtual environment, run the following (assumind Conda is installed):

```
conda env create -f environment.yml
```

### Setup pre-commit
We will use [pre-commit](https://pre-commit.com/) to run Git hook scripts on every commit to automatically run code linting and checking. Here we use [isort](https://pycqa.github.io/isort/) to automatically sort imports, [black](https://black.readthedocs.io/en/stable/) to style the code and [flake8](https://flake8.pycqa.org/en/latest/) to check the code.

Pre-commit will set up these Git hooks for you. Change directory to the root of the repo. Make sure that the virtual environment is active and run:

```
pre-commit install
```

Now when you make a Git commit it should run the Git hooks for isort, black and flake8. Note that this will only check the files that are being committed. If any changes are made to these, the Git commit will not be completed and you will need to add the changed files again and commit

Alternatively you can run the hooks on all files prior to staging and commiting as follows:

```
pre-install run --all
```

### Pytest
To run all tests in the repo, simply run:

```
pytest -vv
```

### Data
In the project conducted for Udacity the raw data and all subsequent outputs and models were stored in an AWS S3 bucket. However, you will not have access to this so you need to download the raw data from the [UCI](https://archive.ics.uci.edu/ml/datasets/census+income) website and put it in the `data` folder. You can then rerun the pipeline detailed in the next section.

### DVC

#### DVC remote
In this project I used an S3 bucket provided by Udacity to store the data and models. However, this will likely be deleted after completing the course and nobody else has access to it.

Instead, you will need to create your own remote storage for DVC. This can just be a local directory. To do this:

```
dvc remote add -d localremote <PATH_OF_LOCAL_FOLDER>
```

#### Rerun pipeline
Now you can rerun the whole pipeline taking us from raw data to trained model using the DVC pipeline contained in `dvc.yaml`. Run:

```
dvc repro
```

### FastAPI
Here we use FastAPI to create a web framework for deploying the model. 

#### Run locally
To run the app locally, you can serve the app as follows from the root of the repo:

```
uvicorn app:app --reload
```

Note that the `--reload` argument means that uvicorn will automatically redeploy if you make changes to the code. By default, the route of the app will be available at `http://127.0.0.1:8000/`. To see the FastAPI docs use 

#### Heroku deployment
The app has been deployed to Heroku. You can find the app at: https://robsmith155-salary-prediction.herokuapp.com/

The FastAPI docs can be accessed [here](https://robsmith155-salary-prediction.herokuapp.com/docs).

An example of sending a POST request to the deployed model on Heroku can be found in the `example_heroku_request.py` file in the repo. With the virtual environment active, just run:

```
python example_heroku_request.py
```
