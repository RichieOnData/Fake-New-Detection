#!/bin/bash

ENV_NAME=fakenews
python -m venv $ENV_NAME
source $ENV_NAME/bin/activate

pip install transformers matplotlib ipykernel
pip install pandas
pip install nltk
pip install seaborn tqdm
pip install scikit-learn
pip install xgboost

python -m ipykernel install --user --name $ENV_NAME  #Register the env as a kernal for using it with jupyter notebook