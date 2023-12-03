# Weekday Time Window Imputation for Car Accident Data 

Welcome to the Weekday Time Window Imputation for Car Accident Data project! This comprehensive project repository offers an in-depth analysis and modeling of car accident data for Davidson County, spanning from 2015 to 2022. The repository is structured into several key directories, each serving a specific purpose in the data analysis and modeling process:

## Directories

- `00_Raw_Data`: This directory houses CSV files for each quarter from 2015 to 2022, encompassing a total of 279,707 car accident observations in Davidson County. For additional reference and troubleshooting, a dataset pertaining to New York City car accidents is also included.

- `01_Exploratory_Data_Analysis`: This section delves into a detailed examination of the car accident data. It includes analyses of the various columns, distributions of different variables, and comparative studies between observations with missing timestamp information and those without.

- `02_Feature_Engineering`: In this directory, the focus is on preprocessing the data to facilitate the development of various predictive models. It includes the preparation of training, validation, and testing datasets for models like KNN, Matrix Factorization, XGBoost, and MLP.

- `03_Data_for_Modeling`: This folder contains the prepared train, validation, and test datasets, each tailored for the specific requirements of the different models explored in this project.

- `04_Modeling`: Here, you will find the modeling code for each of the implemented models. Additionally, each modeling file has train and validation loss plots, train and validation accuracy plots, and confusion matrix plots for validation and test sets.

## Hyperparameters Adjustment

If you would like to experiment with different hyperparameters, you can adjust the following:

- `KNN.py`: line 34 `k_value`
- `matrix_factorization.py`: line 25 `learning_rate`, `rank`, `max_iters`, `shrinkage_value`
- `xg_boost.ipynb`: `params` dictionary with `colsample_bylevel`, `colsample_bytree`, `learning_rate`, `max_depth`, `n_estimators`, `subsample`
- `mlp.py`: line 48 `# hidden layers` and `# of nodes in each hidden layer`, line 68 `learning rate` and `optimizer`, line 70 `# epochs`, line 71 `batch size`

## Troubleshooting

- `05_Troubleshooting`: This directory provides an exploratory data analysis and feature engineering file specifically for the New York City car accident dataset. It also includes a modeling file implementing an MLP.

## Environment Setup

Lastly, an `environment.yml` file is available to recreate the development environment used for this project.

