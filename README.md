# Regression

- Repository: `https://github.com/olhasl/challenge-regression`
- Type of Challenge: `Consolidation`
- Duration: `6 days`
- Deadline: `09/12/2024 16:00`
- Team challenge : Solo

## Table of Contents
1. [Project Description](#project-description)
2. [Learning Objectives](#learning-objectives)
3. [Installation](#installation)
4. [Usage](#usage)

## Project Description

This is the third phase of a large project for predicting real estate sales prices in Belgium. The current task is to to build a machine learning (ML) prediction model based on the CatBoost model.

The data prepared in the previous phases (**scraped**, **preprocessed** and **analyzed**) "./input_data/ready_initial_data.csv" was used to build the model, and some additional features collected from open sources were added at the preprocessing step.

The final data for training the model is in the folder "./output_data/data_for_model.csv", and the evaluation data of the resulting model is also in the folder "output_data".

## Learning Objectives
The project is designed to:
- Be able to apply a linear regression or a regression model in a real context.
- Be able to preprocess data for machine learning:
        - handle NANs
        - handle categorical data
        - select features
        - remove features that have too strong correlation, etc.
- Utilize object-oriented programming (OOP) concepts effectively
- Create clean, reusable code by applying good OOP principles
- Organize code with imports and project structure

## Installation
1. Clone the repository: ```https://github.com/olhasl/challenge-regression```
2. Install dependencies: 
  - ```Python 3.12.4```  
  - ```pip install pandas numpy matplotlib.pyplot os catboost sklearn shap time```

## Usage
- In code folder you will find to files: 1_adding_features.py and 2_modelling.py. Run the codes of these files one by one.
- In the output_data folder you can find two files 'model_metrics.csv' and 'features_importance.csv' with main results of the model accuracy. Visualizations of some of the model results with graphs can be found in the same folder.
- In the file Linear_regression.ipynb you can find some analysis and visualizations that helped determine the features for the model (e.g. correlation matrix, etc.).
