# LINEAR-RANDOMFOREST-XGBOOST

Housing Price Prediction Project
This project demonstrates how to build machine learning pipelines for housing price prediction using popular regression algorithms. The project includes implementations for both an XGBoost model and a Random Forest model, as well as a Linear Regression model with proper preprocessing.

Overview
The goal of this project is to predict the median value of owner-occupied homes (MEDV) using the Boston housing dataset. The project uses the following techniques:

Data Preprocessing:

Renaming columns to lowercase for consistency.
Handling missing values using mean imputation.
Standardizing features using StandardScaler.
Modeling:

XGBoost Pipeline: A pipeline that standardizes the data and trains an XGBoost regressor.
Random Forest Pipeline: A pipeline that standardizes the data and trains a Random Forest regressor.
Linear Regression Pipeline: A pipeline that includes imputation, scaling, and trains a Linear Regression model.
Model Evaluation:

The project evaluates model performance using metrics such as Mean Squared Error (MSE) and the R² score.

Pipeline Persistence:

All models are saved as joblib artifacts, allowing for pipeline reloading and making predictions on new data.

Project Structure
data.csv:

The dataset file containing the Boston housing data.

Python Scripts / Notebooks:

The code is organized into modular sections for preprocessing, modeling, evaluation, and pipeline persistence.

Saved Pipelines:

The pipelines for each model are saved as joblib files (e.g., xgb_pipeline.joblib, rf_pipeline.joblib, lr_pipeline.joblib), which can be loaded later for predictions.

How to Run
Install Dependencies:

The project requires common data science libraries such as pandas, numpy, scikit-learn, xgboost, and joblib. You can install these dependencies using pip:

 pip install pandas numpy scikit-learn xgboost joblib  
Running the Code:

The code can be executed in a Jupyter Notebook or from a Python script. It is structured into:

Data preprocessing
Train-test splitting and scaling
Training models and saving/loading pipelines
Making new predictions with the loaded pipelines
Making Predictions:

After training, you can load any pipeline (e.g., lr_pipeline.joblib) and use it to predict the housing price for new data.

Example Pipelines
Linear Regression Pipeline
Steps:
Impute missing values using the mean strategy.
Standardize features with StandardScaler.
Train a Linear Regression model.
Usage:
The pipeline is trained on the training data and evaluated on the test set. A new prediction is made using the average of feature values.
Random Forest and XGBoost Pipelines
Steps:
Preprocess the data (imputation, scaling).
Train the corresponding regression model.
Usage:
These pipelines follow similar steps and help compare the performance between ensemble methods and linear models.
Notes
Data Cleaning:

The project includes a data cleaning and preprocessing step to ensure that missing values are handled.

Reproducibility:

Setting a random state in train-test split and model initialization ensures reproducible results.

Model Expansion:

You can easily extend this project by adding new preprocessing steps, experimenting with other regression models, or performing hyperparameter tuning.

Conclusion
This project provides a clear demonstration of how to build, evaluate, and persist machine learning pipelines for regression tasks using Python. The modular design and use of pipelines ensure that your workflow is both reproducible and scalable.

Feel free to customize this README.md to better fit your project's details and any additional features you wish to document.
