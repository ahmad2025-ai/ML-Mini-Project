# ML-Mini-Project

Bundesliga Match Outcome Prediction (2023-24 Season)
Project Overview
This machine learning project aims to predict match outcomes in the Bundesliga 2023-24 season. The target variable contains three possible results:

-1: Home team loss

0: Draw

1: Home team win

The project involves both classification (predicting match outcomes) and regression (predicting goal margins) tasks.

Key Features and Methodology
Data Preparation
Class Balancing: Used SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance, particularly for draw results (class 0)

Feature Engineering:

Created attack efficiency metrics (goals scored per shot)

Developed defense efficiency metrics (goals conceded per defensive action)

Added last 5 match results as rolling features

Calculated various team form indicators

Classification Models (Match Outcome Prediction)
Evaluated multiple models including:

Logistic Regression

Random Forest

Support Vector Machines

Gradient Boosting

Best Performing Model: Logistic Regression

Hyperparameter Tuning: Used GridSearchCV with 5-fold cross-validation to optimize model performance

Regression Models (Goal Margin Prediction)
Predicted goal difference (e.g., -5 means home team lost by 5 goals)

Tested various regression models:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Best Performing Model: Linear Regression

Results
The Logistic Regression model achieved the best performance for match outcome prediction with an accuracy of 71%. For goal margin prediction, Linear Regression performed best with an R-squared value of 0.703.

Presentation
For more details about the project methodology and results, please see the project presentation. https://docs.google.com/presentation/d/1GTZYJrl1Zd23pMj9iiQGjjm6iqfokI4KmUy9gksQRCY/edit?pli=1&slide=id.g361fd1a114b_1_0#slide=id.g361fd1a114b_1_0

Requirements
Python 3.x

Libraries: pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn

How to Run
Install dependencies: pip install -r requirements.txt

Run the preprocessing script: python src/preprocessing.py

Train and evaluate models: python src/modeling.py

Future Work
Incorporate player-level statistics

Add real-time betting odds data

Implement more advanced feature engineering techniques