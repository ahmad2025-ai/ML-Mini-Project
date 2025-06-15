#!/usr/bin/env python
# coding: utf-8

# In[80]:


#Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# In[84]:


import pandas as pd
df=pd.read_csv('/Users/ahmadkhalilghamai/Downloads/updated_last5_stats.csv')
df


# In[9]:


df['goals_difference'] = df['goals_for'] - df['goals_against']


# ### EDA Analysis

# In[11]:


df.columns


# In[13]:


df.shape


# In[15]:


df.isnull().sum()


# In[17]:


df.nunique()


# In[19]:


df.dtypes


# In[25]:


import matplotlib.pyplot as plt

# Plot distribution of goals_difference
plt.figure(figsize=(8, 5))  # Single plot, slightly wider

plt.hist(df["goals_difference"], bins=20, color="green", alpha=0.7)
plt.title("Distribution of Goals Difference (Goals For - Goals Against)")
plt.xlabel("Goal Difference")
plt.ylabel("Frequency")

# Add a vertical line at 0 to show neutral difference
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)

plt.tight_layout()
plt.show()


# ### Data Cleaning and Wrangling

# In[21]:


df = df.drop(['started', 'finished','is_win','is_loss','is_draw', 'cancelled', 'rating_diff', 'points_diff', 'round', 'team', 'opponent','goals_for', 'goals_against'], axis=1)


# In[23]:


import pandas as pd

# Fill missing values with column mean
df = df.fillna(df.mean())

# Fill all numerical columns with their means
df.fillna(df.mean(numeric_only=True), inplace=True)


# ### Features/Target 

# In[27]:


features = df.drop(columns=['goals_difference'])  # Features (X)
target = df[['goals_difference']]                # Targets (y - DataFrame)


# ###  Test/Train split

# In[29]:


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


# ### Scaling the features 
# 

# In[33]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # fit and transform in one step.
X_test_scaled = scaler.transform(X_test)


# In[35]:


# The step above returns a numpy array, if you want to convert it back to a DataFrame:
X_train_scal = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scal = pd.DataFrame(X_test_scaled, columns=X_test.columns)


# ### Linear Regression+Gridsearch

# In[37]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



# Define Ridge regression and param grid
ridge = Ridge()
param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}

# GridSearchCV
grid = GridSearchCV(ridge, param_grid, scoring='neg_mean_squared_error', cv=5)
grid.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = grid.predict(X_test_scaled)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("✅ Best Parameters:", grid.best_params_)
print(f"📉 Mean Squared Error: {mse:.4f}")
print(f"📈 R² Score: {r2:.4f}")


# ### Random Forest Regressor + Gridsearch

# In[39]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score



# Random Forest Regressor with Grid Search
rf = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)  # Use unscaled data here for RF

# Predict
y_pred = grid_search.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("✅ Best Parameters Found:", grid_search.best_params_)
print(f"📉 Mean Squared Error: {mse:.4f}")
print(f"📈 R² Score: {r2:.4f}")


# ### KNN Regressor + Gridsearch

# In[40]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score



# Initialize KNN regressor
knn = KNeighborsRegressor()

# Grid search parameters
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # 1 = Manhattan, 2 = Euclidean
}

grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Predict
y_pred = grid_search.predict(X_test_scaled)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("✅ Best Parameters Found:", grid_search.best_params_)
print(f"📉 Mean Squared Error: {mse:.4f}")
print(f"📈 R² Score: {r2:.4f}")


# ### AdaBoost

# In[43]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score



# Define AdaBoost with a shallow Decision Tree
base_tree = DecisionTreeRegressor(max_depth=3, random_state=42)
ada = AdaBoostRegressor(estimator=base_tree, random_state=42)  # <- Fixed here

# Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0]
}

grid_search = GridSearchCV(ada, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Predictions and Evaluation
y_pred = grid_search.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output
print("✅ Best Parameters Found:", grid_search.best_params_)
print(f"📉 Mean Squared Error: {mse:.4f}")
print(f"📈 R² Score: {r2:.4f}")


# ### Decision Tree Regressor

# In[45]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score



# Decision Tree Regressor
dtree = DecisionTreeRegressor(random_state=42)

# Grid Search for hyperparameter tuning
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(dtree, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Prediction and evaluation
y_pred = grid_search.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output
print("✅ Best Parameters Found:", grid_search.best_params_)
print(f"📉 Mean Squared Error: {mse:.4f}")
print(f"📈 R² Score: {r2:.4f}")


# In[47]:


pip install lightgbm catboost


# ### All Models Together

# In[49]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# ------------------- Step 1: Prepare Data -------------------
features = df.drop(columns=['goals_difference'])
target = df['goals_difference']

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------- Step 2: Train Models -------------------

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# KNN
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
mse_knn = mean_squared_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)

# AdaBoost
ada = AdaBoostRegressor(random_state=42)
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)
mse_ada = mean_squared_error(y_test, y_pred_ada)
r2_ada = r2_score(y_test, y_pred_ada)

# Decision Tree
dtree = DecisionTreeRegressor(random_state=42)
dtree.fit(X_train, y_train)
y_pred_dtree = dtree.predict(X_test)
mse_dtree = mean_squared_error(y_test, y_pred_dtree)
r2_dtree = r2_score(y_test, y_pred_dtree)

# Gradient Boosting
gb = GradientBoostingRegressor(
    learning_rate=0.05, n_estimators=100, max_depth=3, subsample=1.0, random_state=42)
gb.fit(X_train_scaled, y_train)
y_pred_gb = gb.predict(X_test_scaled)
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

# LightGBM
lgb = LGBMRegressor(random_state=42)
lgb.fit(X_train, y_train)
y_pred_lgb = lgb.predict(X_test)
mse_lgb = mean_squared_error(y_test, y_pred_lgb)
r2_lgb = r2_score(y_test, y_pred_lgb)

# CatBoost
cat = CatBoostRegressor(verbose=0, random_state=42)
cat.fit(X_train, y_train)
y_pred_cat = cat.predict(X_test)
mse_cat = mean_squared_error(y_test, y_pred_cat)
r2_cat = r2_score(y_test, y_pred_cat)

# ------------------- Step 3: Comparison Table -------------------
results = pd.DataFrame([
    ["Linear Regression", mse_lr, r2_lr],
    ["Random Forest", mse_rf, r2_rf],
    ["KNN", mse_knn, r2_knn],
    ["AdaBoost", mse_ada, r2_ada],
    ["Decision Tree", mse_dtree, r2_dtree],
    ["Gradient Boosting", mse_gb, r2_gb],
    ["LightGBM", mse_lgb, r2_lgb],
    ["CatBoost", mse_cat, r2_cat],
], columns=["Model", "MSE", "R2 Score"])

# Sort by R2 Score descending
results = results.sort_values(by="R2 Score", ascending=False).reset_index(drop=True)

# Show results
print("\n📊 Regression Model Comparison:")
print(results)


# In[50]:


results


# ### R2 Score comparison

# In[51]:


import seaborn as sns
import matplotlib.pyplot as plt

# Bar chart for R2 Scores
plt.figure(figsize=(10, 6))
sns.barplot(x='R2 Score', y='Model', data=results, palette='viridis')
plt.title('R² Score Comparison Across Models')
plt.xlabel('R² Score')
plt.ylabel('Model')
plt.xlim(0, 1)
plt.tight_layout()
plt.show()


# ### Linear Regression: Actual vs Predicted

# In[59]:


plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred_lr)  # change model here
plt.xlabel("Actual Goals Difference")
plt.ylabel("Predicted")
plt.title("Linear Regression: Actual vs Predicted")
plt.axline((0, 0), slope=1, color='red', linestyle='--')
plt.show()


# In[76]:


import matplotlib.pyplot as plt

# Model names and corresponding R2 scores
models = [
    "Linear Regression", "Gradient Boosting", "LightGBM", 
    "Random Forest", "AdaBoost", "KNN", "Decision Tree"
]
r2_scores = [0.703, 0.672, 0.671, 0.644, 0.626, 0.537, 0.382]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(models, r2_scores, marker='o', color='tab:blue', linewidth=2)

# Annotate points
for i, score in enumerate(r2_scores):
    plt.text(i, score + 0.01, f"{score:.3f}", ha='center', fontsize=9)

# Styling
plt.title("Comparison of R² Scores by Model")
plt.xlabel("Model")
plt.ylabel("R² Score")
plt.ylim(0.3, 0.75)
plt.grid(True)
plt.xticks(rotation=30)
plt.tight_layout()

plt.show()


# In[ ]:




