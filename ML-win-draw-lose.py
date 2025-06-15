#!/usr/bin/env python
# coding: utf-8

# In[12]:


#Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# In[16]:


import pandas as pd
df=pd.read_csv('/Users/ahmadkhalilghamai/Downloads/etc/updated_last5_stats.csv')
df.head()


# ## EDA Analysis

# In[19]:


df.shape


# In[21]:


df.columns


# In[25]:


df.dtypes


# In[27]:


df.nunique()


# In[29]:


df.describe()


# ## Omiting unnecessary columns

# In[33]:


df = df.drop(['started', 'finished','is_win','is_loss','is_draw', 'cancelled', 'rating_diff', 'points_diff', 'round', 'team', 'opponent','goals_for', 'goals_against'], axis=1)


# ## dealing with nulls

# In[36]:


# Fill missing values with column mean
df = df.fillna(df.mean())

# Fill all numerical columns with their means
df.fillna(df.mean(numeric_only=True), inplace=True)


# In[38]:


df.isnull().sum()


# ## Target column class distribution

# In[41]:


import matplotlib.pyplot as plt  # Add this import at the top

import matplotlib.pyplot as plt

# Your existing code with color fix
distribution = df["result"].value_counts()
distribution.plot(kind="bar", color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])

# Customize the plot
plt.title("Distribution of Results")
plt.xlabel("Result Categories")
plt.ylabel("Count")
plt.xticks(rotation=45)  # Rotate x-axis labels if needed

plt.tight_layout()
plt.show()
print('result')


# ##  Splitting the data into features and target variable

# In[44]:


# The original 'Date' column can't be used directly as a feature, as ML models require numerical input.
features= df.drop(columns =['result'])
target = df['result']


# ## Test/Train split

# In[47]:


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


# ## Scaling

# In[50]:


scaler = StandardScaler()


# In[52]:


# Scaling the features 
X_train_scaled = scaler.fit_transform(X_train) # fit and transform in one step.
X_test_scaled = scaler.transform(X_test)


# In[54]:


X_train_scal = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scal = pd.DataFrame(X_test_scaled, columns=X_test.columns)


# In[56]:


pip install imbalanced-learn


# ## Logistic Classifier+SMOTE+GridSearch

# In[58]:


# Create pipeline


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Step 1: Split Data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Step 2: Apply SMOTE separately for visualization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

# Step 3: Plot class distribution after SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Plot class distribution after SMOTE
plt.figure(figsize=(6, 4))
sns.countplot(x=y_resampled, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])  # Default matplotlib blue

# Customize the plot
plt.title("Distribution of Results After SMOTE", fontsize=14)
plt.xlabel("Result Categories", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# Step 4: Define Pipeline and GridSearch
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('logreg', LogisticRegression(max_iter=1000, solver='saga', multi_class='multinomial'))
])

param_grid = {
    'logreg__C': [0.01, 0.1, 1, 10],
    'logreg__penalty': ['l2']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = grid_search.predict(X_test)
print("Best Parameters:", grid_search.best_params_)
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# ## Decision Tree

# In[62]:


from sklearn.tree import DecisionTreeClassifier

pipeline_dt = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('dt', DecisionTreeClassifier(random_state=42))
])

param_grid_dt = {
    'dt__max_depth': [3, 5, 10, None],
    'dt__min_samples_split': [2, 5, 10]
}

grid_search_dt = GridSearchCV(pipeline_dt, param_grid_dt, cv=5, scoring='f1_macro', n_jobs=-1)
grid_search_dt.fit(X_train, y_train)

y_pred_dt = grid_search_dt.predict(X_test)
print("📌 Decision Tree - Best Parameters:", grid_search_dt.best_params_)
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt))


# ## KNN Model

# In[65]:


from sklearn.neighbors import KNeighborsClassifier

pipeline_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('knn', KNeighborsClassifier())
])

param_grid_knn = {
    'knn__n_neighbors': [3, 5, 7, 9],
    'knn__weights': ['uniform', 'distance']
}

grid_search_knn = GridSearchCV(pipeline_knn, param_grid_knn, cv=5, scoring='f1_macro', n_jobs=-1)
grid_search_knn.fit(X_train, y_train)

y_pred_knn = grid_search_knn.predict(X_test)
print("📌 KNN - Best Parameters:", grid_search_knn.best_params_)
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn))


# ## Random Forest

# In[68]:


from sklearn.ensemble import RandomForestClassifier

pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42))
])

param_grid_rf = {
    'rf__n_estimators': [50, 100],
    'rf__max_depth': [5, 10, None],
    'rf__min_samples_split': [2, 5]
}

grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=5, scoring='f1_macro', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

y_pred_rf = grid_search_rf.predict(X_test)
print("📌 Random Forest - Best Parameters:", grid_search_rf.best_params_)
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))


# ## Ada Boost 

# In[70]:


from sklearn.ensemble import AdaBoostClassifier

pipeline_ada = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('ada', AdaBoostClassifier(random_state=42))
])

param_grid_ada = {
    'ada__n_estimators': [50, 100, 200],
    'ada__learning_rate': [0.5, 1.0, 1.5]
}

grid_search_ada = GridSearchCV(pipeline_ada, param_grid_ada, cv=5, scoring='f1_macro', n_jobs=-1)
grid_search_ada.fit(X_train, y_train)

y_pred_ada = grid_search_ada.predict(X_test)
print("📌 AdaBoost - Best Parameters:", grid_search_ada.best_params_)
print("\nClassification Report:\n", classification_report(y_test, y_pred_ada))


# ## Models Camprision

# In[72]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd

# --------------------- Logistic Regression ---------------------
pipe_logreg = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('logreg', LogisticRegression(max_iter=1000, solver='saga', multi_class='multinomial'))
])
param_logreg = {'logreg__C': [0.01, 0.1, 1], 'logreg__penalty': ['l2']}
grid_logreg = GridSearchCV(pipe_logreg, param_logreg, cv=5, scoring='f1_macro', n_jobs=-1)
grid_logreg.fit(X_train, y_train)
y_pred_logreg = grid_logreg.predict(X_test)

# --------------------- AdaBoost ---------------------
pipe_ada = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('ada', AdaBoostClassifier(random_state=42))
])
param_ada = {'ada__n_estimators': [50, 100], 'ada__learning_rate': [0.5, 1.0]}
grid_ada = GridSearchCV(pipe_ada, param_ada, cv=5, scoring='f1_macro', n_jobs=-1)
grid_ada.fit(X_train, y_train)
y_pred_ada = grid_ada.predict(X_test)

# --------------------- KNN ---------------------
pipe_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('knn', KNeighborsClassifier())
])
param_knn = {'knn__n_neighbors': [3, 5, 7]}
grid_knn = GridSearchCV(pipe_knn, param_knn, cv=5, scoring='f1_macro', n_jobs=-1)
grid_knn.fit(X_train, y_train)
y_pred_knn = grid_knn.predict(X_test)

# --------------------- Random Forest ---------------------
pipe_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42))
])
param_rf = {'rf__n_estimators': [100], 'rf__max_depth': [5, 10]}
grid_rf = GridSearchCV(pipe_rf, param_rf, cv=5, scoring='f1_macro', n_jobs=-1)
grid_rf.fit(X_train, y_train)
y_pred_rf = grid_rf.predict(X_test)

# --------------------- Decision Tree ---------------------
pipe_dt = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('dt', DecisionTreeClassifier(random_state=42))
])
param_dt = {'dt__max_depth': [3, 5, 10]}
grid_dt = GridSearchCV(pipe_dt, param_dt, cv=5, scoring='f1_macro', n_jobs=-1)
grid_dt.fit(X_train, y_train)
y_pred_dt = grid_dt.predict(X_test)

# --------------------- Model Comparison ---------------------
model_predictions = {
    'Logistic Regression': y_pred_logreg,
    'AdaBoost': y_pred_ada,
    'KNN': y_pred_knn,
    'Random Forest': y_pred_rf,
    'Decision Tree': y_pred_dt
}

results = []
for model_name, y_pred in model_predictions.items():
    results.append({
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision (macro)': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'Recall (macro)': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'F1 Score (macro)': f1_score(y_test, y_pred, average='macro', zero_division=0)
    })

comparison_df = pd.DataFrame(results)
print("📊 Model Comparison:\n")
print(comparison_df.sort_values(by='F1 Score (macro)', ascending=False).reset_index(drop=True))


# In[73]:


comparison_df


# ## confusion Matrix

# In[75]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred_logreg)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                             display_labels=grid_logreg.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()


# In[81]:


get_ipython().run_line_magic('History', '-f main1.py')


# In[ ]:




