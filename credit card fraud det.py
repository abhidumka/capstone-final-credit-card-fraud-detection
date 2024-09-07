#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd


# ## 1. Load the Dataset

# In[33]:


# Load the dataset from a CSV file
df=pd.read_csv("C:\Final_capstone\creditcard.csv")


# In[34]:


# Display the first few rows of the dataframe
print(df.head())


# In[35]:


df.dtypes


# ## 2. Exploratory Data Analysis (EDA)

# ### a. Data Quality Check

# In[36]:


# Check for missing values
print("\nMissing Values Count:")
print(df.isnull().sum())
# Handle missing values (e.g., impute with mean)
df.fillna(df.mean(), inplace=True)


# In[37]:


# Statistical summary
print("\nStatistical Summary:")
print(df.describe())


# In[38]:


# Identify outliers (simple example using Z-score)
from scipy import stats
import numpy as np


# In[39]:


# Calculate Z-scores
z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
outliers = (z_scores > 3).all(axis=1)
print("\nNumber of outliers detected:", np.sum(outliers))


# ## 3. Data Cleaning

# ### a. Handling Missing Values

# In[16]:


# Option 1: Fill missing values with the mean (or other strategies)
df.fillna(df.mean(), inplace=True)

# Option 2: Drop rows with missing values
df.dropna(inplace=True)


# ### b. Standardization

# In[17]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.select_dtypes(include=[np.number]))
df_scaled = pd.DataFrame(scaled_features, columns=df.select_dtypes(include=[np.number]).columns, index=df.index)


# In[22]:


print(df.columns)


# ## 4. Feature Engineering and Selection

# In[28]:


# Example of feature transformation
df['log_Amount'] = np.log1p(df['Amount'])

# Example of feature creation
df['V1_V2_ratio'] = df['V1'] / (df['V2'] + 1e-8)  # Avoid division by zero


# ## 5. Model Selection

# In[31]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report



# In[40]:


import sklearn
print("scikit-learn version:", sklearn.__version__)


# In[41]:


from sklearn.model_selection import train_test_split


# In[45]:


from sklearn.linear_model import LogisticRegression


# In[46]:


from sklearn.metrics import classification_report


# In[51]:


from imblearn.over_sampling import SMOTE
X = df.drop(columns=['Class'])
y = df['Class']
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)
# Create a new DataFrame with resampled data
resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
resampled_df['Class'] = y_resampled


# In[47]:


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)


# In[52]:


# Initialize models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name} Classification Report:")
    print(classification_report(y_test, y_pred))


# ### 6. Model_Training

# In[53]:


# Train the chosen model (e.g., Random Forest)
best_model = RandomForestClassifier()
best_model.fit(X_train, y_train)


# ## 7. Model Validation

# In[54]:


from sklearn.metrics import confusion_matrix, roc_auc_score

y_pred = best_model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
print(f"ROC AUC Score: {roc_auc}")


# ## 8. Model Deployment

# In[55]:


import joblib

# Save the model
joblib.dump(best_model, 'model.pkl')

# Load the model for prediction
loaded_model = joblib.load('model.pkl')


# In[ ]:




