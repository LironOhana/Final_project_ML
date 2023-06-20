import pandas as pd
import re
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder, PolynomialFeatures, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import json
from sklearn.metrics import mean_squared_error

import requests
from madlan_data_prep import prepare_data



# In[24]:


url = "C:/Users/liron/OneDrive - Ariel University/Toar 1/third year/Yeda&Netunim/Jupyter/output_all_students_Train_v10.xlsx"
data = pd.read_excel(url)
data = data.drop(153)
data = data.dropna(subset=['price']).copy()

data = data[data['price'].apply(lambda x: bool(re.search(r'\d', str(x))))]
data['price'] = data['price'].apply(lambda x: str(x).replace(',', ''))
data['price'] = data['price'].apply(lambda x: re.findall(r'\d+', str(x))[0] if re.findall(r'\d+', str(x)) else print(x))
data['price'] = data['price'].apply(lambda x: float(x) if isinstance(x, str) and x.isdigit() else None)
scaler =StandardScaler()
data['price_normalized'] = scaler.fit_transform(data[['price']])

data['description '] = data['description '].astype(str)
data['description '] = data['description '].replace(',', '', regex=True)


data = data.dropna(subset=['price']).copy()

X = data[['City','city_area','type', 'room_number', 'Area',  'hasElevator ', 'hasParking ', 'hasBars ', 'hasStorage ',
          'condition ', 'hasAirCondition ', 'hasBalcony ', 'hasMamad ', 'handicapFriendly ', 'entranceDate ',
          'furniture ', 'floor_out_of']]
# y = data['price']
y = data['price_normalized']




# Define the number of folds for cross-validation
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)

rmse_scores = []
r2_scores = []

# Perform K-fold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
    y_train, y_test = y.iloc[train_index].copy(), y.iloc[test_index].copy()
#     print(X_train)
    X_train = prepare_data(X_train)
    scaler = StandardScaler()
    y_train = np.reshape(y_train, (-1, 1))
    y_train = scaler.fit_transform(y_train)
    
    # Train an Elastic Net model on the raw X_train
    elasticnet = ElasticNet(alpha=0.05, l1_ratio=0.3)
    elasticnet.fit(X_train, y_train)
    
    X_test = prepare_data(X_test)
    # Predict on the test set
    y_pred = elasticnet.predict(X_test)
    y_pred = y_pred.reshape(-1, 1)
    y_pred = scaler.inverse_transform(y_pred)
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_scores.append(rmse)

    # Calculate R-squared score

    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)
    print("ok")
# Calculate average RMSE
average_rmse = np.mean(rmse_scores)
print("Average RMSE:", average_rmse)

# Calculate average R-squared score
average_r2 = np.mean(r2_scores)
print("Average R-squared:", average_r2)


# In[35]:


# Find the best fold based on the lowest RMSE
best_fold_index = np.argmin(rmse_scores)

# Get the corresponding train and test indices for the best fold
best_train_index, best_test_index = list(kf.split(X))[best_fold_index]
X_train, X_test = X.iloc[best_train_index].copy(), X.iloc[best_test_index].copy()
y_train, y_test = y.iloc[best_train_index].copy(), y.iloc[best_test_index].copy()

# Prepare the final training data
X_train = prepare_data(X_train)
# y_train = prepare_data(y_train)
best_model = ElasticNet(alpha=0.05, l1_ratio=0.3)
# Train the final model using the best fold
best_model.fit(X_train, y_train)


# In[36]:


joblib.dump(best_model,'trained_model.pkl')

