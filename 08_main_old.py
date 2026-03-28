import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

# Load the data
df = pd.read_csv('housing.csv')


# We will do the stratified sampling
df['category'] = pd.cut(df['median_income'], bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(df, df['category']):
    strat_train_set = df.loc[train_index].drop('category', axis=1)
    strat_test_set = df.loc[test_index].drop('category', axis=1)


# We will work on the copy of the training data
df1 = strat_train_set.copy()


# Seperate the labels and features from the training data
housing_features = df1.drop('median_house_value', axis=1)
housing_labels = df1['median_house_value']


# Separate the numerical and categorical column
housing_features_num_col = housing_features.drop('ocean_proximity', axis=1).columns.tolist()
housing_features_cat_col = ['ocean_proximity']


# Building Pipelines

## Numerical Pipeline

num_pipeline = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler())
])

## Categorical Pipeline
cat_pipeline = Pipeline([
    ('encode', OneHotEncoder())
])


# Full Pipeline
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, housing_features_num_col),
    ('cat', cat_pipeline, housing_features_cat_col)
])


housing_prepared_data = full_pipeline.fit_transform(housing_features)


# Understanding which model is best using Cross Validation

## Linear Regression
lin_reg = LinearRegression()
lin_rmse = -cross_val_score(lin_reg, housing_prepared_data, housing_labels, scoring='neg_root_mean_squared_error', cv=10)
print(type(lin_rmse))
print(pd.Series(lin_rmse).describe())


## Decision Tree
dec_tree_reg = DecisionTreeRegressor()
dec_tree_rmse = -cross_val_score(dec_tree_reg, housing_prepared_data, housing_labels, scoring='neg_root_mean_squared_error', cv=10)
print(pd.Series(dec_tree_rmse).describe())


## Random Forest Regressor
rand_for_reg = RandomForestRegressor()
rand_for_rmse = -cross_val_score(rand_for_reg, housing_prepared_data, housing_labels, scoring='neg_root_mean_squared_error', cv=10)
print(pd.Series(rand_for_rmse).describe())


## Gradient Boost Regressor
grad_boost_reg = GradientBoostingRegressor()
grad_boost_rmse = -cross_val_score(grad_boost_reg, housing_prepared_data, housing_labels, scoring='neg_root_mean_squared_error', cv=10)
print(pd.Series(grad_boost_rmse).describe())