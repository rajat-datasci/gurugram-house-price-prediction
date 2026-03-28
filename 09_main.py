import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = 'model.pkl' # The model file
PIPELINE_FILE = 'pipeline.pkl' # The pipeline file

# Method returning Full Pipeline
def build_pipeline(housing_numerical_attribs, housing_categorical_attribs):

    # Pipeline for numerical columns
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline for categorical columns
    cat_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Full Pipeline
    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, housing_numerical_attribs),
        ('cat', cat_pipeline, housing_categorical_attribs)
    ])

    return full_pipeline


# Will run if the MODEL_FILE doesn't exists
if not os.path.exists(MODEL_FILE):

# Training Phase

    # Read the data
    df = pd.read_csv('housing.csv')


    # Done the stratified Sampling
    # Made the extra column 'income_cat' so that each category can be present in identical proportions in the training and testing sets
    df['income_cat'] = pd.cut(df['median_income'], bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5])

    spliter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    # Divided the data into training and testing sets
    for train_index, test_index in spliter.split(df, df['income_cat']):
        house_train_set = df.loc[train_index].drop('income_cat', axis=1)
        house_test_set = df.loc[test_index].drop('income_cat', axis=1).to_csv('input.csv', index=False)


    # Now we will seperate the training data ('house_train_set') such that we have predictors and labels
    housing_features = house_train_set.drop('median_house_value', axis=1)
    housing_label = house_train_set['median_house_value']

    # Now we will seperate numerical and categorical columns from the 'housing_features' data
    housing_numerical_attribs = housing_features.drop('ocean_proximity', axis=1).columns.tolist()
    housing_categorical_attribs = ['ocean_proximity']

    # Called the 'build_pipeline' function
    pipeline = build_pipeline(housing_numerical_attribs, housing_categorical_attribs)

    # We get the cleaned, transformed data
    housing_prepared_data = pipeline.fit_transform(housing_features)

    # Trained the model
    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared_data, housing_label)

    # Saved the model and the pipeline
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)


else:
    # Inference

    # Loaded the pipeline and the model
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    # Transformed the input new data and then predictions are made and stored the data (results) to output.csv
    input_data = pd.read_csv('input.csv')
    transformed_input = pipeline.transform(input_data)

    predictions = model.predict(transformed_input)
    input_data['median_house_value'] = predictions
    input_data.to_csv('output.csv', index=False)

    print('Inference is Completed, Results Saved to output.csv')