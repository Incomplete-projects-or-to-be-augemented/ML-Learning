
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin

import joblib
import random
import numpy as np
import pandas as pd

from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance


data = pd.DataFrame({
    "age": [22, 25, 47, 52, 46, 56, 23, 30, 36, 40,
            28, 33, 45, 50, 27, 38, 41, 29, 54, 31],

    "salary": [25000, 32000, 58000, 72000, 62000, 80000, 27000, 35000, 48000, 52000,
               39000, 41000, 61000, 69000, 30000, 50000, 54000, 37000, 76000, 43000],

    "country": ["UK", "UK", "Germany", "France", "Germany", "France", "UK", "Spain", "Spain", "Germany",
                "France", "Spain", "UK", "Germany", "Spain", "France", "UK", "Germany", "France", "Spain"],

    "job_type": ["student", "junior", "senior", "senior", "manager", "manager", "student", "junior", "junior", "senior",
                 "junior", "junior", "manager", "manager", "student", "senior", "senior", "junior", "manager", "junior"],

    "is_human": [1] * 20,

    "random_number": [7, 2, 9, 4, 1, 8, 3, 6, 5, 10,
                      12, 14, 11, 13, 15, 16, 18, 17, 19, 20],

    "bought": [0, 0, 1, 1, 1, 1, 0, 0, 0, 1,
               0, 0, 1, 1, 0, 1, 1, 0, 1, 0],
    "commaed": [
        "25,000", "32,000", "58,000", "72,000", "62,000",
        "80,000", "27,000", "35,000", "48,000", "52,000",
        "39,000", "41,000", "61,000", "69,000", "30,000",
        "50,000", "54,000", "37,000", "76,000", "43,000"
    ]
})


x = data.drop("bought", axis=1)
y = data["bought"]


numeric_features = ["age", "salary", "is_human", "random_number"]
categorical_features = ["country", "job_type"]


numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])


categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])
