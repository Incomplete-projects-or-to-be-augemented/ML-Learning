import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier


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
               0, 0, 1, 1, 0, 1, 1, 0, 1, 0]
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
