import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


data = pd.DataFrame({
    "age": [22, 25, 47, 52, 46, 56, 23, 30, 36, 40,
            28, 33, 45, 50, 27, 38, 41, 29, 54, 31],
    
    "salary": [25000, 32000, 58000, 72000, 62000, 80000, 27000, 35000, 48000, 52000,
               39000, 41000, 61000, 69000, 30000, 50000, 54000, 37000, 76000, 43000],
    
    "country": ["UK", "UK", "Germany", "France", "Germany", "France", "UK", "Spain", "Spain", "Germany",
                "France", "Spain", "UK", "Germany", "Spain", "France", "UK", "Germany", "France", "Spain"],
    
    "job_type": ["student", "junior", "senior", "senior", "manager", "manager", "student", "junior", "junior", "senior",
                 "junior", "junior", "manager", "manager", "student", "senior", "senior", "junior", "manager", "junior"],
    
    "bought": [0, 0, 1, 1, 1, 1, 0, 0, 0, 1,
               0, 0, 1, 1, 0, 1, 1, 0, 1, 0]
})


X = data.drop("bought", axis=1)
y = data["bought"]


numeric_features = ["age", "salary"]
categorical_features = ["country", "job_type"]


numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean"))
])


categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])


pipe = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])


params = {
    "model__max_depth": [1, 2, 3, 4, None],
    "model__min_samples_split": [2, 5, 10],
    "model__n_estimators": [10,50,100]
}


grid = GridSearchCV(
    pipe,
    param_grid=params,
    cv=5
)


grid.fit(X, y)


print("Best parameters:")
print(grid.best_params_)

print("\nBest cross-validation score:")
print(grid.best_score_)


pred = grid.predict(X)


print("\nTraining accuracy:")
print(accuracy_score(y, pred))

print("\nConfusion matrix:")
print(confusion_matrix(y, pred))

print("\nClassification report:")
print(classification_report(y, pred))