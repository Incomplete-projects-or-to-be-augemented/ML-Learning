import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score


df = pd.DataFrame({
    "Age": [21, 30, 42],
    "Salary": [30000, 60000, 90000],
    "Country": ["UK", "France", "Germany"],
    "BuysProduct": [0, 1, 1]
})


x = df.drop(columns=["BuysProduct"])
y = df["BuysProduct"]

numeric_features = ["Age", "Salary"]

categorical_features = ["Country"]



numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

pipe = Pipeline([
    ("preprocessing", preprocessor),
    ("pca", PCA(n_components=2)),
    ("model", LogisticRegression())
])

pipe.fit(x, y)

print(pipe.predict(x))

scores = cross_val_score(pipe, x, y, cv=2)


print(scores)
print(scores.mean())