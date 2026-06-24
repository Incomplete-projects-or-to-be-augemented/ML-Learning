import pandas as pd

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


data = pd.DataFrame({
    "age": [22, 25, 47, 52, 46, 56, 23, 30, 36, 40,
            28, 33, 45, 50, 27, 38, 41, 29, 54, 31],

    "salary": [25000, 32000, 58000, 72000, 62000, 80000, 27000, 35000, 48000, 52000,
               39000, 41000, 61000, 69000, 30000, 50000, 54000, 37000, 76000, 43000],

    "country": ["UK", "UK", "Germany", "France", "Germany", "France", "UK", "Spain", "Spain", "Germany",
                "France", "Spain", "UK", "Germany", "Spain", "France", "UK", "Germany", "France", "Spain"],

    "job_type": ["student", "junior", "senior", "senior", "manager", "manager", "student", "junior", "junior", "senior",
                 "junior", "junior", "manager", "manager", "student", "senior", "senior", "junior", "manager", "junior"]
})


numeric_features = ["age", "salary"]
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


kmeans_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", KMeans(
        n_clusters=3,
        init="k-means++",
        random_state=42,
        n_init=10
    ))
])


kmeans_pipeline.fit(data)


clusters = kmeans_pipeline.named_steps["model"].labels_


data["cluster"] = clusters


print(data)


inertia = kmeans_pipeline.named_steps["model"].inertia_
print("\nInertia:")
print(inertia)


X_transformed = kmeans_pipeline.named_steps["preprocessing"].transform(data.drop("cluster", axis=1))

silhouette = silhouette_score(X_transformed, clusters)
print("\nSilhouette score:")
print(silhouette)