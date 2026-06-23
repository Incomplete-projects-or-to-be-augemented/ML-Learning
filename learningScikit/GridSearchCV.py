from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
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
                 "junior", "junior", "manager", "manager", "student", "senior", "senior", "junior", "manager", "junior"],
    
    "bought": [0, 0, 1, 1, 1, 1, 0, 0, 0, 1,
               0, 0, 1, 1, 0, 1, 1, 0, 1, 0]
})



x = data.drop("bought", axis=1)
y = data["bought"]

print(y.value_counts())


numeric_features = ["age", "salary"]

categorical_features = ["country","job_type"]

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





params = {
    "pca__n_components": [1,2],
    "model__C": [0.1,1,10]
}


grid = GridSearchCV(
    pipe,
    param_grid=params,
    cv=5
)