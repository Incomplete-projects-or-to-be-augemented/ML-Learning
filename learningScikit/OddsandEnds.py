"""
Sklearn Miscellaneous Utilities

Topics:
- make_pipeline()
- set_params()
- Parameter naming
- make_regression()
- make_classification()
- make_blobs()
"""

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import (
    make_regression,
    make_classification,
    make_blobs
)


# ======================================================
# 1. Synthetic Dataset Generation
# ======================================================

print("===== make_regression =====")

X_reg, y_reg = make_regression(
    n_samples=100,
    n_features=8,
    noise=0.1,
    random_state=42
)

print("X shape:", X_reg.shape)
print("y shape:", y_reg.shape)


print("\n===== make_classification =====")

X_class, y_class = make_classification(
    n_samples=100,
    n_features=10,
    n_classes=2,
    random_state=42
)

print("X shape:", X_class.shape)
print("y shape:", y_class.shape)


print("\n===== make_blobs =====")

X_blob, y_blob = make_blobs(
    n_samples=100,
    centers=3,
    n_features=2,
    cluster_std=1.5,
    random_state=42
)

print("X shape:", X_blob.shape)
print("Unique Clusters:", set(y_blob))


# ======================================================
# 2. Pipeline()
# ======================================================

print("\n===== Pipeline =====")

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestRegressor())
])

print(pipeline)


# ======================================================
# 3. make_pipeline()
# ======================================================

print("\n===== make_pipeline =====")

auto_pipeline = make_pipeline(
    StandardScaler(),
    RandomForestRegressor()
)

print(auto_pipeline)


print("\nAutomatically generated step names:")

for name in auto_pipeline.named_steps:
    print(name)


# ======================================================
# 4. set_params()
# ======================================================

print("\n===== set_params =====")

pipeline.set_params(
    model__n_estimators=200
)

print(
    "Updated n_estimators:",
    pipeline.named_steps["model"].n_estimators
)


# ======================================================
# 5. Parameter Naming Convention
# ======================================================

print("\n===== Parameter Names =====")

params = pipeline.get_params()

for name in sorted(params):
    if "model__" in name:
        print(name)


# ======================================================
# 6. Accessing Objects
# ======================================================

print("\n===== named_steps =====")

scaler = pipeline.named_steps["scaler"]
model = pipeline.named_steps["model"]

print(type(scaler))
print(type(model))


# ======================================================
# 7. Fit Example
# ======================================================

print("\n===== Training =====")

pipeline.fit(X_reg, y_reg)

predictions = pipeline.predict(X_reg[:5])

print("Predictions:")
print(predictions)