# catboostClassifier.py

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from prepData import x, y


# 1. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 2. Identify categorical columns
cat_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

print("Categorical features:", cat_features)


# 3. Create CatBoost Pool objects
# Pool is CatBoost's native dataset wrapper.
train_pool = Pool(
    data=X_train,
    label=y_train,
    cat_features=cat_features
)

test_pool = Pool(
    data=X_test,
    label=y_test,
    cat_features=cat_features
)


# 4. Create model
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=42,
    verbose=50
)


# 5. Train model
model.fit(
    train_pool,
    eval_set=test_pool,
    use_best_model=True
)


# 6. Predict
y_pred = model.predict(test_pool)


# 7. Evaluate
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 8. Feature importance
print("\nFeature Importances:")
for feature, importance in zip(X_train.columns, model.feature_importances_):
    print(f"{feature}: {importance:.4f}")


# 9. Save model
model.save_model("catboost_classifier_model.cbm")