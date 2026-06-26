# Extreme Gradient Boosting

from prepData import Pipeline,XGBClassifier
from prepData import train_test_split,accuracy_score,confusion_matrix,classification_report
from prepData import x,y,preprocessor,pd


x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

xgb = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=3,
    objective="binary:logistic",
    eval_metric="logloss",
    early_stopping_rounds=10,
    random_state=42
)


model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("xgboost", xgb)
    ]
)

x_train_processed = preprocessor.fit_transform(x_train)
x_val_processed = preprocessor.transform(x_val)
x_test_processed = preprocessor.transform(x_test)

xgb.fit(
    x_train_processed,
    y_train,
    eval_set=[(x_val_processed, y_val)]
)
y_pred = xgb.predict(x_test_processed)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


xgb = model.named_steps["xgboost"]

feature_importance = pd.DataFrame({
    "Feature": model.named_steps["preprocessing"].get_feature_names_out(),
    "Importance": xgb.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)

print("\nBest Iteration:")
print(xgb.best_iteration)

from prepData import RandomizedSearchCV

tuning_xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

param_grid = {
    "n_estimators": [100, 200, 500],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

search = RandomizedSearchCV(
    estimator=tuning_xgb,
    param_distributions=param_grid,
    n_iter=20,
    cv=5,
    scoring="f1",
    random_state=42
)

search.fit(x_train_processed, y_train)

print(search.best_params_)
print(search.best_score_)