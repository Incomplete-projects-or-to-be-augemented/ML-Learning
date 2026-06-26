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


model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("xgboost", XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42
        ))
    ]
)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

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

print(feature_importance)
