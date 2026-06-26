from lightgbm import LGBMClassifier, LGBMRegressor

# LightGBM Classifier

from prepData import Pipeline
from prepData import train_test_split, accuracy_score, confusion_matrix, classification_report
from prepData import x, y, preprocessor,pd


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
        ("lightgbm", LGBMClassifier(
            boosting_type="goss",
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=7,
            max_depth=3,
            min_child_samples=1,
            min_data_in_bin=1,
            random_state=42,
            verbose=-1
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


lgbm = model.named_steps["lightgbm"]

print(lgbm.feature_importances_)
importance = pd.DataFrame({
    "Feature": model.named_steps["preprocessing"].get_feature_names_out(),
    "Importance": lgbm.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

print(importance)