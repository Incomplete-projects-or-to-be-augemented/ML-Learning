from lightgbm import LGBMClassifier, LGBMRegressor

# LightGBM Classifier

from prepData import Pipeline
from prepData import train_test_split, accuracy_score, confusion_matrix, classification_report
from prepData import x, y, preprocessor


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
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            max_depth=-1,
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