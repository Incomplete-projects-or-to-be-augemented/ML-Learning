from prepData import x, preprocessor
from prepData import Pipeline, RandomForestClassifier, RandomizedSearchCV
from prepData import train_test_split
from prepData import accuracy_score, confusion_matrix, classification_report


# Artificial imbalanced target for learning:
# 0 = did not buy
# 1 = bought
# Only 4 buyers, 16 non-buyers
y_imbalanced = [
    0, 0, 1, 1, 0,
    1, 0, 0, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 0, 0, 0
]


print("Class balance:")
print("0 count:", y_imbalanced.count(0))
print("1 count:", y_imbalanced.count(1))


X_train, X_test, y_train, y_test = train_test_split(
    x,
    y_imbalanced,
    test_size=0.25,
    stratify=y_imbalanced,
    random_state=42
)


pipe = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(
        class_weight="balanced",
        random_state=42
    ))
])


params = {
    "model__n_estimators": [50, 100],
    "model__max_depth": [2, 3, None],
    "model__min_samples_split": [2, 5]
}


search = RandomizedSearchCV(
    pipe,
    param_distributions=params,
    n_iter=4,
    cv=2,
    scoring="f1",
    random_state=42
)


search.fit(X_train, y_train)


print("\nBest parameters:")
print(search.best_params_)

print("\nBest CV F1 score:")
print(search.best_score_)


pred = search.predict(X_test)


print("\nAccuracy:")
print(accuracy_score(y_test, pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, pred))

print("\nClassification report:")
print(classification_report(y_test, pred))