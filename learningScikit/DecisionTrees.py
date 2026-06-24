
from prepData import *

pipe = Pipeline([
    ("preprocessing", preprocessor),
    ("model", DecisionTreeClassifier(random_state=42))
])


params = {
    "model__max_depth": [1, 2, 3, 4, None],
    "model__min_samples_split": [2, 5, 10],
    "model__criterion": ["gini", "entropy"]
}


grid = GridSearchCV(
    pipe,
    param_grid=params,
    cv=5
)


grid.fit(X, y)


print("Best parameters:")
print(grid.best_params_)

print("\nBest cross-validation score:")
print(grid.best_score_)


pred = grid.predict(X)


print("\nTraining accuracy:")
print(accuracy_score(y, pred))

print("\nConfusion matrix:")
print(confusion_matrix(y, pred))

print("\nClassification report:")
print(classification_report(y, pred))