
from prepData import *

pipe = Pipeline([
    ("preprocessing", preprocessor),
    ("model", KNeighborsClassifier())
])


params = {
    "model__n_neighbors":[1,3,5,7,9],
    "model__weights":["uniform","distance"]
}


grid = GridSearchCV(
    pipe,
    param_grid=params,
    cv=5
)


grid.fit(x, y)


print("Best parameters:")
print(grid.best_params_)

print("\nBest cross-validation score:")
print(grid.best_score_)


pred = grid.predict(x)


print("\nTraining accuracy:")
print(accuracy_score(y, pred))

print("\nConfusion matrix:")
print(confusion_matrix(y, pred))

print("\nClassification report:")
print(classification_report(y, pred))