from prepData import *

pipe = Pipeline([
    ("preprocessing", preprocessor),
    ("model", GradientBoostingClassifier())
])


params = {
    "model__n_estimators": [50,100,200],
    "model__learning_rate": [0.01,0.1,0.2],
    "model__max_depth": [2,3,4]
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