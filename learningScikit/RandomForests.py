from prepData import preprocessor,Pipeline,RandomForestClassifier,GridSearchCV
from prepData import x,y, accuracy_score,confusion_matrix,classification_report

pipe = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])


params = {
    "model__max_depth": [1, 2, 3, 4, None],
    "model__min_samples_split": [2, 5, 10],
    "model__n_estimators": [10,50,100]
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