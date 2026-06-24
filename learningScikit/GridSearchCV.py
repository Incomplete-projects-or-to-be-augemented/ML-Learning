from prepData import *

pipe = Pipeline([
    ("preprocessing", preprocessor),
    ("pca", PCA(n_components=2)),
    ("model", LogisticRegression())
])





params = {
    "pca__n_components": [1,2],
    "model__C": [0.1,1,10]
}


grid = GridSearchCV(
    pipe,
    param_grid=params,
    cv=5
)

grid.fit(x, y)
pred = grid.predict(x)

print(grid.best_params_)
print(grid.best_score_)
print(accuracy_score(y, pred))
print(confusion_matrix(y, pred))
print(classification_report(y, pred))