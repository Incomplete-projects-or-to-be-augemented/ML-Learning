from prepData import *

pipe = Pipeline([
    ("preprocessing", preprocessor),
    ("pca", PCA(n_components=2)),
    ("model", LogisticRegression())
])


scores = cross_val_score(pipe, x, y, cv=5)

print(scores)
print(scores.mean())