from prepData import preprocessor,Pipeline,PCA,LogisticRegression
from prepData import cross_val_score
from prepData import x,y

pipe = Pipeline([
    ("preprocessing", preprocessor),
    ("pca", PCA(n_components=2)),
    ("model", LogisticRegression())
])

pipe.fit(x, y)

print(pipe.predict(x))

scores = cross_val_score(pipe, x, y, cv=2)


print(scores)
print(scores.mean())