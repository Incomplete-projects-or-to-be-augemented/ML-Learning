from prepData import preprocessor,PCA,LogisticRegression
from prepData import Pipeline,RandomizedSearchCV
from prepData import x,y, accuracy_score,confusion_matrix,classification_report

pipe = Pipeline([
    ("preprocessing", preprocessor),
    ("pca", PCA(n_components=2)),
    ("model", LogisticRegression())
])

params = {
    "pca__n_components": [1,2],
    "model__C": [0.1,1,10]
}


search = RandomizedSearchCV(
    pipe,
    param_distributions=params,
    n_iter=3,
    cv=5,
    random_state=42
)

search.fit(x, y)
pred = search.predict(x)

print(search.best_params_)
print(search.best_score_)
print(accuracy_score(y, pred))
print(confusion_matrix(y, pred))
print(classification_report(y, pred))