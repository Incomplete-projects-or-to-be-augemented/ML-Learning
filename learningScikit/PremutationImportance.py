from prepData import preprocessor,RandomForestClassifier,mutual_info_classif,SelectKBest
from prepData import Pipeline,GridSearchCV
from prepData import x,y, accuracy_score,confusion_matrix,classification_report,permutation_importance

pipe = Pipeline([
    ("preprocessing", preprocessor),
    ("selection", SelectKBest(score_func=mutual_info_classif)),
    ("model", RandomForestClassifier(random_state=42))
])


params = {
    "selection__k": [2, 4, 6, "all"],
    "model__n_estimators": [50, 100],
    "model__max_depth": [2, 3, None]
}


grid = GridSearchCV(
    pipe,
    param_grid=params,
    cv=5
)


grid.fit(x, y)

result = permutation_importance(
    grid.best_estimator_,
    x,
    y,
    n_repeats=10,
    random_state=42
)

for feature, importance in zip(x.columns, result.importances_mean):
    print(f"{feature}: {importance}")