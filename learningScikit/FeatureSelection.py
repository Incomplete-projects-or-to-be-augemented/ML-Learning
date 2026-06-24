from prepData import *

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


best_pipe = grid.best_estimator_

feature_names = best_pipe.named_steps["preprocessing"].get_feature_names_out()

selector = best_pipe.named_steps["selection"]

selected_mask = selector.get_support()

selected_features = feature_names[selected_mask]

print("\nSelected features:")
print(selected_features)


print("\nFeature scores:")
scores = selector.scores_

for feature, score in zip(feature_names, scores):
    print(f"{feature}: {score}")