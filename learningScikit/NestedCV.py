from prepData import Pipeline,preprocessor,RandomForestClassifier,RandomizedSearchCV
from prepData import x,y,cross_val_score

param_distributions = {
    "model__n_estimators": [10, 20, 50],
    "model__max_depth": [2, 3, None],
    "model__min_samples_split": [2, 5]
}

pipe = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])

inner_search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_distributions,
    n_iter=4,          # only tests 4 random combos
    cv=2,              # inner CV
    random_state=42
)


nested_scores = cross_val_score(
    inner_search,
    x,
    y,
    cv=3               # outer CV
)


print("Nested CV scores:")
print(nested_scores)

print("\nAverage nested CV score:")
print(nested_scores.mean())