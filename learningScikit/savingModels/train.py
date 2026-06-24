import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from prepData import x, y, preprocessor,joblib
from prepData import RandomForestClassifier,GridSearchCV,Pipeline


pipe = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])


params = {
    "model__n_estimators": [50, 100],
    "model__max_depth": [2, 3, None],
    "model__min_samples_split": [2, 5]
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


joblib.dump(
    grid.best_estimator_,
    "customer_model.pkl"
)


print("\nModel saved as customer_model.pkl")