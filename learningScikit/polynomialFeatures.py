# polynomial_features/polynomial_regression.py

from prepData import x, y,train_test_split,preprocessor

from prepData import Pipeline,PolynomialFeatures, StandardScaler,LinearRegression
from prepData import mean_absolute_error, mean_squared_error, r2_score


# 1. Split data first
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)


# 2. Build polynomial regression pipeline
poly_model = Pipeline([
    ("preprocessing", preprocessor),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])


# 3. Train
poly_model.fit(x_train, y_train)


# 4. Predict
y_pred = poly_model.predict(x_test)


# 5. Evaluate
print("Polynomial Regression Results")
print("-" * 40)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
print("R²:", r2_score(y_test, y_pred))

preprocessor = poly_model.named_steps["preprocessing"]

preprocessor_feature_names = preprocessor.get_feature_names_out()

# 6. Inspect generated polynomial feature names
poly_step = poly_model.named_steps["poly"]

feature_names = poly_step.get_feature_names_out(preprocessor_feature_names)

print("\nGenerated Polynomial Features")
print("-" * 40)

for name in feature_names:
    print(name)