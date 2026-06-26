from prepData import BaseEstimator,np,x,y,train_test_split,accuracy_score

class AlwaysPass(BaseEstimator):

    def fit(self, X, y):
        return self

    def predict(self, X):

        return np.ones(len(X))

X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

model = AlwaysPass()

model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(predictions)
# Example 2

# -----------------------------
# Dummy Data
# -----------------------------
X = np.array([
    [1], [2], [3], [4], [5],
    [6], [7], [8], [9], [10]
])

Y = np.array([
    0, 0, 0, 0, 1,
    1, 1, 1, 1, 1
])

# -----------------------------
# Split Data
# -----------------------------
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# Train
# -----------------------------
model2 = AlwaysPass()

model2.fit(X_train2, y_train2)


# -----------------------------
# Predict
# -----------------------------
predictions2 = model2.predict(X_test2)


# -----------------------------
# Evaluate
# -----------------------------
print("Predictions:", predictions2)
print("Accuracy:", accuracy_score(y_test2, predictions2))