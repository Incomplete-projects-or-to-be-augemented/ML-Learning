from prepData import (
    x,
    train_test_split,
    preprocessor,
    Pipeline,
    DBSCAN,
    silhouette_score
)


# -----------------------------
# Split Data
# -----------------------------
x_train, x_test = train_test_split(
    x,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# Build Pipeline
# -----------------------------
dbscan_model = Pipeline([
    ("preprocessing", preprocessor),
    ("model", DBSCAN(
        eps=2,
        min_samples=2
    ))
])


# -----------------------------
# Train
# -----------------------------
dbscan_model.fit(x_train)


# -----------------------------
# Predicted Cluster Labels
# -----------------------------
labels = dbscan_model.named_steps["model"].labels_

print("Cluster Labels")
print(labels)


# -----------------------------
# Number of Clusters
# -----------------------------
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print("\nNumber of Clusters:", n_clusters)


# -----------------------------
# Number of Noise Points
# -----------------------------
n_noise = list(labels).count(-1)

print("Noise Points:", n_noise)


# -----------------------------
# Silhouette Score
# -----------------------------
processed_data = dbscan_model.named_steps["preprocessing"].transform(x_train)

# Silhouette score only works if there are at least 2 clusters
if n_clusters > 1:
    score = silhouette_score(processed_data, labels)
    print("Silhouette Score:", score)
else:
    print("Silhouette Score cannot be calculated.")