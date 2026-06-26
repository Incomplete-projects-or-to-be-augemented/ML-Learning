from prepData import (
    x,
    train_test_split,
    preprocessor,
    Pipeline,
    AgglomerativeClustering,
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
agg_model = Pipeline([
    ("preprocessing", preprocessor),
    ("model", AgglomerativeClustering(
        n_clusters=3,
        linkage="ward"
    ))
])


# -----------------------------
# Fit
# -----------------------------
agg_model.fit(x_train)


# -----------------------------
# Cluster Labels
# -----------------------------
labels = agg_model.named_steps["model"].labels_

print("Cluster Labels")
print(labels)


# -----------------------------
# Number of Clusters
# -----------------------------
n_clusters = len(set(labels))

print("\nNumber of Clusters:", n_clusters)


# -----------------------------
# Silhouette Score
# -----------------------------
processed_data = agg_model.named_steps["preprocessing"].transform(x_train)

if n_clusters > 1:
    score = silhouette_score(processed_data, labels)
    print("Silhouette Score:", score)
else:
    print("Silhouette Score cannot be calculated.")