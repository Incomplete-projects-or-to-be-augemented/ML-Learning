from prepData import *


kmeans_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", KMeans(
        n_clusters=3,
        init="k-means++",
        random_state=42,
        n_init=10
    ))
])


kmeans_pipeline.fit(data)


clusters = kmeans_pipeline.named_steps["model"].labels_


data["cluster"] = clusters


print(data)


inertia = kmeans_pipeline.named_steps["model"].inertia_
print("\nInertia:")
print(inertia)


X_transformed = kmeans_pipeline.named_steps["preprocessing"].transform(data.drop("cluster", axis=1))

silhouette = silhouette_score(X_transformed, clusters)
print("\nSilhouette score:")
print(silhouette)