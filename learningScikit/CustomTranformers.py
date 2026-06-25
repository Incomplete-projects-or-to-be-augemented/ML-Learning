from prepData import BaseEstimator, TransformerMixin
from prepData import Pipeline,preprocessor,LinearRegression,x,y

class RemoveCommas(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        # Nothing to learn from the data
        return self

    def transform(self, X):

        # Avoid modifying the original DataFrame
        X = X.copy()

        # Remove commas and convert to float
        X["commaed"] = (
            X["commaed"]
            .str.replace(",", "", regex=False)
            .astype(float)
        )

        return X
    
    

pipeline = Pipeline([
    ("remove_commas", RemoveCommas()),
    ("preprocessing", preprocessor),
    ("model", LinearRegression())
])

fit = pipeline.fit(x,y)



print("\nTransformed Data")
print(fit)


transformer = pipeline.named_steps["remove_commas"]

print("\nTransformer Object")
print(transformer)
