import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from prepData import pd, joblib

model = joblib.load("customer_model.pkl")


new_customers = pd.DataFrame({
    "age": [24, 52, 35],
    "salary": [28000, 76000, 45000],
    "country": ["UK", "France", "Spain"],
    "job_type": ["student", "manager", "junior"],
    "is_human": [1, 1, 1],
    "random_number": [4, 15, 9]
})


predictions = model.predict(new_customers)


print("Predictions:")
print(predictions)


for customer, prediction in zip(new_customers.to_dict(orient="records"), predictions):
    print(customer, "=> bought:", prediction)