from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import random

x = []
n = 10
for i in range(n):
    x.append([((i+1)*10),i+1])
    
y = []


for i in range(n):
    num = random.randint(0, 1)
    y.append(num)
    
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2
)


model = LogisticRegression()

model.fit(X_train, y_train)

prediction = model.predict_proba(X_test)
print(prediction)