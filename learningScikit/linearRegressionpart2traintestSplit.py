from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import random

x = []
n = 10
for i in range(n):
    x.append([((i+1)*10),i+1])
    
y = []

for i in range(n):
    y.append([40+(x[i][0]*10) + (x[i][1]*5)])
    
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2
)

model = LinearRegression()

model.fit(X_train, y_train)
pred = model.predict(X_test)

for i in pred:
    print(i[0])
    
score = model.score(X_test,y_test)
print(score)