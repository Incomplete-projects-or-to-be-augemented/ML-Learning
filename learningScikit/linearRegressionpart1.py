X_train = []
n = 10
for i in range(n):
    X_train.append([(i+1)*10])
    
y_train = []

for i in range(n):
    y_train.append([40+(X_train[i][0]*10)])
    
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

model = LinearRegression()

model.fit(X_train, y_train)
print(f"y = {model.coef_[0][0]}x + {model.intercept_[0]}")

p1 = model.predict([[150],[250],[1000]])

for i in p1:
    print(i[0])
    
