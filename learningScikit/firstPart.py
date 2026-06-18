X_train = []

for i in range(4):
    X_train.append([(i+1)*100])
    
y_train = []

for i in range(4):
    y_train.append([X_train[i][0]//2])
    
print(y_train)
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)
print(f"y = {model.coef_[0][0]}x + {model.intercept_[0]}")

p1 = model.predict([[150],[250],[1000]])

for i in p1:
    print(i[0])