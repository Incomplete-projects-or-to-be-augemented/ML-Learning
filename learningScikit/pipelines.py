from prepData import Pipeline,StandardScaler,LogisticRegression
from prepData import random,train_test_split

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
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)