from sklearn.preprocessing import StandardScaler
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
    

scaler = StandardScaler()
stats = scaler.fit(x)

X_scaled = stats.transform(x)

print(stats)
print(X_scaled)