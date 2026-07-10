import torch

# requires_grad flag gets PyToch to keep a record of all the things that happen to this
# this tensor because I'll want it's gradient later
x = torch.tensor(2.0, requires_grad=True)

y = x * 3

# Walk back through the computation graph and calculate the 
# gradient of y with respect to every tensor that has requires_grad=True
y.backward()

print(x.grad)

# Multi variable autograd

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

c = a * b

c.backward()

print(a.grad)
print(b.grad)

# Longer chain
x = torch.tensor(2.0, requires_grad=True)

a = x * 3
b = a + 1
c = b ** 2

c.backward()

print(x.grad)