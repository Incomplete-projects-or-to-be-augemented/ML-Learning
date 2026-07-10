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

a.retain_grad()
b.retain_grad()

c.backward()

print(x.grad)
print(a.grad)
print(b.grad)

# noGrad
# no gradient is caluclated. pretty obvious
weight = torch.tensor(5.0, requires_grad=True)

with torch.no_grad():
    prediction = weight * 3

print(prediction)
print(prediction.requires_grad)


# detach 
# Recipe of computation graph ready for backward instead detach removes a step and severs it from the graph
weight = torch.tensor(5.0, requires_grad=True)

result = weight * 3

detached_result = result.detach()

print(result.requires_grad)
print(detached_result.requires_grad)