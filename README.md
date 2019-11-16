# Neural Dojo: A Reverse-mode Automatic Differentiation library for Neural Networks

### Perceptron implementation:

```python
import numpy as np
import nujo as nj

##########
# Create example data
M = 500
x = np.array([np.random.rand(1, 1) for _ in range(M)])
y = x@[[4]] - 2

x, y = nj.Constant(x), nj.Constant(y)

##########
# Initialize weights
m, b = nj.Variable(np.random.randn(1, 1)), nj.Variable(np.random.randn(1, 1))

##########
# Training loop

# Number of epochs and learning rate
n_epochs = 10_000
lr = 10

# Loop
for epoch in range(1, n_epochs+1):
    
    # Forward
    output = x@m + b
    # Compute loss
    loss = 1/M*((output-y)**2)
    
    # Backprop errors
    loss.backward()

    # Print the loss every 100th epoch for monitoring
    if epoch % 100 == 0:
        print('EPOCH:', epoch, '| LOSS: ', np.mean(loss.value))
    
    # Gradient Descent update rule
    with nj.no_diff():
        m -= lr*np.mean(m.grad)
        b -= lr*np.mean(b.grad)
    
    # Zero grad
    m.zero_grad()
    b.zero_grad()

##########
# Test with trained weights
y_pred = x[:50]@m + b
```
