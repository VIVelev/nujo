import numpy as np
import nujo as nj

# Define the net
net = nj.Linear(3, 6) >> nj.Linear(6, 2) >> nj.Linear(2, 1)

# Training loop
def train(net, x, y, num_epochs, lr):
    for epoch in range(1, num_epochs+1):

        # Forward
        output = net(x)
        # Compute Loss
        loss = (1/x.shape[0])*(output-y)**2
        
        # Print the loss every 10th epoch for monitoring
        if epoch % 10 == 0:
            print('EPOCH:', epoch, '| LOSS: ', np.mean(loss.value))
        
        # Backprop
        loss.backward()
        
        # Update
        with nj.no_diff():
            for layer in net:
                layer.weights -= lr*layer.weights.grad
                layer.bias -= lr*layer.bias.grad
        
        # Zero grad
        for layer in net:
            layer.weights.zero_grad()
            layer.bias.zero_grad()

if __name__ == '__main__':
    # Create example data
    x = np.random.rand(30, 3)
    y = x@[[2], [3], [4]] - 10
    x, y = nj.Constant(x), nj.Constant(y)

    # Train
    train(net, x, y, 100, 0.1)
