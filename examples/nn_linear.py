import nujo as nj
import nujo.nn as nn
import nujo.optim as optim

# Define the net and optimizer
net = nn.Linear(3, 6) >> nn.Linear(6, 2) >> nn.Linear(2, 1)
print('Defined net:', net)
optimizer = optim.Adam(net.parameters, lr=0.1)


# Training loop
def train(net, x, y, num_epochs):
    for epoch in range(1, num_epochs + 1):

        # Forward
        output = net(x)
        # Compute Loss
        loss = nj.mean((output - y)**2)

        # Print the loss every 10th epoch for monitoring
        if epoch % 10 == 0:
            print('EPOCH:', epoch, '| LOSS: ', loss.value)

        # Backprop
        loss.backward()

        # Update
        optimizer.step()

        # Zero grad
        optimizer.zero_grad()


if __name__ == '__main__':
    # Create example data
    x = nj.rand(30, 3, diff=False)
    y = x @ [[2], [3], [4]] - 10

    # Train
    train(net, x, y, 100)
